from projects.sa2va.models.mllm.internvl_train import InternVLMLLM_Train
from projects.sa2va.models.compression_method import TransformerScorer, topk

import torch
import torch.nn.functional as F
import torch.distributed
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from mmengine import print_log
from mmengine.model import BaseModel
from xtuner.registry import BUILDER


from vlm.utils import decode_tokens_with_counts


class InternVLMLLM_Train_Dev(InternVLMLLM_Train):
    def __init__(
        self,
        model_path: str,
        freeze_llm: bool = False,
        freeze_visual_encoder: bool = False,
        llm_lora: Optional[dict] = None,
        visual_encoder_lora: Optional[dict] = None,
        quantization_vit: bool = False,
        quantization_llm: bool = False,
        pretrained_pth: Optional[str] = None,
        use_flash_attn: bool = True,
        # ------ADD------
        # 用mmengine的BUILD自动装配
        importance_scorer: Optional[dict] = None,
        budgets: float = None,
    ):
        assert importance_scorer is not None, "importance_scorer should not be none."
        assert budgets is not None, "budgets should not be none."

        super().__init__(
            model_path,
            freeze_llm,
            freeze_visual_encoder,
            llm_lora,
            visual_encoder_lora,
            quantization_vit,
            quantization_llm,
            pretrained_pth,
            use_flash_attn,
        )
        self.model.importance_scorer = BUILDER.build(importance_scorer)
        self.budgets = budgets

    # 重写state_dict方法，添加打分器的参数
    def state_dict(self, *args, **kwargs):
        to_return = super().state_dict(*args, **kwargs)
        # 保留scorer的参数
        scorer_dict = self.model.importance_scorer.state_dict()
        to_return.update(
            {"mllm.model.importance_scorer." + k: v for k, v in scorer_dict.items()}
        )
        return to_return

    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vp_overall_mask: Optional[torch.Tensor] = None,
        prompt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Enhanced LLM forward pass with visual prompt support.
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        # Process inputs
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        ).clone()

        # Extract and process visual features
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = vit_embeds.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1

        # Process visual prompts if available
        visual_embeds = self._process_visual_prompts(
            vit_embeds, vp_overall_mask, prompt_masks, image_flags, C
        )

        # 原模型做法：
        # 1. 动态分割导致每个batch里小图数量不一样，导致，如果直接用b，n表示，n不同无法reshape
        # 2. 模型给的方法是把所有batch的小图拼在一起，此时就变成了all_small_pic_num, patch_num_per_small_pic, hidden_size的维度了
        # 3. 然后模型把input_embeds也reshape成了b*n, hidden_size的维度，这样就能把动态数量的小图一次性赋值到input_embeds里
        # 使用LIS出现的问题：
        # 1. 目前加的打分机制必须针对于每个batch的所有小图，不能把所有batch拼一起，必须每个batch分开打分，最后再把结果拼一起
        # 2. 而且在推理过程也是一样，要先按batch分开打分再合并，我需要记住每个batch的小图数量
        # 3. 
        # 问题：
        # 训练加入纯对话后，没有了图片的话应该怎么设置这个模块的损失？
        # -------创新代码加在这里------
        if visual_embeds.shape[0] != 0:
            hidden_states = visual_embeds
            hidden_states_unsqueezed = hidden_states.unsqueeze(0)
            learned_scores = self.model.importance_scorer(hidden_states_unsqueezed).squeeze(
                0
            )
            total_tokens = learned_scores.shape[0]
            k = int(total_tokens * self.budgets)
            img_mask = topk(learned_scores.unsqueeze(0), k).squeeze(0)
            img_mask_expanded = img_mask.unsqueeze(1).expand(
                -1, hidden_states_unsqueezed.shape[-1]
            )
            hidden_states_new = img_mask_expanded * hidden_states
            hidden_states_new = hidden_states_new.type(hidden_states.dtype)

            with torch.no_grad():
                constraint_topk_indices = learned_scores.topk(k, dim=0).indices
                constraint_img_mask = torch.zeros_like(
                    learned_scores, device=learned_scores.device
                )
                constraint_img_mask.scatter_(
                    dim=-1, index=constraint_topk_indices, value=1.0
                )
            # 计算出打分损失
            scorer_loss = F.binary_cross_entropy(img_mask, constraint_img_mask)
            visual_embeds = hidden_states_new
        # --------创新代码结束-----------
        else:
            # 这里改成tensor，而且不要连带着前面的参数
            scorer_loss = torch.zeros((), device=visual_embeds.device, dtype=visual_embeds.dtype)

        # Embed visual features into text embeddings
        input_embeds = self._embed_visual_features(
            input_embeds, input_ids, visual_embeds, B, N, C
        )

        input_embeds = input_embeds.reshape(B, N, C)

        # Forward through language model
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Compute loss if labels provided
        loss = (
            self._compute_loss(outputs.logits, labels) if labels is not None else None
        )

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastDev(
            # llm的损失
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # ADD
            scorer_loss=scorer_loss,
        )


from dataclasses import dataclass


@dataclass
class CausalLMOutputWithPastDev(CausalLMOutputWithPast):
    # 打分损失
    scorer_loss: Optional[torch.FloatTensor] = None
