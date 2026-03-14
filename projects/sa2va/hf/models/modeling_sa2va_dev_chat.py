from .modeling_sa2va_chat import Sa2VAChatModel
from .configuration_sa2va_dev_chat import Sa2VADevChatConfig
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)

# LIS
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerScorer(nn.Module):
    """
    Lightweight Transformer Scorer using simplified attention for importance scoring.
    Initializes outputs close to zero to minimally interfere with the original attention_sum.
    """

    def __init__(
        self, in_features: int, hidden_dim: int = 1792, init_scale: float = 0.0001
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Lightweight projection layers for Key and Query
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.q_proj = nn.Linear(in_features, hidden_dim)

        # Initialize all weights to produce near-zero output
        self._init_near_zero(init_scale)

    def _init_near_zero(self, scale: float = 0.0001):
        """Initialize parameters with small values to ensure near-zero output."""
        # Initialize k_proj and q_proj weights with small scale
        nn.init.normal_(self.k_proj.weight, std=scale)
        nn.init.zeros_(self.k_proj.bias)

        nn.init.normal_(self.q_proj.weight, std=scale)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates importance scores via simplified self-attention.

        Args:
            x (torch.Tensor): Visual tokens of shape [B, N, D]
                             (B: batch size, N: token count, D: embedding dim)
        Returns:
            torch.Tensor: Learned importance scores, shape [B, N]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Generate Key and Query representations
        k = self.k_proj(x)  # [B, N, hidden_dim]
        q = self.q_proj(x)  # [B, N, hidden_dim]

        # Simplified self-attention: compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (
            self.hidden_dim**0.5
        )  # [B, N, N]
        # Scores are the mean of attention weights across the attention dimension
        scores = attn_weights.mean(dim=-1)

        return scores


class Sa2VADevChatModel(Sa2VAChatModel):
    config_class = Sa2VADevChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
        "SAM2",
        "TransformerScorer",
    ]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Sa2VADevChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config, vision_model, language_model, use_flash_attn)
        self.importance_scorer = TransformerScorer(config.hidden_size)
        self.budgets = config.budgets
    
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1

                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None
            
            # add there
            # -------ADD START------
            # <1> 通过LIS并直接topk，得到需要的视觉特征索引
            hidden_states = vp_embeds
            total_token_num = hidden_states.shape[0]
            hidden_states_unsqueezed = hidden_states.unsqueeze(0)
            # detach：复制一份参数，但其不参与梯度计算
            learned_scores: torch.Tensor = self.importance_scorer(hidden_states_unsqueezed.detach()).squeeze(0) 
            dominant_num = max(1, int(total_token_num * self.budgets))
            # tensor自带的topk方法，索引是按分数从大到小排列的（会打乱原有顺序）
            # 因此在后面还要sort一次恢复原有顺序
            all_indices = learned_scores.topk(dominant_num, dim=0).indices   # get topk indices
            all_indices = all_indices.sort().values
            # 布尔索引只保留topk的token，且相对顺序不变
            hidden_states_new = hidden_states[all_indices,:]

            # <2> 再加上文本embeds的索引，得到seleted_indices，之后的positionid和attention_mask只用索引里的
            # from 1:
            _, all_indices, _ = hidden_states_new, all_indices, hidden_states_new.shape[0]
            origin_image_indices = torch.where(input_ids == self.img_context_token_id)[1]
            # 只包含topk后的index
            retain_image_indices = origin_image_indices[all_indices]
            origin_text_indices = torch.where(input_ids != self.img_context_token_id)[1]
            # topk的index和所有text的index，归为正确顺序
            combined_indices = torch.cat((retain_image_indices, origin_text_indices))
            selected_indices, _ = torch.sort(combined_indices)
            # --------ADD END-----------

            # 按原逻辑照常拼接
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # -------ADD START------
        # 只是筛选视觉特征，因此只考虑输入有图像的情况
        if pixel_values is not None:
            if isinstance(self.language_model, Qwen2ForCausalLM):
                
                pass
            pass
        # -------ADD END------

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
