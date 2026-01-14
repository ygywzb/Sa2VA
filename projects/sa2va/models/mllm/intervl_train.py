# Copyright (c) OpenMMLab. All rights reserved.
# 功能几乎和xtuner上的一样，但是添加了一个禁用falsh-attn的选项
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast

from xtuner.registry import BUILDER
from xtuner.model.utils import (find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, make_inputs_require_grad)


class InternVL_V1_5_Train(BaseModel):

    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 use_flash_attn=True):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        self.model = AutoModel.from_pretrained(
            model_path,
            # add
            use_flash_attn=use_flash_attn,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            config=config,
            trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(self.model.language_model,
                                                   lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model,
                                                 lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.vision_model.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.language_model.' in k
            })
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.mlp1.' in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat([
                image.to(self.model.vision_model.dtype)
                for image in pixel_values
            ],
                                      dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        # Directly calling this code in LORA fine-tuning
        # will result in an error,so we must rewrite it.
        # TODO: Once the official is fixed, we can remove it.
        # outputs = self.model(input_ids=input_ids,
        #                      position_ids=position_ids,
        #                      attention_mask=attention_mask,
        #                      image_flags=image_flags,
        #                      pixel_values=concat_images,
        #                      labels=labels,
        #                      use_cache=use_cache)
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, '
                  f'images per sample: {vit_batch_size / B}, '
                  f'dynamic token length: {N}')
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        try:
            input_embeds[
                selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                    -1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape='
                  f'{input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[
                selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

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
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


import torch
import torch.distributed
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from mmengine import print_log

from vlm.utils import decode_tokens_with_counts


# 使用use-flash-attn的参数被xtuner的这个父类写死了
# 接下来要写一个继承这个父类的类改写构造方法，然后让这个类去继承我改构造的子类
class InternVLMLLM_Train(InternVL_V1_5_Train):
    def __init__(self,
                 model_path: str,
                 freeze_llm: bool = False,
                 freeze_visual_encoder: bool = False,
                 llm_lora: Optional[dict] = None,
                 visual_encoder_lora: Optional[dict] = None,
                 quantization_vit: bool = False,
                 quantization_llm: bool = False,
                 pretrained_pth: Optional[str] = None,
                 use_flash_attn: bool = True
    ):
        """
        Initialize InternVL MLLM with enhanced capabilities.
        
        Args:
            model_path: Path to the pretrained model
            freeze_llm: Whether to freeze the language model
            freeze_visual_encoder: Whether to freeze the visual encoder
            llm_lora: LoRA configuration for language model
            visual_encoder_lora: LoRA configuration for visual encoder
            quantization_vit: Whether to quantize vision transformer
            quantization_llm: Whether to quantize language model
            pretrained_pth: Path to additional pretrained weights
        """
        
        # Initialize parent class with core parameters
        assert pretrained_pth is None, f"{pretrained_pth} should be none please use model_path instead."
        
        super().__init__(
            model_path=model_path,
            freeze_llm=freeze_llm,
            freeze_visual_encoder=freeze_visual_encoder,
            llm_lora=None, # override the lora config
            visual_encoder_lora=None, # override the lora config
            quantization_vit=quantization_vit,
            quantization_llm=quantization_llm,
            pretrained_pth=None, # Do not use this. 
            use_flash_attn=use_flash_attn,
        )


        self.llm_lora_config = llm_lora
        self.visual_encoder_lora_config = visual_encoder_lora


        self.tokenizer = None

    def add_special_tokens(self, tokenizer, special_tokens: List[str]) -> None:
        """Add special tokens to the tokenizer and resize embeddings if needed."""
        print_log(f'{self.__class__.__name__}:add_special_tokens [Before] The total number of tokens is now {len(tokenizer)}', logger='current')
        num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(tokenizer))
            print_log(f'{self.__class__.__name__}:add_special_tokens Added {num_new_tokens} special tokens', logger='current')
            print_log(f'{self.__class__.__name__}:add_special_tokens [After] The total number of tokens is now {len(tokenizer)}', logger='current')
        self.tokenizer = tokenizer


    def manual_prepare_llm_for_lora(self):
        self._prepare_llm_for_lora(self.llm_lora_config)


    def get_embedding_size(self):
        return self.model.config.llm_config.hidden_size


    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, data: dict, data_samples=None, mode: str = 'loss') -> dict:
        """
        Enhanced forward pass with better visual prompt support.
        
        Args:
            data: Input data dictionary containing pixel_values, input_ids, etc.
            data_samples: Additional data samples (unused)
            mode: Forward mode (unused, kept for compatibility)
            
        Returns:
            Dictionary containing model outputs
        """
        # Process pixel values
        pixel_values = data['pixel_values']
        if isinstance(pixel_values, list) or pixel_values.ndim == 5:
            if isinstance(pixel_values, list):
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            concat_images = torch.cat([
                image.to(self.model.vision_model.dtype) for image in pixel_values
            ], dim=0)
        else:
            raise NotImplementedError("Unsupported pixel_values format")

        # Extract other inputs
        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']
        
        # Determine which positions have images
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        # Get visual prompt masks if available
        vp_overall_mask = data.get('vp_overall_mask', None)
        prompt_masks = data.get('prompt_masks', None)

        # Forward through the model
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )
        
        return outputs

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
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # Process inputs
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        # Extract and process visual features
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = vit_embeds.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)


        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, '
                  f'images per sample: {vit_batch_size / B}, '
                  f'dynamic token length: {N}')
        self._count += 1

        # Process visual prompts if available
        visual_embeds = self._process_visual_prompts(
            vit_embeds, vp_overall_mask, prompt_masks, image_flags, C
        )

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
        loss = self._compute_loss(outputs.logits, labels) if labels is not None else None

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _process_visual_prompts(
        self,
        vit_embeds: torch.Tensor,
        vp_overall_mask: Optional[torch.Tensor],
        prompt_masks: Optional[List[torch.Tensor]],
        image_flags: torch.Tensor,
        embed_dim: int
    ) -> torch.Tensor:
        """
        Process visual prompts for object-level understanding.
        
        Args:
            vit_embeds: Visual embeddings from the vision encoder
            vp_overall_mask: Overall mask for visual prompts
            prompt_masks: List of prompt masks for each object
            image_flags: Flags indicating which positions have images
            embed_dim: Embedding dimension
            
        Returns:
            Processed visual embeddings
        """
        if vp_overall_mask is None and prompt_masks is None:
            return vit_embeds.reshape(-1, embed_dim)
        else:
            assert vp_overall_mask is not None and prompt_masks is not None

        vp_embeds = []
        vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
        prompt_masks = [mask.to(vit_embeds.device).bool() for mask in prompt_masks]
        
        vp_overall_mask = vp_overall_mask[image_flags == 1]
        overall_tile_vit_embeds = vit_embeds[vp_overall_mask]

        vp_img_idx = 0
        for img_idx in range(len(vit_embeds)):
            # Add base visual embedding
            vp_embeds.append(vit_embeds[img_idx].reshape(-1, embed_dim))
            
            # Add object-specific embeddings if this image has visual prompts
            if vp_overall_mask[img_idx]:
                tile_embeds = overall_tile_vit_embeds[vp_img_idx].reshape(-1, embed_dim)
                object_masks = prompt_masks[vp_img_idx]
                n_objects = len(object_masks)
                
                # Repeat tile embeddings for each object
                tile_embeds = tile_embeds.unsqueeze(0).repeat(n_objects, 1, 1)
                object_masks = object_masks.reshape(n_objects, -1)
                
                # Apply masks to get object-specific embeddings
                vp_embeds.append(tile_embeds[object_masks])
                vp_img_idx += 1

        return torch.cat(vp_embeds, dim=0)

    def _embed_visual_features(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        batch_size: int,
        seq_len: int,
        embed_dim: int
    ) -> torch.Tensor:
        """
        Embed visual features into the input embeddings.
        
        Args:
            input_embeds: Input text embeddings
            input_ids: Input token IDs
            visual_embeds: Visual embeddings to inject
            batch_size: Batch size
            seq_len: Sequence length
            embed_dim: Embedding dimension
        """
        input_ids_flat = input_ids.reshape(batch_size * seq_len)
        selected = (input_ids_flat == self.model.img_context_token_id)
        
        try:
            input_embeds[selected] = visual_embeds
        except RuntimeError as e:
            # decode the input_ids
            input_ids_decoded = decode_tokens_with_counts(input_ids_flat.cpu().tolist(), self.tokenizer)
            print('--------------------------')
            print("".join(input_ids_decoded))
            print('--------------------------')
            raise ValueError(f"Incompatible shapes. {selected.sum()} vs {visual_embeds.shape[0]}")
        return input_embeds

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling."""
        # Shift labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        
        return loss_fct(shift_logits, shift_labels)

