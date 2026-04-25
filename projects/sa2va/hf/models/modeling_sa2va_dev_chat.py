from .modeling_sa2va_chat import Sa2VAChatModel
from .configuration_sa2va_dev_chat import Sa2VADevChatConfig
from transformers import GenerationConfig

# LIS
import os
import torch
import torch.nn as nn
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
        self.importance_scorer = TransformerScorer(
            config.hidden_size,
            hidden_dim=config.scorer_hidden_dim,
            init_scale=config.scorer_init_scale,
        )
        self.budgets = config.budgets
        self._last_efficiency_metrics = None

    def _hard_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        total_tokens = int(scores.shape[0])
        keep_tokens = max(1, int(total_tokens * self.budgets))
        keep_tokens = min(keep_tokens, total_tokens)
        selected = torch.topk(scores, k=keep_tokens, dim=0).indices
        selected = selected.sort().values
        return selected

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.LongTensor] = None,
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

        eval_time_enabled = os.environ.get('EVAL_TIME', 'false').lower() == 'true'
        collect_cuda_metrics = eval_time_enabled and torch.cuda.is_available()
        sample_metrics = {
            'input_visual_token_num': 0,
            'selected_visual_token_num': 0,
            'generation_prefill_time_ms': None,
            'generation_latency_time_ms': None,
            'after_generation_memory_bytes': None,
        }

        # This is only used for timing the first prefill forward during generation.
        prefill_time_ms = None

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
            if B != 1:
                raise NotImplementedError(
                    "Sa2VADevChatModel.generate currently supports batch_size=1 when visual token pruning is enabled."
                )

            input_embeds = input_embeds.reshape(B * N, C)
            flat_input_ids = input_ids.to(device).reshape(B * N)

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

            flat_image_positions = torch.nonzero(
                flat_input_ids == self.img_context_token_id, as_tuple=False
            ).squeeze(-1)
            if flat_image_positions.numel() == 0:
                raise ValueError("No <IMG_CONTEXT> token found in input_ids, cannot align visual embeddings.")

            visual_tokens = vit_embeds.reshape(-1, C) if vp_embeds is None else vp_embeds.reshape(-1, C)
            visual_tokens = visual_tokens.to(input_embeds.device)

            required = int(flat_image_positions.numel())
            available = int(visual_tokens.shape[0])
            if available == 0:
                raise ValueError("Visual tokens are empty after feature extraction; cannot perform top-k pruning.")
            if available < required:
                repeat_times = required // max(available, 1) + 1
                visual_tokens = visual_tokens.repeat(repeat_times, 1)
            visual_tokens = visual_tokens[:required]

            input_embeds[flat_image_positions] = visual_tokens

            learned_scores = self.importance_scorer(visual_tokens.unsqueeze(0).detach()).squeeze(0)
            selected_visual_indices = self._hard_topk_indices(learned_scores)
            retained_image_positions = flat_image_positions[selected_visual_indices]
            sample_metrics['input_visual_token_num'] = required
            sample_metrics['selected_visual_token_num'] = int(selected_visual_indices.numel())

            flat_text_positions = torch.nonzero(
                flat_input_ids != self.img_context_token_id, as_tuple=False
            ).squeeze(-1)
            selected_positions = torch.cat(
                (retained_image_positions, flat_text_positions), dim=0
            ).sort().values

            input_embeds = input_embeds[selected_positions].unsqueeze(0)

            if attention_mask is not None:
                flat_attention_mask = attention_mask.to(device).reshape(B * N)
                attention_mask = flat_attention_mask[selected_positions].unsqueeze(0)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        model_generate_kwargs = dict(generate_kwargs)
        if return_dict is not None and "return_dict_in_generate" not in model_generate_kwargs:
            model_generate_kwargs["return_dict_in_generate"] = return_dict

        original_forward = None
        e2e_start = None
        e2e_end = None

        if collect_cuda_metrics:
            e2e_start = torch.cuda.Event(enable_timing=True)
            e2e_end = torch.cuda.Event(enable_timing=True)
            e2e_start.record()
            original_forward = self.language_model.forward

            def wrapped_forward(*args, **kwargs):
                nonlocal prefill_time_ms
                seq_len = None
                if kwargs.get('inputs_embeds', None) is not None:
                    seq_len = kwargs['inputs_embeds'].shape[1]
                elif kwargs.get('input_ids', None) is not None:
                    seq_len = kwargs['input_ids'].shape[1]

                if prefill_time_ms is None and seq_len is not None and seq_len > 1:
                    prefill_start = torch.cuda.Event(enable_timing=True)
                    prefill_end = torch.cuda.Event(enable_timing=True)
                    prefill_start.record()
                    output = original_forward(*args, **kwargs)
                    prefill_end.record()
                    torch.cuda.synchronize()
                    prefill_time_ms = float(prefill_start.elapsed_time(prefill_end))
                    return output

                return original_forward(*args, **kwargs)

            self.language_model.forward = wrapped_forward

        try:
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **model_generate_kwargs,
            )
        finally:
            if original_forward is not None:
                self.language_model.forward = original_forward

        if collect_cuda_metrics:
            e2e_end.record()
            torch.cuda.synchronize()
            sample_metrics['generation_prefill_time_ms'] = prefill_time_ms
            sample_metrics['generation_latency_time_ms'] = float(e2e_start.elapsed_time(e2e_end))
            sample_metrics['after_generation_memory_bytes'] = int(torch.cuda.max_memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)

            print(f"Input visual token number is: {sample_metrics['input_visual_token_num']}")
            if sample_metrics['generation_prefill_time_ms'] is not None:
                print(f"Generation prefill time is: {sample_metrics['generation_prefill_time_ms']}")
            print(f"Generation latency time is: {sample_metrics['generation_latency_time_ms']}")
            print(f"after generation memory: {sample_metrics['after_generation_memory_bytes']}")

            self._last_efficiency_metrics = sample_metrics
        else:
            self._last_efficiency_metrics = None

        return outputs
