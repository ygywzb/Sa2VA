from .modeling_sa2va_chat import Sa2VAChatModel
from transformers import GenerationConfig

import os
import types
from typing import Optional

import torch


class Sa2VAChatEfficiencyModel(Sa2VAChatModel):
    """
    Baseline Sa2VA chat model with VisionSelector-style efficiency metrics.

    Controlled by environment variables:
    - EVAL_TIME: enable metric collection when set to true/1/yes/on
    - EVAL_TIME_PRINT_PER_SAMPLE: print per-sample metrics (default true)
    - EVAL_TIME_RESET_PEAK_BEFORE_GENERATE: reset peak memory before generate (default true)

    Collected metrics:
    - after generation memory (bytes, torch.cuda.max_memory_allocated)
    - Generation prefill time (ms)
    - Generation latency time (ms)
    - Input visual token number
    """

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

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

        eval_time = Sa2VAChatEfficiencyModel._env_flag("EVAL_TIME", default=False)
        print_per_sample = Sa2VAChatEfficiencyModel._env_flag(
            "EVAL_TIME_PRINT_PER_SAMPLE", default=True
        )
        reset_peak_before = Sa2VAChatEfficiencyModel._env_flag(
            "EVAL_TIME_RESET_PEAK_BEFORE_GENERATE", default=True
        )

        visual_token_num = 0

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0
                    )

                vit_embeds = self.extract_feature(pixel_values.to(device))

            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            bsz, seq_len, channels = input_embeds.shape
            input_embeds = input_embeds.reshape(bsz * seq_len, channels)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, channels))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(
                            -1, channels
                        )
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1

                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            input_ids = input_ids.reshape(bsz * seq_len)
            selected = input_ids == self.img_context_token_id
            required = int(selected.sum().item())
            visual_token_num = required
            assert required != 0

            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, channels).to(input_embeds.device)
            else:
                vp_reshaped = vp_embeds.reshape(-1, channels)
                if len(input_embeds[selected]) != len(vp_reshaped):
                    print(
                        "Shape mismatch, selected is {}, vp embeds is {} !!!".format(
                            len(input_embeds[selected]), len(vp_reshaped)
                        )
                    )
                    min_tokens = min(len(input_embeds[selected]), len(vp_reshaped))
                    input_embeds[selected][:min_tokens] = vp_reshaped[:min_tokens].to(
                        input_embeds.device
                    )
                else:
                    input_embeds[selected] = vp_reshaped.to(input_embeds.device)

            input_embeds = input_embeds.reshape(bsz, seq_len, channels)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        prefill_ms = None
        latency_ms = None
        peak_memory_bytes = None

        if eval_time and torch.cuda.is_available():
            if reset_peak_before:
                torch.cuda.reset_peak_memory_stats(device)

            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            latency_start = torch.cuda.Event(enable_timing=True)
            latency_end = torch.cuda.Event(enable_timing=True)
            prefill_state = {"recorded": False}

            original_forward = self.language_model.forward
            prefill_start.record()

            def _wrapped_forward(module_self, *f_args, **f_kwargs):
                outputs = original_forward(*f_args, **f_kwargs)
                if not prefill_state["recorded"]:
                    seq = None
                    ids = f_kwargs.get("input_ids")
                    embeds = f_kwargs.get("inputs_embeds")
                    if embeds is not None and embeds.ndim >= 2:
                        seq = int(embeds.shape[1])
                    elif ids is not None and ids.ndim >= 2:
                        seq = int(ids.shape[1])
                    if seq is None and hasattr(outputs, "logits") and outputs.logits is not None:
                        if outputs.logits.ndim >= 2:
                            seq = int(outputs.logits.shape[1])
                    if seq is not None and seq != 1:
                        prefill_end.record()
                        prefill_state["recorded"] = True
                return outputs

            self.language_model.forward = types.MethodType(
                _wrapped_forward, self.language_model
            )

            latency_start.record()
            try:
                outputs = self.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                    generation_config=generation_config,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                    **generate_kwargs,
                )
            finally:
                self.language_model.forward = original_forward

            latency_end.record()
            torch.cuda.synchronize(device)

            latency_ms = float(latency_start.elapsed_time(latency_end))
            if prefill_state["recorded"]:
                prefill_ms = float(prefill_start.elapsed_time(prefill_end))

            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )

        self._last_efficiency_metrics = {
            "input_visual_token_number": int(visual_token_num),
            "generation_prefill_time_ms": prefill_ms,
            "generation_latency_time_ms": latency_ms,
            "after_generation_memory": peak_memory_bytes,
        }

        if eval_time and print_per_sample:
            print(f"Input visual token number is: {visual_token_num}")
            if prefill_ms is not None:
                print(f"Generation prefill time is: {prefill_ms}")
            if latency_ms is not None:
                print(f"Generation latency time is: {latency_ms}")
            if peak_memory_bytes is not None:
                print(f"after generation memory: {peak_memory_bytes}")

        return outputs
