from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from .modeling_sa2va_chat import get_seg_hidden_states
from .modeling_sa2va_dev_chat import Sa2VADevChatModel


class Sa2VADevChatVisualizeModel(Sa2VADevChatModel):
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
            self._store_importance_grid(
                learned_scores=learned_scores,
                num_images=vit_embeds.shape[0],
                tokens_per_image=vit_embeds.shape[1],
                vp_overall_mask=vp_overall_mask,
                prompt_masks=prompt_masks,
                required=required,
            )
            selected_visual_indices = self._hard_topk_indices(learned_scores)
            retained_image_positions = flat_image_positions[selected_visual_indices]

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

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **model_generate_kwargs,
        )

        return outputs

    def _store_importance_grid(
        self,
        learned_scores: torch.Tensor,
        num_images: int,
        tokens_per_image: int,
        vp_overall_mask,
        prompt_masks,
        required: Optional[int],
    ) -> None:
        grid_size = int(tokens_per_image ** 0.5)
        if grid_size * grid_size != tokens_per_image:
            return

        mappings = self._build_visual_token_mappings(
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )
        if not mappings:
            return

        if required is not None:
            if len(mappings) < required:
                repeat_times = required // max(len(mappings), 1) + 1
                mappings = (mappings * repeat_times)[:required]
            else:
                mappings = mappings[:required]

        scores = learned_scores.detach().float().cpu()
        grid_scores = torch.full((num_images, tokens_per_image), float("-inf"))
        for idx, (img_idx, tok_idx) in enumerate(mappings):
            if idx >= scores.numel():
                break
            score = float(scores[idx])
            if score > grid_scores[img_idx, tok_idx]:
                grid_scores[img_idx, tok_idx] = score

        grid_scores[grid_scores == float("-inf")] = 0.0
        self._last_importance_grid = {
            "grid_scores": grid_scores.view(num_images, grid_size, grid_size),
            "grid_size": grid_size,
            "num_images": num_images,
            "tokens_per_image": tokens_per_image,
        }

    def _build_visual_token_mappings(
        self,
        num_images: int,
        tokens_per_image: int,
        vp_overall_mask,
        prompt_masks,
    ):
        mappings = []
        if vp_overall_mask is not None and prompt_masks is not None:
            vp_mask = vp_overall_mask.detach().cpu().bool()
            prompt_masks_cpu = [mask.detach().cpu() for mask in prompt_masks]
            i_vp_img = 0
            for i_img in range(num_images):
                mappings.extend((i_img, idx) for idx in range(tokens_per_image))
                if bool(vp_mask[i_img]):
                    obj_mask = prompt_masks_cpu[i_vp_img].reshape(
                        prompt_masks_cpu[i_vp_img].shape[0], -1
                    )
                    flat_mask = obj_mask.reshape(-1)
                    selected = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
                    for flat_idx in selected.tolist():
                        mappings.append((i_img, int(flat_idx % tokens_per_image)))
                    i_vp_img += 1
        else:
            for i_img in range(num_images):
                mappings.extend((i_img, idx) for idx in range(tokens_per_image))
        return mappings

    def _build_importance_maps(self, num_frames, ori_image_size, images=None):
        info = getattr(self, "_last_importance_grid", None)
        if info is None:
            return None

        grid_scores = info["grid_scores"]
        orig_w, orig_h = ori_image_size
        image_size = int(self.image_size)

        if images is None or num_frames > 1:
            frame_count = min(num_frames, grid_scores.shape[0])
            maps = []
            for i in range(frame_count):
                grid = grid_scores[i].unsqueeze(0).unsqueeze(0)
                patch_map = F.interpolate(
                    grid, size=(image_size, image_size), mode="nearest"
                )
                pixel_map = F.interpolate(
                    patch_map,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                maps.append(pixel_map.cpu().numpy())
            return {"importance_maps": maps}

        layout = get_dynamic_preprocess_layout(
            orig_w, orig_h, self.min_dynamic_patch, self.max_dynamic_patch, image_size
        )
        grid_w, grid_h, target_width, target_height, blocks = layout

        tile_count = min(blocks, grid_scores.shape[0])
        canvas = torch.zeros((target_height, target_width), dtype=grid_scores.dtype)
        for idx in range(tile_count):
            grid = grid_scores[idx].unsqueeze(0).unsqueeze(0)
            tile_map = F.interpolate(
                grid, size=(image_size, image_size), mode="nearest"
            ).squeeze(0).squeeze(0)
            row = idx // grid_w
            col = idx % grid_w
            y0 = row * image_size
            x0 = col * image_size
            canvas[y0:y0 + image_size, x0:x0 + image_size] = tile_map

        tile_pixel_map = F.interpolate(
            canvas.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        thumbnail_map = None
        if grid_scores.shape[0] > blocks:
            grid = grid_scores[blocks].unsqueeze(0).unsqueeze(0)
            thumb = F.interpolate(
                grid, size=(image_size, image_size), mode="nearest"
            )
            thumbnail_map = F.interpolate(
                thumb,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        combined = (
            torch.maximum(tile_pixel_map, thumbnail_map)
            if thumbnail_map is not None
            else tile_pixel_map
        )

        payload = {"importance_maps": [combined.cpu().numpy()]}
        payload["importance_tile_map"] = tile_pixel_map.cpu().numpy()
        if thumbnail_map is not None:
            payload["importance_thumbnail_map"] = thumbnail_map.cpu().numpy()
        return payload

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        images = None

        if image is None and video is None and '<image>' not in past_text:
            # TEXT
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            # attention_mask是二维的
            # @TODO: 可能需要改generate里attnmask和positionid的处理，更甚者可能要改llm的forward，因为visionselector论文改了llm的forward来适配LIS了
            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                # attention mask和positionid照样传入，会给llm的forward
                # 以generate_args的形式传入generate函数，generate函数会传给model的forward
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                # VIDEO
                pixel_values = []
                extra_pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    # assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        img = self.transformer(frame_image)
                        pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                num_image_tokens = self.patch_token
                num_frames = len(pixel_values)

                input_dict['vp_overall_mask'] = None
            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None

                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1
            input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': mask_prompts,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        output = {'prediction': predict, 'prediction_masks': ret_masks}
        if image is None and video is None and '<image>' not in past_text:
            return output

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        importance_payload = self._build_importance_maps(
            num_frames=num_frames,
            ori_image_size=ori_image_size,
            images=images if video is None else None,
        )
        if importance_payload is not None:
            output.update(importance_payload)

        return output


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_dynamic_preprocess_layout(orig_width, orig_height, min_num, max_num, image_size):
    aspect_ratio = orig_width / orig_height

    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    grid_w = target_aspect_ratio[0]
    grid_h = target_aspect_ratio[1]
    target_width = image_size * grid_w
    target_height = image_size * grid_h
    blocks = grid_w * grid_h
    return grid_w, grid_h, target_width, target_height, blocks
