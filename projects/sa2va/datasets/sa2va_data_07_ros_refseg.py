import random
from typing import Literal

import numpy as np
import torch
from pycocotools import mask as mask_utils

from .base import Sa2VABaseDataset
from .common import ANSWER_LIST, SEG_QUESTIONS
from third_parts.mmdet.datasets.refcoco import RefCocoDataset


class Sa2VAROSRefSegBase(RefCocoDataset, Sa2VABaseDataset):
    """Base class for ROS-Sa2VA referring segmentation datasets.

    The ROS datasets provide COCO-like instances plus refs files, where
    segmentation is stored as RLE dicts. This adapter keeps compatibility with
    polygon masks when present.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split_file,
                 data_prefix,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats: int = 1,
                 name: str = 'Sa2VAROSRefSegBase',
                 **kwargs):

        RefCocoDataset.__init__(
            self,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )

        Sa2VABaseDataset.__init__(
            self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
        )

        self.begin_str = '<image>\n'
        self.image_folder = data_root
        self.num_classes_per_sample = num_classes_per_sample
        self.single_image_mode = single_image_mode

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def _decode_mask(self, seg, height, width):
        """Decode polygon or RLE mask into a uint8 binary map."""
        if isinstance(seg, dict):
            decoded = mask_utils.decode(seg)
        elif isinstance(seg, str):
            # Some annotations store compressed RLE counts as plain strings.
            decoded = mask_utils.decode({'size': [height, width], 'counts': seg})
        elif isinstance(seg, list):
            rles = mask_utils.frPyObjects([seg], height, width)
            decoded = mask_utils.decode(rles)
        else:
            raise TypeError(f'Unsupported segmentation type: {type(seg)}')

        decoded = decoded.astype(np.uint8)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2).astype(np.uint8)
        return decoded

    def _align_mask_to_image_size(self, mask, height, width):
        """Align decoded mask to image size via crop/pad when metadata drifts."""
        h, w = mask.shape
        if (h, w) == (height, width):
            return mask

        aligned = np.zeros((height, width), dtype=np.uint8)
        copy_h = min(h, height)
        copy_w = min(w, width)
        aligned[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
        return aligned

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        instances = ann_info['instances']
        text = ann_info['text']
        if len(instances) == 0 or len(text) == 0:
            return None

        replace = len(instances) < self.num_classes_per_sample
        indices = np.random.choice(
            range(len(instances)),
            self.num_classes_per_sample,
            replace=replace,
        )

        masks, phrases = [], []
        for idx in indices:
            inst = instances[idx]
            phrase = str(text[idx]).strip().lower()
            if not phrase:
                continue
            if phrase.endswith('.'):
                phrase = phrase[:-1]

            binary_mask = np.zeros((height, width), dtype=np.uint8)

            # RefCOCO-like masks are usually list-based, but ROS data may also
            # contain single-RLE dicts for some samples.
            mask_data = inst['mask']
            if isinstance(mask_data, dict):
                seg_iter = [mask_data]
            elif isinstance(mask_data, list):
                seg_iter = mask_data
            elif isinstance(mask_data, str):
                seg_iter = [mask_data]
            else:
                raise TypeError(f'Unsupported mask container type: {type(mask_data)}')

            for seg in seg_iter:
                decoded = self._decode_mask(seg, height, width)
                decoded = self._align_mask_to_image_size(decoded, height, width)
                binary_mask = np.clip(binary_mask + decoded, 0, 1)

            if binary_mask.sum() == 0:
                continue

            masks.append(binary_mask)
            phrases.append(phrase)

        if len(masks) == 0:
            return None

        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

        mask_tensors = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)

        ann_info.update({
            'masks': mask_tensors,
            'conversations': conversation,
            'image': image_path,
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self._read_image(image_file)
            if image is None:
                return None

            image_data = self._process_single_image(image, self.single_image_mode)
            out_data_dict.update(image_data)

            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
        else:
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], None)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)

        return out_data_dict

    def real_len(self):
        if self.serialize_data:
            return len(self.data_address)
        return len(self.data_list)

    def __len__(self):
        return int(self.real_len() * self.repeats)

    def __getitem__(self, index):
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]

        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data

        raise RuntimeError(f'Failed to get valid data after {self._max_refetch + 1} attempts')


class Sa2VARISLADRefSeg(Sa2VAROSRefSegBase):
    """RIS-LAD referring segmentation adapter."""

    def __init__(self,
                 data_root,
                 ann_file='ris_lad/instances.json',
                 split_file='ris_lad/refs(unc).p',
                 data_prefix=dict(img_path='images/ris_lad/'),
                 name: str = 'Sa2VARISLADRefSeg',
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split_file=split_file,
            data_prefix=data_prefix,
            name=name,
            **kwargs,
        )


class Sa2VARRSISDRefSeg(Sa2VAROSRefSegBase):
    """RRSIS-D referring segmentation adapter."""

    def __init__(self,
                 data_root,
                 ann_file='rrsisd/instances.json',
                 split_file='rrsisd/refs(unc).p',
                 data_prefix=dict(img_path='images/rrsisd/JPEGImages/'),
                 name: str = 'Sa2VARRSISDRefSeg',
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split_file=split_file,
            data_prefix=data_prefix,
            name=name,
            **kwargs,
        )