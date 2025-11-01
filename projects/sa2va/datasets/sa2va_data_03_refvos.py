import os
from typing import Literal
import pickle

import torch
import numpy as np
import copy
import json
import random
import pycocotools.mask as maskUtils

from .common import SEG_QUESTIONS, ANSWER_LIST
from .base import Sa2VABaseDataset
from .data_utils import sam2_path_patch, get_video_frames, decode_masklet, opencvimg_to_pil

class Sa2VA03RefVOS(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 select_number=5,
                 sampled_frames=5,
                 dataset_type: Literal['default', 'refytvos', 'refsav']='default',
                 **kwargs):
        
        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            **kwargs
        )
        
        # RefVOS-specific configurations
        self.dataset_type = dataset_type
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert expression_file and tokenizer

        if self.dataset_type in ['default', 'refytvos']:
            assert mask_file is not None
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            # 每个视频对应的所有meta在metas列表中的索引
            self.video_infos = vid2metaid
            # 所有视频的ids列表（每个id都是一长串字符串）
            self.videos = list(self.video_infos.keys())
            # 所有mask的字典，直接由json/pkl文件加载得到
            self.mask_dict = mask_dict
            # 所有expression的列表
            self.text_data = metas
        elif self.dataset_type == 'refsav':
            # prepare expression annotation files
            assert mask_file is None
            with open(expression_file, 'r') as f:
                expression_datas = json.load(f)
            self.video_infos = expression_datas
            self.videos = list(self.video_infos.keys())
            self.mask_dict = None # refsav masks are saved seperately
            self.text_data = None # text data are in the anno_dict

        self.image_folder = image_folder

    def real_len(self):
        # 返回key的数量，也就是视频的数量
        return len(self.video_infos)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                if self.dataset_type == 'default':
                    meta['mask_anno_id'] = exp_dict['anno_id']
                elif self.dataset_type == 'refytvos':
                    meta['mask_anno_id'] = [str(anno_count), ]
                else:
                    raise NotImplementedError

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                # 注意，metas是一个列表，储存的是每个视频中的所有expression，每个expression用一个exp_id表示
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                # 注意，vid2metaid是一个字典，储存的是每个视频对应的所有meta在metas列表中的索引
                vid2metaid[vid_name].append(len(metas) - 1)
        if mask_file.endswith('.pkl'):
            with open(mask_file, 'rb') as f:
                mask_dict = pickle.load(f)
        elif mask_file.endswith('.json'):
            with open(mask_file, 'r') as f:
                mask_dict = json.load(f)
        else:
            raise NotImplementedError

        return vid2metaid, metas, mask_dict

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        # 每一个expression对应一个object_masks
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                # 每个annotation对应的mask信息个数相同，都是所选帧的数量
                # 遍历长度挑第一项就行，因为帧的个数都一样
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    # 这里的i_anno是角标
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        # json里存的是rle编码的mask，需要decode成二值mask
                        # json里mask的大致压缩原理：连续记录0101的数量，然后编码成字符串存储
                        # 这里还原成nparray的二值矩阵
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            # axis是从0开始的，这是第三维的求和，一般是通道的维度
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        # 把所有annotation的mask或一起，合并成一个mask
                        _mask = _mask | m
                    _object_masks.append(_mask)
                # 把所有帧的mask列表合成一个大的nparray
                # np.stack就是可以指定维数的[] + list.append函数
                _object_masks = np.stack(_object_masks, axis=0)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        # 检测所有object的mask shape是否一致
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        # 又加了一个维度
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)
        ret_masks = torch.from_numpy(ret_masks)
        # 展平操作，合并前两个维度，变成(n_obj*n_frames, h, w)
        # 使得ret_mask成为一系列的掩码集合，顺序就是先所有第一个object的所有帧，然后所有第二个object的所有帧，以此类推
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def prepare_data(self, index):
        """Prepare data for a given index using unified base class methods."""
        # 找到index对应的视频，因为重复的原因，找到具体位置
        index = index % self.real_len()
        # 当前视频的所有expression的索引列表
        selected_video_objects = self.video_infos[self.videos[index]]
        # 所有的expression信息都在self.text_data中
        if self.text_data is not None:
            # 根据里面的索引找真正的expression信息
            video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

            # 在一个视频的所有expression中，选择select_number个expression(默认5个)
            if len(video_objects_infos) > self.select_number:
                selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
                video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
            else:
                selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
                video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
            # 这里是为了保证每个视频都有相同数量的expression
            # 传入的是一个expression信息的列表
            data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames)
        else:
            assert self.dataset_type == 'refsav'
            # for refsav dataset, all the info is in the video_info rather than
            # saved to objects. We need to extract it from there.
            object_ids = list(selected_video_objects['objects'].keys())

            video_path = os.path.join(self.image_folder, selected_video_objects['video_path'])
            anno_path = os.path.join(self.image_folder, selected_video_objects['anno_path'])
            video_path, anno_path = sam2_path_patch(video_path, anno_path)

            video_frames = get_video_frames(video_path)
            video_frames = video_frames[::4]
            if not video_frames:
                return None
            # mask annotation
            with open(anno_path, 'r') as f:
                mask_data = json.load(f)
            masklets = decode_masklet(mask_data['masklet'])
            assert len(masklets) == len(video_frames), "frames should be equal to frames"
            n_objects = len(object_ids)
                # sample object
            if n_objects > self.select_number:
                selected_indexes = np.random.choice(n_objects, self.select_number)
            else:
                selected_indexes = np.random.choice(n_objects, self.select_number, replace=True)

            selected_object_ids = [object_ids[_idx] for _idx in selected_indexes]
            video_objects_infos = [selected_video_objects['objects'][_idx] for _idx in selected_object_ids]
            mask_temp_list = []
            for _mask in masklets:
                _mask_selected = []
                for _idx in selected_object_ids:
                    _mask_selected.append(_mask[:, :, int(_idx)])
                _mask_selected = np.stack(_mask_selected, axis=0)
                mask_temp_list.append(_mask_selected)
            masklets = mask_temp_list

            data_dict = self.dataset_map_fn_refsav(video_frames, masklets, video_objects_infos, select_k=self.sampled_frames)
        
        if data_dict is None:
            return None

        out_data_dict = {}
        
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('images', None) is not None:
            try:
                # Load images from paths
                if self.dataset_type in ['default', 'refytvos']:
                    # PIL的image对象列表
                    images = []
                    for img_path in data_dict['images']:
                        full_img_path = os.path.join(self.image_folder, img_path)
                        image = self._read_image(full_img_path)
                        if image is None:
                            raise Exception(f"Failed to read image: {full_img_path}")
                        images.append(image)
                else:
                    assert self.dataset_type == 'refsav'
                    images = [opencvimg_to_pil(img) for img in data_dict['images']]
                    
                # Process multiple images using base class method
                image_data = self._process_multiple_images(images)
                out_data_dict.update(image_data)
                
                # Create video token string
                num_frames = len(images)
                image_token_str = self._create_token_string(image_data['num_image_tokens'], num_frames)
                
                # Process conversations using unified method
                conversations = self._process_conversations_for_encoding(
                    data_dict['conversations'], image_token_str, is_video=True
                )
                
                # Handle token expansion for qwen if needed
                if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data:
                    conversations = self._expand_video_tokens(
                        conversations, image_data['num_frame_tokens'], image_data['num_image_tokens']
                    )
                
                # Get input/labels using base class method
                token_dict = self.get_inputid_labels(conversations)
                out_data_dict.update(token_dict)
                
            except Exception as e:
                print(f'Error processing images: {e}', flush=True)
                return None
        else:
            # No images case
            conversations = self._process_conversations_for_encoding(data_dict['conversations'], None, is_video=True)
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            out_data_dict['masks'] = None

        out_data_dict['type'] = 'video'
        return out_data_dict
    

    def dataset_map_fn_refsav(self, video_frames, masklets, video_objects_infos, select_k=5):
        len_frames = len(masklets)
        # prepare images, random select k frames
        if len_frames > select_k + 1:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()
        video_frames = [video_frames[_idx] for _idx in selected_frame_indexes]
        masklets = [masklets[_idx] for _idx in selected_frame_indexes]


        ret_masks = np.stack(masklets, axis=1)  # (n_obj, n_frames, h, w), align with other datasets
        ret_masks = torch.from_numpy(ret_masks)
        ret_masks = ret_masks.flatten(0, 1)

        expressions = [object_info['formated'] for object_info in video_objects_infos]

         # Convert to unified conversation format
        conversations = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            question_template = random.choice(SEG_QUESTIONS)
            question = question_template.format(class_name=exp)
            if i == 0:
                # Add <image> placeholder for first question
                conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        ret = {'images': video_frames, 'conversations': conversations, 'masks': ret_masks}
        return ret

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        # 保证都是在一个视频里的frame
        len_frames = len(data_dict[0]['frames'])
        for object_info in data_dict:
            assert len_frames == len(object_info['frames'])

        # prepare images, random select k frames
        # 从所有帧中找select_k的帧，注意不是expression
        if len_frames > select_k + 1:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()
        
        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            # images存的是选中的frame的文件路径
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))

        # prepare text
        # 从meta中取出所有的expression
        expressions = [object_info['exp'] for object_info in data_dict]

        # Convert to unified conversation format
        # 每一项都是一个字典，话来自谁 + 话的内容是什么
        conversations = []
        for i, exp in enumerate(expressions):
            # 生成问题
            # 若本身expression就是一个问题，那么就不动
            # the exp is a question
            if '?' in exp:
                question = exp
            # 若expression本身不是问题，而是该图像的描述，那么就将该表述插到问题模板中
            # 例：
            # 模板：图片中的 xxx 是什么？请给出分割掩码。
            # 表述：一只红色的狗
            # 结果：图片中的 一只红色的狗 是什么？请给出分割掩码。
            else:
                exp = exp.replace('.', '').strip()
                # SEG_QUESTIONS里是所有问题模板的列表，这里随机挑了一个
                question_template = random.choice(SEG_QUESTIONS)
                # 生成问题
                question = question_template.format(class_name=exp.lower())

            if i == 0:
                # Add <image> placeholder for first question
                conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                conversations.append({'from': 'human', 'value': question})
            # 回答模板中没有占位符，就是一句话（例如：“是的，分割结果是[SEG]”） ——纯纯模板
            conversations.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

        # prepare masks
        # one exp can have multiple annos
        # 存储所有expression的mask信息，也就是每个expression包含的annotation对应的mask信息
        video_masks = []
        for object_info in data_dict:
            # 一个列表，里面是该expression对应的所有annotation ids
            anno_ids = object_info['mask_anno_id']
            # 一个列表，存储的是每一项都是一个annotation对应的所有选择帧的mask信息
            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                # 在mask_dict中（通过读取另一个json/pkl获取的）根据id找到对应的真实的mask
                # 这里是列表，存储了所有帧的mask信息
                frames_masks = self.mask_dict[anno_id]
                frames_masks_ = []
                for frame_idx in selected_frame_indexes:
                    # 找到所有选择帧的mask信息
                    frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        # read image size from the first image
        first_image_path = images[0]
        first_image_path = os.path.join(self.image_folder, first_image_path)
        first_image = self._read_image(first_image_path)
        if first_image is None:
            return None
        
        # switch height and width (PIL system (WH vs HW system)
        _image_size = first_image.size
        image_size = (_image_size[1], _image_size[0])
        # 将mask从rle编码解码成二值掩码矩阵张量，并且形状为(n_obj*n_frames, h, w) ——前两维展平了
        masks = self.decode_mask(video_masks, image_size=image_size)
        if masks is None:
            return None

        # 返回一个字典，包含图片路径列表、对话列表、掩码张量
        ret = {'images': images, 'conversations': conversations, 'masks': masks}
        return ret

    @property
    def modality_length(self):
        if self.name == 'Ref-SAV':
            return [self._get_modality_length_default(20000) for _ in range(self.real_len())]
        else:
            return [self._get_modality_length_default(10000) for _ in range(self.real_len())]

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if images don't exist
        """
        if self.dataset_type in ['refsav']:
            mock_data_dict = {}
            video_info = self.video_infos[self.videos[index]]
            video_path = os.path.join(self.image_folder, video_info['video_path'])
            anno_path = os.path.join(self.image_folder, video_info['anno_path'])
            video_path, anno_path = sam2_path_patch(video_path, anno_path)

            if not os.path.exists(video_path):
                print(f'Video path does not exist: {video_path}', flush=True)
                return None
            if not os.path.exists(anno_path):
                print(f'Annotation path does not exist: {anno_path}', flush=True)
                return None
            
            mock_data_dict.update({
                'video_name': video_info['video_path'],
                'has_images': True,
                'num_frames': 5,
                'num_objects': len(video_info['objects']),
                'status': 'valid',
                'type': 'video'
            })
            return mock_data_dict

        index = index % self.real_len()
        selected_video_objects = self.video_infos[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        mock_data_dict = {}
        
        # Check image existence
        len_frames = len(video_objects_infos[0]['frames'])
        if len_frames > self.sampled_frames + 1:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=True)
        selected_frame_indexes.sort()
        
        # Check if images exist
        for selected_frame_index in selected_frame_indexes:
            frame_id = video_objects_infos[0]['frames'][selected_frame_index]
            image_path = os.path.join(video_objects_infos[0]['video'], frame_id + '.jpg')
            full_image_path = os.path.join(self.image_folder, image_path)
            if not self._check_image_exists(full_image_path):
                print(f'Image does not exist: {full_image_path}', flush=True)
                return None
        
        mock_data_dict.update({
            'video_name': video_objects_infos[0]['video'],
            'has_images': True,
            'num_frames': len(selected_frame_indexes),
            'num_objects': len(video_objects_infos),
            'num_conversations': len(video_objects_infos) * 2,  # Each object creates 2 conversation turns
            'status': 'valid',
            'type': 'video'
        })
            
        return mock_data_dict
