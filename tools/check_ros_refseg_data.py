import json
from pathlib import Path

from PIL import Image
from pycocotools import mask as mask_utils


def decode_mask(seg, h, w):
    if isinstance(seg, dict):
        m = mask_utils.decode(seg)
    elif isinstance(seg, list):
        rles = mask_utils.frPyObjects([seg], h, w)
        m = mask_utils.decode(rles)
    elif isinstance(seg, str):
        m = mask_utils.decode({'size': [h, w], 'counts': seg})
    else:
        raise TypeError(f'Unsupported segmentation type: {type(seg)}')

    if m.ndim == 3:
        m = m.any(axis=2).astype('uint8')
    return m


def align_mask(m, h, w):
    mh, mw = m.shape
    if (mh, mw) == (h, w):
        return m, False

    aligned = __import__('numpy').zeros((h, w), dtype='uint8')
    copy_h = min(mh, h)
    copy_w = min(mw, w)
    aligned[:copy_h, :copy_w] = m[:copy_h, :copy_w]
    return aligned, True


def check_dataset(name, instances_path, image_root):
    data = json.load(open(instances_path, 'r'))
    id_to_img = {im['id']: im for im in data['images']}

    total_anns = 0
    decode_errors = 0
    image_errors = 0
    shape_mismatch = 0

    bad_samples = []

    for ann in data['annotations']:
        total_anns += 1
        img_info = id_to_img[ann['image_id']]
        img_path = Path(image_root) / img_info['file_name']

        try:
            w, h = Image.open(img_path).size
        except Exception as exc:
            image_errors += 1
            if len(bad_samples) < 20:
                bad_samples.append({
                    'type': 'image_error',
                    'ann_id': ann['id'],
                    'image_id': ann['image_id'],
                    'file_name': img_info['file_name'],
                    'error': str(exc),
                })
            continue

        seg = ann.get('segmentation')
        segs = seg if isinstance(seg, list) else [seg]

        for idx, s in enumerate(segs):
            try:
                m = decode_mask(s, h, w)
                _, changed = align_mask(m, h, w)
                if changed:
                    shape_mismatch += 1
            except Exception as exc:
                decode_errors += 1
                if len(bad_samples) < 20:
                    bad_samples.append({
                        'type': 'decode_error',
                        'ann_id': ann['id'],
                        'image_id': ann['image_id'],
                        'file_name': img_info['file_name'],
                        'seg_index': idx,
                        'seg_type': type(s).__name__,
                        'error': str(exc),
                    })

    print(f'[{name}] total_annotations={total_anns}')
    print(f'[{name}] image_errors={image_errors}, decode_errors={decode_errors}, shape_mismatch={shape_mismatch}')
    if bad_samples:
        print(f'[{name}] examples:')
        for s in bad_samples:
            print(s)

    return {
        'dataset': name,
        'total_annotations': total_anns,
        'image_errors': image_errors,
        'decode_errors': decode_errors,
        'shape_mismatch': shape_mismatch,
        'ok': image_errors == 0 and decode_errors == 0,
    }


def main():
    root = Path('data/ROS-Sa2VA')

    results = []
    results.append(
        check_dataset(
            'RIS-LAD',
            root / 'RIS-LAD' / 'ris_lad' / 'instances.json',
            root / 'RIS-LAD' / 'images' / 'ris_lad',
        )
    )
    results.append(
        check_dataset(
            'RRSIS-D',
            root / 'RRSIS-D' / 'rrsisd' / 'instances.json',
            root / 'RRSIS-D' / 'images' / 'rrsisd' / 'JPEGImages',
        )
    )

    ok = all(r['ok'] for r in results)
    print('\nSUMMARY')
    for r in results:
        print(r)
    if ok:
        print('CHECK_RESULT: PASS')
    else:
        print('CHECK_RESULT: FAIL')


if __name__ == '__main__':
    main()
