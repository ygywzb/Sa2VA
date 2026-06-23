from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
VIS_ROOT = ROOT / "work_dirs" / "visualize"
OUT_ROOT = ROOT / "work_dirs" / "qual_preview"


CASE_SPECS = [
    {
        "dataset": "revos",
        "label": "revos_mose_exp56",
        "case_dir": VIS_ROOT / "revos_case_pack" / "MOSE__train__c28d81a9" / "exp_56",
        "frames": ["00005", "00015", "00025"],
    },
    {
        "dataset": "revos",
        "label": "revos_lvvis_exp14",
        "case_dir": VIS_ROOT / "revos_case_pack" / "LV-VIS__train__02698" / "exp_14",
        "frames": ["00003", "00010", "00018"],
    },
    {
        "dataset": "revos",
        "label": "revos_ovis_exp15",
        "case_dir": VIS_ROOT / "revos_case_pack" / "OVIS__train__5ba84e3f" / "exp_15",
        "frames": ["img_0000015", "img_0000038", "img_0000065"],
    },
    {
        "dataset": "mevis",
        "label": "mevis_boydog_exp9",
        "case_dir": VIS_ROOT / "mevis_case_pack" / "6eac00b5f389" / "exp_9",
        "frames": ["00008", "00018", "00030"],
    },
    {
        "dataset": "mevis",
        "label": "mevis_horse_exp12",
        "case_dir": VIS_ROOT / "mevis_case_pack" / "f610df51c78a" / "exp_12",
        "frames": ["00010", "00036", "00062"],
    },
    {
        "dataset": "mevis",
        "label": "mevis_turtle_exp4",
        "case_dir": VIS_ROOT / "mevis_case_pack" / "9f542dded87c" / "exp_4",
        "frames": ["00008", "00030", "00052"],
    },
]


ROW_SPECS = [
    ("frames", "Frame"),
    ("baseline_overlay", "Baseline"),
    ("ours_overlay", "Ours"),
    ("gt_overlay", "GT"),
]


def load_meta(case_dir: Path) -> dict:
    return json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))


def fit_image(path: Path, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", tile_size, "white")
    img.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
    offset = ((tile_w - img.width) // 2, (tile_h - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def build_sheet(case_dir: Path, frame_ids: Iterable[str], out_path: Path) -> None:
    meta = load_meta(case_dir)
    frame_ids = list(frame_ids)

    tile_size = (320, 180)
    label_w = 120
    top_h = 88
    gap = 10
    left_pad = 18
    right_pad = 18
    bottom_pad = 18

    grid_w = len(frame_ids) * tile_size[0] + (len(frame_ids) - 1) * gap
    grid_h = len(ROW_SPECS) * tile_size[1] + (len(ROW_SPECS) - 1) * gap
    canvas_w = left_pad + label_w + grid_w + right_pad
    canvas_h = top_h + grid_h + bottom_pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{meta['video_id']} / exp_{meta['exp_id']}"
    prompt = f"Prompt: {meta['prompt']}"
    ours_jf = meta["scores"]["ours"]["JF"]
    base_jf = meta["scores"]["baseline"]["JF"]
    delta_jf = meta["scores"]["delta"]["JF"]
    score_line = f"JF ours={ours_jf} | baseline={base_jf} | delta={delta_jf}"

    draw.text((left_pad, 14), title, fill="black")
    draw.text((left_pad, 36), prompt, fill="black")
    draw.text((left_pad, 58), score_line, fill="black")

    for col, frame_id in enumerate(frame_ids):
        x = left_pad + label_w + col * (tile_size[0] + gap)
        draw.text((x + 6, top_h - 24), frame_id, fill="black")

    for row, (subdir, row_label) in enumerate(ROW_SPECS):
        y = top_h + row * (tile_size[1] + gap)
        draw.text((left_pad, y + tile_size[1] // 2 - 8), row_label, fill="black")

        for col, frame_id in enumerate(frame_ids):
            x = left_pad + label_w + col * (tile_size[0] + gap)
            img_path = case_dir / subdir / f"{frame_id}.png"
            tile = fit_image(img_path, tile_size)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_size[0], y + tile_size[1]), outline=(180, 180, 180), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=75, optimize=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in CASE_SPECS:
        out_path = OUT_ROOT / f"{spec['label']}.jpg"
        build_sheet(spec["case_dir"], spec["frames"], out_path)
        manifest.append(
            {
                "dataset": spec["dataset"],
                "label": spec["label"],
                "case_dir": str(spec["case_dir"].relative_to(ROOT)),
                "frames": spec["frames"],
                "preview": str(out_path.relative_to(ROOT)),
            }
        )

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
