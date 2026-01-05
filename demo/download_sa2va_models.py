from huggingface_hub import snapshot_download
from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()
PRETRAINED_MODELS_PATH = BASE_PATH / "pretrained"

snapshot_download(repo_id="ByteDance/Sa2VA-1B", local_dir=PRETRAINED_MODELS_PATH / "Sa2VA-1B")