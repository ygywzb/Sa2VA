# Sa2VA for Demo
对Sa2VA提供的Demo调试，仅支持VSCode，请先安装好`VSCode Python`插件。
## 显卡配置与运行情况
`NVIDIA Tesla P40`：**单帧条件下**，可跑完VQA和Ref-SEG任务。
## 步骤（`Sa2VA-1B`）
1. 根据项目根目录`README.md`，安装环境并激活(uv)，采用最新环境。
```bash
uv sync --extra=latest
source .venv/bin/activate
```
2. 下载模型至`./demo/pretrained/Sa2VA-1B`下
```bash
# 有代理
python ./demo/download_sa2va_models.py
# 无代理
mkdir ./demo/pretrained/Sa2VA-1B
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ByteDance/Sa2VA-1B --local-dir ./demo/pretrained/Sa2VA-1B
```
3. 在`./demo`下需要两个文件夹，`input`存放输入图片，`output`存放输出结果。
4. 打开`./demo/demo_1B.py`文件，设置断点后，打开VSCode边栏，选择配置名，即可调试。