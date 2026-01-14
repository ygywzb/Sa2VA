# Sa2VA Baseline Train Debug
针对Sa2VA基线的训练代码调试，可根据显卡自行选择是否使用`flash-attn`加速，调试仅适配VSCode，请安装好`VSCode Python`插件。
## 显卡配置与运行情况
`NVIDIA Tesla P40`: 禁用flash-attn下，可顺利跑完Sa2VA训练的`forward`函数。
## 步骤（以Sa2VA-1B为例）
1. 根据项目根目录`README.md`，配置并安装训练所需包，激活环境。
2. 根据项目根目录`README.md`，准备数据集到`./data/`下。
3. 根据项目根目录`README.md`，准备预训练模型至`./pretrained/`下。
4. 对`Sa2VAModel`打断点，打开VSCode侧边栏，选择配置名，即可调试。
## 提示
1. 虚拟环境需要用`最新版本`
```bash
uv sync --extra=latest
```
2. 数据集和预训练模型Sa2VA作者都已经在`Huggingface`上开源，可通过`hf download`命令下载
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 数据集
hf download --repo-type dataset xxxx/yyyyy --local-dir /xxxx/yyyy/zzz
# 模型
hf download xxxx/yyyyy --local-dir /xxxx/yyyy/zzz
```
3. 作者提供的SA-V数据集**不完整(only 000)**，需要根据`README.md`到Meta官网上下载完整版(000 ~ 055)，配置文件中已经将该数据集**注释掉**，不会参与训练代码调试。