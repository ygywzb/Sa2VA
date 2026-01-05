# Sa2VA for Demo
对Sa2VA提供的Demo调试，仅支持VSCode，请先安装好`VSCode Python`插件。
## 步骤（`Sa2VA-1B`）
1. 根据项目根目录`README.md`，安装环境并激活(uv)
2. 运行脚本`download_sa2va_models.py`，下载模型
```bash
# 模型会下载至`./demo/pretrained/Sa2VA-1B`下
python ./demo/download_sa2va_models.py
```
3. 在`./demo`下需要两个文件夹，`input`存放输入图片，`output`存放输出结果。
4. 编写调试配置，具体如下：
```json
// in .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "sa2va demo 调试(1B)",
            "type": "debugpy",
            "request": "launch",
            "program": "./demo/demo_1B.py",
            "args": [
                "./demo/input",
                "--model_path", "./demo/pretrained/Sa2VA-1B",
                "--work-dir", "./demo/output",
                "--text", "<image>Please describe the video content.",
                "--select", "1"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        }
    ]
}
```
5. 打开`./demo/demo_1B.py`文件，设置断点后，打开VSCode边栏，选择配置名，即可调试。