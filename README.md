# learning-onnx

跟着 **MMDeploy「模型部署入门教程」** 走一遍：从 PyTorch 导出 ONNX，到自定义算子、图修改调试，再到 TensorRT 的模型构建与插件编写。仓库主体是每章的脚本、可复现的产物，以及自己写的笔记。

## 学习材料出处

全部 7 章内容来自 **OpenMMLab 的 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 仓库**，原始文件位于
[`docs/zh_cn/tutorial/`](https://github.com/open-mmlab/mmdeploy/tree/main/docs/zh_cn/tutorial)（Apache-2.0 许可）。

| 章 | 主题 | 原文 |
|---|---|---|
| 1 | 模型部署简介，把 SRCNN 部署到 ONNX Runtime | [`01_introduction_to_model_deployment.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md) |
| 2 | 部署中的难题：让放大倍数变成运行时输入 | [`02_challenges.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/02_challenges.md) |
| 3 | PyTorch 转 ONNX 详解（`torch.onnx.export`） | [`03_pytorch2onnx.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/03_pytorch2onnx.md) |
| 4 | 在 PyTorch 中支持更多 ONNX 算子 | [`04_onnx_custom_op.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/04_onnx_custom_op.md) |
| 5 | ONNX 模型的修改与调试 | [`05_onnx_model_editing.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/05_onnx_model_editing.md) |
| 6 | TensorRT 模型构建与推理 | [`06_introduction_to_tensorrt.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/06_introduction_to_tensorrt.md) |
| 7 | TensorRT 自定义插件 | [`07_write_a_plugin.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/07_write_a_plugin.md) |

> 本仓库**不转载教程正文**（版权归 OpenMMLab 所有，Apache-2.0）。各章 `README.md` 里是我自己的转述、可运行的脚本和实测结果；读原文请点上表链接。原文若更新，以 GitHub 上的版本为准。

## 目录结构

```
.
├── assets/          # 公共输入样本（face.png）
├── models/          # 公共权重（srcnn.pth，85KB，已入库；缺失时脚本会自动下载）
├── common/          # 跨章节复用：paths / download / srcnn
├── ch01_introduction_to_deployment/   # 第 1 章：简介（代码并入 ch03）
├── ch02_deployment_challenges/        # 第 2 章：动态放大倍数
├── ch03_pytorch2onnx/                 # 第 3 章：导出 → 校验 → ORT 推理  ✅ 已跑通
│   ├── export.py
│   └── outputs/     # srcnn.onnx(+.data)、face_torch.png、face_ort.png
├── ch04_onnx_custom_op/               # 第 4 章：自定义算子
├── ch05_onnx_model_editing/           # 第 5 章：图的读写与调试
├── ch06_tensorrt_inference/           # 第 6 章：TensorRT
└── ch07_tensorrt_plugin/              # 第 7 章：TensorRT 插件
```

## 环境

统一的 Python 环境，**不要在章节目录里各自建 venv**（torch CUDA 一份就几个 G）：

```bash
uv sync          # 按 pyproject.toml + uv.lock 建 .venv
```

## 运行约定

1. **从仓库根目录运行**，并且用 `-m` 形式，这样 `common` 才在 `sys.path` 上：

   ```bash
   uv run python -m ch03_pytorch2onnx.export
   ```

2. **路径只从 `common.paths` 取**（`ASSETS`、`MODELS`、`outputs("ch03_pytorch2onnx")`），脚本里不写 `"face.png"` 这种相对 CWD 的字面量。
3. **每章只写自己的 `chNN_xxx/outputs/`**，不往仓库根目录或 `models/` 写。`models/srcnn.pth` 只读。
4. **入库标准**：<1MB 的样本、权重、导出模型、对比图提交进 git；大体积或可重建的（`*.engine`、`*.plan`、`*.so`、下载缓存）走 `.gitignore` + 脚本重建。
5. 每章的 `README.md` 是笔记主体；本文件只做索引。

## 进度

- [x] 1 模型部署简介
- [ ] 2 部署中的难题
- [x] 3 PyTorch 转 ONNX 详解（`ch03_pytorch2onnx/export.py`）
- [ ] 4 在 PyTorch 中支持更多 ONNX 算子
- [ ] 5 ONNX 模型的修改与调试
- [ ] 6 TensorRT 模型构建与推理
- [ ] 7 TensorRT 自定义插件

## 致谢

- [open-mmlab/mmdeploy](https://github.com/open-mmlab/mmdeploy) —— 教程原文与 C++ 插件实现的来源，Apache-2.0
- [open-mmlab/mmagic](https://github.com/open-mmlab/mmagic) —— SRCNN 权重与 `face.png` 测试图
