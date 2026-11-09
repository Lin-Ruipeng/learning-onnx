# learning-onnx

跟着 **MMDeploy「模型部署入门教程」** 走一遍：从 PyTorch 导出 ONNX，到自定义算子、图修改调试。仓库主体是每章的脚本、可复现的产物，以及自己写的笔记。

## 学习材料出处

原文教程共 7 章，本仓库只跟进与 ONNX 直接相关的前 5 章——第 6 章（TensorRT 模型构建与推理）与第 7 章（TensorRT 自定义插件）不在学习范围内。

前 5 章内容来自 **OpenMMLab 的 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 仓库**，原始文件位于
[`docs/zh_cn/tutorial/`](https://github.com/open-mmlab/mmdeploy/tree/main/docs/zh_cn/tutorial)（Apache-2.0 许可）。

| 章 | 主题 | 原文 |
|---|---|---|
| 1 | 模型部署简介，把 SRCNN 部署到 ONNX Runtime | [`01_introduction_to_model_deployment.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md) |
| 2 | 部署中的难题：让放大倍数变成运行时输入 | [`02_challenges.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/02_challenges.md) |
| 3 | PyTorch 转 ONNX 详解（`torch.onnx.export`） | [`03_pytorch2onnx.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/03_pytorch2onnx.md) |
| 4 | 在 PyTorch 中支持更多 ONNX 算子 | [`04_onnx_custom_op.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/04_onnx_custom_op.md) |
| 5 | ONNX 模型的修改与调试 | [`05_onnx_model_editing.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/05_onnx_model_editing.md) |

> 本仓库**不转载教程正文**（版权归 OpenMMLab 所有，Apache-2.0）。各章 `README.md` 里是我自己的转述、可运行的脚本和实测结果；读原文请点上表链接。原文若更新，以 GitHub 上的版本为准。

## 目录结构

```
.
├── assets/          # 公共输入样本（face.png）
├── models/          # 公共权重（srcnn.pth，85KB，已入库；缺失时脚本会自动下载）
├── common/          # 跨章节复用：paths / download / image / srcnn
├── ch01_introduction_to_deployment/   # 第 1 章：部署第一个模型  ✅ 已跑通
│   ├── export.py    # 下载 → torch 推理 → 导出 ONNX → 校验 → ORT 推理
│   └── outputs/     # srcnn.onnx(+.data)、face_torch.png、face_ort.png
├── ch02_deployment_challenges/        # 第 2 章：动态放大倍数  ✅ 已跑通
│   ├── export_dynamic.py            # autograd.Function → Resize 节点，倍数变成输入
│   └── outputs/     # srcnn_dynamic.onnx、face_torch_x3.png、face_ort_x4.png
├── ch03_pytorch2onnx/                 # 第 3 章：导出原理（trace/script、symbolic）  ✅ 已跑通
│   ├── trace_and_script.py          # trace vs script：控制流被展开还是保留为 Loop 节点
│   ├── dynamic_axes.py              # 三种 dynamic_axes 导出 + ORT 形状验证
│   ├── export_tips.py               # is_in_onnx_export 条件推理、中断张量跟踪
│   └── outputs/     # model_{2,3}_{trace,script}.onnx、model_static/dynamic_*.onnx
├── ch04_onnx_custom_op/               # 第 4 章：自定义算子
└── ch05_onnx_model_editing/           # 第 5 章：图的读写与调试
```

## 环境

统一的 Python 环境，**不要在章节目录里各自建 venv**（torch CUDA 一份就几个 G）：

```bash
uv sync          # 按 pyproject.toml + uv.lock 建 .venv
```

## 运行约定

1. **从仓库根目录运行**，并且用 `-m` 形式，这样 `common` 才在 `sys.path` 上：

   ```bash
   uv run python -m ch01_introduction_to_deployment.export
   ```

2. **路径只从 `common.paths` 取**（`ASSETS`、`MODELS`、`outputs("ch01_introduction_to_deployment")`），脚本里不写 `"face.png"` 这种相对 CWD 的字面量。
3. **每章只写自己的 `chNN_xxx/outputs/`**，不往仓库根目录或 `models/` 写。`models/srcnn.pth` 只读。
4. **入库标准**：<2MB 的样本与权重（`models/srcnn.pth`）提交进 git；各章 `outputs/` 均为可重建产物、不入库，大体积或可重建的（`*.so`、下载缓存）同样走 `.gitignore` + 脚本重建。
5. 每章的 `README.md` 是笔记主体；本文件只做索引。

## 进度

- [x] 1 模型部署简介（`ch01_introduction_to_deployment/export.py`）
- [x] 2 部署中的难题（`ch02_deployment_challenges/export_dynamic.py`）
- [x] 3 PyTorch 转 ONNX 详解（`ch03_pytorch2onnx/trace_and_script.py`、`dynamic_axes.py`、`export_tips.py`）
- [ ] 4 在 PyTorch 中支持更多 ONNX 算子
- [ ] 5 ONNX 模型的修改与调试

## 致谢

- [open-mmlab/mmdeploy](https://github.com/open-mmlab/mmdeploy) —— 教程原文的来源，Apache-2.0
- [open-mmlab/mmagic](https://github.com/open-mmlab/mmagic) —— SRCNN 权重与 `face.png` 测试图

## 许可证

本仓库的代码与笔记以 [MIT](LICENSE) 许可发布（Copyright (c) 2026 Lin Ruipeng）。
引用的教程原文版权归 OpenMMLab 所有，遵循 Apache-2.0，本仓库不包含其正文。
