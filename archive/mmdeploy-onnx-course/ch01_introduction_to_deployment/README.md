# 第 1 章：模型部署简介

来源：[`01_introduction_to_model_deployment.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md)

## 本章要点

- **模型部署是什么**：训练好的权重不能直接用，要转成能在服务端、移动端、特殊设备上高效运行的形态；难点在框架多样性、硬件多样性、性能约束三者的组合。
- **中间表示 ONNX 的作用**：`PyTorch / TF / Paddle` → ONNX → `ONNX Runtime / TensorRT / ncnn / OpenVINO`，把 `M 框架 × N 硬件` 的适配问题拆成 `M + N`。
- **跑通第一个部署流程**：用 PyTorch 实现 SRCNN（超分辨率，放大 3 倍），下载 MMEditing 的权重，先 `torch` 推理，再 `torch.onnx.export`，最后用 `onnxruntime.InferenceSession` 推理并比对两张输出图。

## 代码与产物

| 文件 | 说明 |
|---|---|
| `export.py` | 下载权重/测试图 → torch 推理 → `torch.onnx.export` → `onnx.checker` → ORT 推理，并打印两条路径的最大像素差 |
| `outputs/srcnn.onnx` + `srcnn.onnx.data` | 导出结果（外部数据文件必须与 `.onnx` 同目录） |
| `outputs/face_torch.png` | PyTorch 的 3 倍超分结果 |
| `outputs/face_ort.png` | ONNX Runtime 的同一张图，应与上一张肉眼无差 |

模型定义与权重加载第一、二章都在用，已抽到 `common/srcnn.py`（第 3 章只分析它的导出产物，不 import 这个模块）。

```bash
uv run python -m ch01_introduction_to_deployment.export
```

实测（torch 2.15 + onnxruntime 1.31，CPU）：`Model correct`，两条路径输出最大像素差 1/255，肉眼无差。

## 与原教程的差异

- 原文用 `opset_version=11`，这里用 **18**：torch 2.x 默认走 `torch.export` 导出器，低 opset 会被降级/告警。
- 原文把下载、推理、导出写成顶层脚本；这里的 `export.py` 拆成函数 + `__main__` 守卫，路径由 `common.paths` 提供（原文依赖 CWD）。
- 原文 ORT 推理用默认 provider；这里显式指定 `CPUExecutionProvider`，避免 CUDA EP 的初始化开销。

## 笔记

<!-- 待补：ONNX Runtime 的 providers 参数怎么选、CPU/CUDA/TensorRT EP 的差别 -->
