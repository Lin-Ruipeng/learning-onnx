# 第 1 章：模型部署简介

来源：[`01_introduction_to_model_deployment.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md)

## 本章要点

- **模型部署是什么**：训练好的权重不能直接用，要转成能在服务端、移动端、特殊设备上高效运行的形态；难点在框架多样性、硬件多样性、性能约束三者的组合。
- **中间表示 ONNX 的作用**：`PyTorch / TF / Paddle` → ONNX → `ONNX Runtime / TensorRT / ncnn / OpenVINO`，把 `M 框架 × N 硬件` 的适配问题拆成 `M + N`。
- **跑通第一个部署流程**：用 PyTorch 实现 SRCNN（超分辨率，放大 3 倍），下载 MMEditing 的权重，先 `torch` 推理，再 `torch.onnx.export`，最后用 `onnxruntime.InferenceSession` 推理并比对两张输出图。

## 代码与产物

本章的示例模型和推理代码与第 3 章是同一份，已合并到：

- 模型定义与权重加载：`common/srcnn.py`
- 完整流程（下载 → torch 推理 → 导出 → 校验 → ORT 推理）：`ch03_pytorch2onnx/export.py`
- 产物：`ch03_pytorch2onnx/outputs/`

```bash
uv run python -m ch03_pytorch2onnx.export
```

## 笔记

<!-- 待补：ONNX Runtime 的 providers 参数怎么选、CPU/CUDA/TensorRT EP 的差别 -->
