# 第 2 章：解决模型部署中的难题

来源：[`02_challenges.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/02_challenges.md)

## 本章要点

- **问题**：第 1 章的 SRCNN 把 `upscale_factor=3` 写死在模型里。用户只有一个 `.onnx` 文件，却希望放大倍数能自由设置 —— 不改 `.onnx` 的前提下做不到。
- **思路**：把放大倍数从"模型参数"变成"推理输入"。去掉 `nn.Upsample`，改用 `F.interpolate(x, size=...)`，让放大后的尺寸由第二个输入张量决定。
- **卡点**：PyTorch 的 ONNX 导出并不支持「动态的 interpolate 缩放系数」，直接导出会失败或在 ORT 里报错。这正是**算子支持不全**这一类部署难题的最小样例。
- **出路**：为 PyTorch 补充 ONNX 算子（symbolic function / `autograd.Function`），第 4 章展开。

## 代码与产物

本章的模型改造（动态 `size` 输入）是第 4 章自定义算子的前置，代码将在 `ch04_onnx_custom_op/` 中一并实现：

```text
ch04_onnx_custom_op/
├── dynamic_upsample.py     # 用 interpolate(size=...) 改造后的 SRCNN
└── outputs/
```

## 笔记

<!-- 待补：导出时的原始报错信息、PyTorch 版本、以及 interpolate → Resize 的符号映射关系 -->
