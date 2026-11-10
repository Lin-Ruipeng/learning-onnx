# 第 2 章：解决模型部署中的难题

来源：[`02_challenges.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/02_challenges.md)

## 本章要点

- **问题**：第 1 章的 SRCNN 把 `upscale_factor=3` 写死在 `nn.Upsample` 里，导出的 ONNX 只能出 3 倍图。用户手里只有一个 `.onnx`，却想自由设置放大倍数 —— 不改模型文件的前提下一开始做不到。
- **思路**：把放大倍数从"模型参数"变成"模型输入"。删掉 `nn.Upsample`，让 `forward(x, upscale_factor)` 多收一个缩放系数张量 `[1, 1, h, w]`。
- **卡点**：PyTorch 的 ONNX 导出不支持「运行时的缩放系数」。`F.interpolate(scale_factor=...)` 在 trace 时要求这个值是常量，导出会失败或退化。
- **解法**：`torch.autograd.Function` 是**一个算子、两条实现路径**——
  - `forward`：PyTorch 里实际计算时走这里（`F.interpolate`）；
  - `symbolic`：导出 ONNX 时走这里，用 `g.op("Resize", ...)` 直接拼出 ONNX 节点，把 `scales` 接成图的第二个输入。

  于是图里多出一个真正的 `Resize` 节点，缩放系数由推理时喂进来。

## 代码与产物

| 文件 | 说明 |
|---|---|
| `export_dynamic.py` | 改造后的 SRCNN（`NewInterpolate` + `forward(x, upscale_factor)`）→ torch 3 倍推理 → 导出（`dynamo=False`）→ `onnx.checker` → ORT 4 倍推理 |
| `outputs/srcnn_dynamic.onnx` | 动态倍数的模型，输入 `input` + `factor`，80KB |
| `outputs/face_torch_x3.png` | PyTorch 3 倍结果，与第 1 章 `face_torch.png` **逐像素相同**（换实现没改语义） |
| `outputs/face_ort_x4.png` | 同一个 `.onnx` 文件、`factor=[1,1,4,4]` 出来的 4 倍图（1024×1024） |

```bash
uv run python -m ch02_deployment_challenges.export_dynamic
```

实测输出：

```
节点: ['Constant', 'Resize', 'Conv', 'Relu', 'Conv', 'Relu', 'Conv']
输入: ['input', 'factor']
Model correct
4 倍放大：PyTorch 与 ONNX Runtime 的最大像素差 0.000
```

模型图从第 1 章的 6 个节点变成 7 个：`nn.Upsample` 那个常量 `Resize` 换成了吃第二路输入的 `Resize`（前面多了个 `Constant` 给 `roi` 占位）。

## 两条警告是怎么回事

跑起来会看到两条警告，都是意料之中的，值得记下来：

**1. `DeprecationWarning: You are using the legacy TorchScript-based ONNX export...`**

因为导出时写了 `dynamo=False`。torch ≥ 2.9 默认走新的 `torch.export` 导出器，而新导出器**不认 `autograd.Function.symbolic`**，会真的去 trace `forward` 里的 `interpolate`，然后报：

```
TypeError: upsample_bicubic2d(): argument 'scale_factors' must be tuple of floats,
but found element of type SymFloat at pos 1
```

也就是说：本章的写法天然依赖旧的 TorchScript 导出器，这条 DeprecationWarning 只能接受。顺带印证了第 3 章要讲的"跟踪法 vs 脚本化"。

**2. `TracerWarning: Converting a tensor to a Python list...`（`scales.tolist()`）**

trace 时 `forward` 真的被执行了一次，`tolist()` 把张量值取成了 Python 数字，tracer 只能把它当常量记下来。但导出的实际路径是 `symbolic`（`g.op` 拼图），所以这条警告**不影响**正确性 —— 上面的 4 倍结果和逐像素对比可以证明。它提醒的是：如果哪天靠 trace `forward` 来导出，这段代码就会被写死成 3 倍。

## 笔记

<!-- 待补：为什么 scales 要 4 维（NCHW 各一维）、roi 传空张量与 sizes/scales 二选一、非整数倍（如 2.5）是否可行 -->
