# 第 7 章：TensorRT 自定义插件

来源：[`07_write_a_plugin.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/07_write_a_plugin.md)

## 本章要点

- **为什么需要插件**：TensorRT 只认自己实现的算子，ONNX 里的自定义算子 / 新算子会解析失败（`Unsupported ONNX op`）。插件 = 给 TRT 补一个算子实现，并在构建 engine 时注册。
- **先在 ONNX 里造节点**：用一个不冲突的 domain（MMDeploy 用 `mmdeploy::`），`onnx.helper.make_node("HardSwish", ..., domain="mmdeploy")`，导出后 TRT 会把它当插件节点处理。
- **C++ 侧的三个类**：
  - `IPluginV2DynamicExt`（或 TRT 10 的 `IPluginV3`）—— 实现 `getOutputDimensions`、`enqueue`（TRT 10 是 `enqueue`/`supportsFormatCombination`）、`configurePlugin`；
  - `IPluginCreator` —— 负责 `createPlugin` / `deserializePlugin`，名字要和 ONNX 节点里的 op、插件 registry 名字对得上；
  - 注册：`REGISTER_TENSORRT_PLUGIN(...)` + 在构建 engine 前 `initLibNvInferPlugins` / 手动 `getPluginRegistry()->registerCreator`。
- **测试**：编译成 `.so` → 构建 engine → 与 PyTorch 参考实现比对数值。

## 代码与产物

```text
ch07_tensorrt_plugin/
├── onnx_with_custom_op.py   # 生成带 mmdeploy::HardSwish 节点的 onnx
├── plugin/                  # C++ 实现（照抄 MMDeploy 的最小可编译版本）
│   ├── hard_swish_plugin.cpp
│   ├── hard_swish_plugin.hpp
│   └── CMakeLists.txt
├── test_plugin.py           # 构建 engine + 推理 + 与 torch 对比
└── outputs/                 # *.engine、*.so 已被 .gitignore 忽略
```

⚠️ 与第 6 章相同，需要本机 TensorRT 与 CUDA toolkit。

## 笔记

<!-- 待补：TRT 10 的 IPluginV3 与旧 IPluginV2DynamicExt 的接口差异、插件属性序列化（serialize/deserialize 必须自洽） -->
