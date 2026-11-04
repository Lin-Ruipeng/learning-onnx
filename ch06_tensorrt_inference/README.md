# 第 6 章：TensorRT 模型构建与推理

来源：[`06_introduction_to_tensorrt.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/06_introduction_to_tensorrt.md)

## 本章要点

- **TensorRT 是什么**：NVIDIA 的推理 SDK，把 ONNX 图按具体 GPU 做算子融合、kernel 自动调优、精度校准，产出序列化的 `.engine`/`.plan`。代价是**绑定硬件与版本**，不能跨机器拷。
- **安装**：Windows（zip 解压 + 环境变量）、Linux（deb / pip `tensorrt`）、Jetson（JetPack 自带）三条路。
- **两个阶段**：构建期（`Builder` → `INetworkDefinition` → `ONNX parser` → `IBuilderConfig` → `serialize()`）与运行期（反序列化 engine → `IExecutionContext` → 绑定输入输出显存 → `execute_async_v2` + `cudaStreamSynchronize`）。
- **Python 与 C++ 双份 API**：逻辑一一对应，部署到 C++ 应用时照着翻译即可。
- **动态 shape**：`config.add_optimization_profile()` 给每个输入设 min/opt/max，否则 engine 只吃导出时那一组尺寸（和第 3 章的 `dynamic_axes` 呼应）。

## 代码与产物

```text
ch06_tensorrt_inference/
├── build_engine.py         # onnx -> engine，含 OptimizationProfile
├── infer.py                # 反序列化 engine + 显存绑定 + 推理
├── plugin/                 # （与第 7 章共用）C++ 版本的构建与推理
└── outputs/                # *.engine 已被 .gitignore 忽略，只提交日志与结果图
```

⚠️ 本章依赖本机 NVIDIA 驱动 + TensorRT，不能只靠 `uv sync` 得到；跑之前需要在 `pyproject.toml` 里追加 `tensorrt` 依赖或单独建环境。

## 笔记

<!-- 待补：TRT 版本 / 驱动版本、显存绑定 API 在 TRT 8 与 10 之间的差异（execute_v2 vs setTensorAddress）、动态 shape 报错信息 -->
