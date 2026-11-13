# learning-onnx

这是一个面向实践的 ONNX 部署学习仓库：一边完整保留早期的 MMDeploy 学习记录，一边建设一套可执行、可测试、可持续升级的现代 ONNX 课程。

## 两条学习路径

- [Historical Course Archive](archive/mmdeploy-onnx-course/README.md)：基于 MMDeploy ONNX 教程前五章的已完成学习项目。它作为历史快照保留，不再追赶新 API。
- [Modern ONNX Course](modern-onnx-course/README.md)：以 `torch.export` 导出器、ONNX/ONNX Script 和 ONNX Runtime 为通用主线的活教程。

## 课程架构

```text
Modern ONNX Core Course
    ↓
Advanced Graph Engineering + TCN Deployment Case Study
    ↓
Conditional Backend Labs
    ├─ RK3588 Linux / RKNN（第一个后端轨）
    ├─ HarmonyOS / MindSpore Lite
    ├─ TensorRT
    └─ OpenVINO
```

通用课程必须在 CPU + ONNX Runtime 上可验证。GPU、NPU 和真机实验使用能力检测、条件跳过和独立的 Hardware Acceptance Report，不会伪装成已由通用 CI 验证。

## 现在开始

```bash
cd modern-onnx-course
uv sync --locked
uv run pytest
uv run python core/ch00_course_protocol/src/verify_environment.py
```

课程领域语言见 [GLOSSARY.md](GLOSSARY.md)，长期结构决策见 [docs/adr/](docs/adr/)。

## 路线图

- [x] M1：归档旧课程，建立现代课程骨架与 Chapter 00
- [ ] M2：完成 Modern ONNX Core Course
- [ ] M3：完成 Advanced Graph Engineering 与 TCN 贯穿案例
- [ ] M4：完成 RK3588 Linux Backend Track
- [ ] 后续：按需建设 HarmonyOS、TensorRT 和 OpenVINO Backend Track

## 许可证

仓库的代码与笔记以 [MIT](LICENSE) 许可发布。历史课程引用的 MMDeploy 教程来源及授权说明保留在归档中。
