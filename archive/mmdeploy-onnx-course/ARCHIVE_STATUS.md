# Historical Course Archive

本目录是已完成的 MMDeploy ONNX 学习项目快照。它保留原章节、公共代码、素材、权重、依赖锁和当时的未完成项，用于回顾学习路径和新旧工具链差异。

## 维护状态

- 状态：**Frozen**
- 归档日期：2026-11-13
- 允许的修改：归档说明、路径修正、安全问题与迁移本身导致的破损修复
- 不再进行：API 现代化、依赖追新、章节扩展或后端追赶

新的导出、验证、图工程与后端实践进入 [`../../modern-onnx-course/`](../../modern-onnx-course/README.md)。

## 重建环境

这个归档是独立的 `uv` 项目：

```bash
cd archive/mmdeploy-onnx-course
uv sync --locked
uv run python -m ch01_introduction_to_deployment.export
```

不要移动或共享虚拟环境目录；应由本目录中的 `uv.lock` 重建。
