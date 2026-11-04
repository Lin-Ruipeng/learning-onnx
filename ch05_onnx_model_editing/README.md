# 第 5 章：ONNX 模型的修改与调试

来源：[`05_onnx_model_editing.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/05_onnx_model_editing.md)

## 本章要点

- **ONNX 的存储格式**：protobuf 定义的 `ModelProto` → `GraphProto` → `NodeProto` / `TensorProto` / `ValueInfoProto`，权重存在 `initializer` 里，用 `onnx.load` / `onnx.save` 读写。
- **不依赖 PyTorch 构造模型**：`onnx.helper.make_tensor` / `make_node` / `make_graph` / `make_model`，手搭一个最小图并跑通 ORT。
- **读取并修改现有模型**：遍历 `graph.node`，按 `op_type` 找到目标节点，改属性 / 插入前后节点，注意 `input` / `output` 名字的重新连线。
- **子模型提取**：`onnx.utils.extract_model(model, new_model, input_names, output_names)`，把大图裁成一段以便单独调试。
- **调试手段**：把中间节点的名字加进 `graph.output`，用 ORT 输出它，再用 `onnx.checker.check_model` 校验修改后的图是否合法。

## 代码与产物

```text
ch05_onnx_model_editing/
├── build_from_scratch.py   # 纯 onnx.helper 构造一个模型
├── edit_graph.py           # 读 srcnn.onnx，加/改节点后重新保存
├── extract_submodel.py     # 子模型提取 + 中间节点输出
└── outputs/                # build_from_scratch.onnx、srcnn_edited.onnx、中间特征图
```

输入模型复用第 1 章导出的 `ch01_introduction_to_deployment/outputs/srcnn.onnx`（只读）。

## 笔记

<!-- 待补：节点重新连线时容易踩的坑（同名 output、initializer 未被引用导致图校验告警） -->
