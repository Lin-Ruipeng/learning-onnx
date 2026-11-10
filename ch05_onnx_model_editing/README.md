# 第 5 章：ONNX 模型的修改与调试

来源：[`05_onnx_model_editing.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/05_onnx_model_editing.md)

## 本章要点

- **ONNX 的存储格式**：protobuf 定义的 `ModelProto` → `GraphProto` → `NodeProto` / `TensorProto` / `ValueInfoProto`，权重存在 `initializer` 里，用 `onnx.load` / `onnx.save` 读写。
- **不依赖 PyTorch 构造模型**：`onnx.helper.make_tensor` / `make_node` / `make_graph` / `make_model(gen_version)`，手搭一个最小图并跑通 ORT。要显式指定 opset 并用 `make_model_gen_version` 推导 IR，否则新版 onnx 会写出 ORT 还不支持的 IR 版本（如 IR 14）。
- **读取并修改现有模型**：遍历 `graph.node`，按 `op_type` 找到目标节点，改属性 / 插入前后节点，注意 `input` / `output` 名字的重新连线。
- **子模型提取**：`onnx.utils.extract_model(model, new_model, input_names, output_names)`，把大图裁成一段以便单独调试。
- **调试手段**：把中间节点的名字加进 `graph.output`，用 ORT 输出它，再用 `onnx.checker.check_model` 校验修改后的图是否合法。

## 代码与产物

```text
ch05_onnx_model_editing/
├── create_onnx.py       # 纯 onnx.helper 构造 linear_func.onnx（output = a*x + b）
├── modificate_onnx.py   # 读 linear_func.onnx，把 Add 改成 Sub 后另存 linear_func_2.onnx
├── debug_onnx.py        # 导出 whole_model.onnx，再用 extract_model 切子图调试
└── outputs/             # linear_func(_2).onnx、whole_model.onnx(+.data)、partial/submodel_*.onnx
```

三个脚本各自准备所需的模型：前两个用 `create_onnx.py` 导出的 `linear_func.onnx`，
`debug_onnx.py` 会自己导出 `whole_model.onnx`，都不依赖第 1 章的 SRCNN。

## 笔记

<!-- 待补：节点重新连线时容易踩的坑（同名 output、initializer 未被引用导致图校验告警） -->

为什么要学习手搓 ONNX ?
最重要的还是最后的对 ONNX 模型进行 debug 。
如果可以 **输出 ONNX 中间节点的值** 那么就可以定位到精度出现偏差的算子!!!

踩坑记录：

- `torch.onnx.export` 在 torch>=2.9 默认走 dynamo 导出器，中间张量名变成 `x` / `conv1/Conv_output_0` 之类；教程里按 `22`/`28` 切子图的做法需要显式 `dynamo=False`。
- 即使走 TorchScript 导出器，torch 2.15 也不再给中间张量编数字（实际是 `/convs1/convs1.2/Conv_output_0`），所以 `debug_onnx.py` 改为按模型结构解析边界张量，不写死编号。

