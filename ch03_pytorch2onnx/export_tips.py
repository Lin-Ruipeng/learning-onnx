"""条件推理
我们希望一个模型当它是Pytorch模型时和它是ONNX模型时
可以进行 **不同** 的推理流程!
"""

import torch


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        # 请注意下面这行条件判断! 判定运行时是pt还是onnx
        if torch.onnx.is_in_onnx_export():
            # 仅在模型导出时把输出张量的数值限制在[0, 1]之间
            # 能让我们方便地在代码中添加和模型部署相关的逻辑
            # 但是对只关心模型训练的开发者和用户来说很不友好
            # 突兀的部署逻辑会降低代码整体的可读性
            x = torch.clip(x, 0, 1)
        return x


"""
利用中断张量跟踪的操作
PyTorch 转 ONNX 的跟踪导出法是不是万能的。
如果我们在模型中做了一些很“出格”的操作
跟踪法(trace)会把某些取决于输入的中间结果变成常量
从而使导出的ONNX模型和原来的模型有出入。
"""


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * x[0].item()
        return x, torch.Tensor([i for i in x])


"""
使用了.item()把 torch 中的张量转换成了普通的 Python 变量
还尝试遍历 torch 张量，并用一个列表新建一个 torch 张量。
这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。 
---
另一方面，我们也可以利用这个性质
在保证正确性的前提下令模型的中间结果变成常量。
这个技巧常常用于模型的静态化上
即令模型中所有的张量形状都变成常量。
---
使用张量为输入 (PyTorch版本 < 1.9.0)
"""
