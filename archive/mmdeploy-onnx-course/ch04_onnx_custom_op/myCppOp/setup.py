from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension

# 编译 C++ 代码
# 模块命名为 my_lib 源文件只有一个 my_add.cpp

# 用脚本所在目录定位源码，保证从任意工作目录执行 setup.py 都能找到 my_add.cpp
HERE = Path(__file__).resolve().parent

setup(
    name="my_add",
    ext_modules=[
        cpp_extension.CppExtension(
            "my_lib",  # 模块名
            [str(HERE / "my_add.cpp")],  # 源码文件
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

# 编译（必须在 ch04_onnx_custom_op/myCppOp 目录下执行，产物 my_lib.*.so 会落在当前目录）:
#   uv run python setup.py build_ext --inplace
# 注意：在仓库根目录执行 setup.py 会因 setuptools 的 flat-layout 自动发现
# （根目录下有多个顶层目录）而报错，教程里的 develop 写法同理。
