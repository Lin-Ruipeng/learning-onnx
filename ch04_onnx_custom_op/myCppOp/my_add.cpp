// torch/extension.h = C++ 前端 + pybind11 绑定，写扩展时用它最省事
// (旧教程里的 torch/torch.h 依赖编译参数 TORCH_API_INCLUDE_EXTENSION_H 才会带上 pybind11)
#include <torch/extension.h>

// 自定义的 C++ 算子
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
  // 算子的功能就是返回 2a+b 刻意为之
  return 2 * a + b;
}

// 用 pybind11 来绑定到 python
PYBIND11_MODULE(my_lib, m) {
  // 把 C++ 的 my_add 函数注册到 python 中命名为 "my_add"
  m.def("my_add", my_add);
}
