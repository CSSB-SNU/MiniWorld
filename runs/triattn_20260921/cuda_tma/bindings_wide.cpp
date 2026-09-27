#include <torch/extension.h>
void prologue_wide64_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
void prologue_wide128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prologue64",&prologue_wide64_cuda);m.def("prologue128",&prologue_wide128_cuda);}
