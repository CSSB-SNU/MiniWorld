#include <torch/extension.h>
void prologue64_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
void prologue128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
void prologue_wide_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
void prologue_async_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("async2",&prologue_async_cuda);m.def("wide",&prologue_wide_cuda);m.def("pro64",&prologue64_cuda);m.def("pro128",&prologue128_cuda);}
