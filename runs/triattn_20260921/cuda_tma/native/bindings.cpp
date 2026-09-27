#include <torch/extension.h>
void prologue_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
void epilogue_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prologue",&prologue_cuda);m.def("epilogue",&epilogue_cuda);}
