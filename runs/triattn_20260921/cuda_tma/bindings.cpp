#include <torch/extension.h>
void prologue_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
void epilogue_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void prologue128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
void epilogue128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void prologue256_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
void epilogue256_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prologue128",&prologue128_cuda);m.def("epilogue128",&epilogue128_cuda);m.def("prologue256",&prologue256_cuda);m.def("epilogue256",&epilogue256_cuda);m.def("prologue",&prologue_cuda);m.def("epilogue",&epilogue_cuda);}
