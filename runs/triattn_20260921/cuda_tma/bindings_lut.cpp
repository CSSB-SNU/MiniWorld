#include <torch/extension.h>
void epilogue_lut_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void epilogue_packed64_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void epilogue_packed128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed64",&epilogue_packed64_cuda);m.def("packed128",&epilogue_packed128_cuda);m.def("epilogue",&epilogue_lut_cuda);}
