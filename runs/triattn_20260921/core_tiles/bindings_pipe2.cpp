#include <torch/extension.h>
at::Tensor core_pipe2_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&core_pipe2_cuda);}
