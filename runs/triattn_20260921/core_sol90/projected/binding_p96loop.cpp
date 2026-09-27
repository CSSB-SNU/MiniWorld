#include <torch/extension.h>
void projected_p96loop_prepare_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
at::Tensor projected_p96loop_forward_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&projected_p96loop_prepare_cuda);m.def("forward",&projected_p96loop_forward_cuda);}
