#include <torch/extension.h>
void projected_prepare_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
void projected_qkv_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("prepare",&projected_prepare_cuda);
 m.def("project",&projected_qkv_cuda);
}
