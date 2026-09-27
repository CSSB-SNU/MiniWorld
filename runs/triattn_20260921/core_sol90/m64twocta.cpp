#include <torch/extension.h>
at::Tensor sol_forward_m64twocta(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_m64twocta);}
