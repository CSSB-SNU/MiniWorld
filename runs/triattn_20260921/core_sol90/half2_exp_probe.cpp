#include <torch/extension.h>
at::Tensor probe_exp(at::Tensor,int);
at::Tensor probe_cost(at::Tensor,int,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("map",&probe_exp);m.def("cost",&probe_cost);}
