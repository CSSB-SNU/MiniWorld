#include <torch/extension.h>
void epilogue_pipeline_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
void epilogue_pipeline128_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
void epilogue_split_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("split",&epilogue_split_cuda);m.def("epi128",&epilogue_pipeline128_cuda);m.def("epilogue",&epilogue_pipeline_cuda);}
