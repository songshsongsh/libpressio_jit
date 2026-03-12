#include <torch/extension.h>

__global__ void add_sub_index_Eemb_h_index_Eemb_t_index_Remb_r_kernel(int batch_size, int dim, int nnodes, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Eemb, torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> h, torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> t, int nedges, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Remb, torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> r, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> arr38){
    
    for (int _l4 = (blockIdx.x * blockDim.y); _l4 < ((blockIdx.x * blockDim.y) + (batch_size / (batch_size/16))); _l4 += 16) {
for (int _l5 = (_l4 + threadIdx.y); _l5 < ((_l4 + 16) < batch_size ? ((_l4 + 16)) : (batch_size)); _l5 += blockDim.y) {
for (int _l6 = 0; _l6 < dim; _l6 += 64) {
for (int _l7 = (_l6 + threadIdx.x); _l7 < ((_l6 + 64) < dim ? ((_l6 + 64)) : (dim)); _l7 += blockDim.x) {
arr38[_l5][_l7] = ((Eemb[h[_l5]][(_l7)] - Eemb[t[_l5]][(_l7)]) + Remb[r[_l5]][(_l7)]);
} 
} 
} 
} 

}

torch::Tensor add_sub_index_Eemb_h_index_Eemb_t_index_Remb_r(int batch_size, int dim, int nnodes, torch::Tensor obj_Eemb, torch::Tensor obj_h, torch::Tensor obj_t, int nedges, torch::Tensor obj_Remb, torch::Tensor obj_r)
{   
    torch::Tensor obj_arr38 = torch::empty({batch_size,dim}, torch::TensorOptions(torch::kFloat).device(torch::kCUDA));

    add_sub_index_Eemb_h_index_Eemb_t_index_Remb_r_kernel<<< (batch_size/16), dim3(32,16) >>>(batch_size, dim, nnodes, obj_Eemb.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), obj_h.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(), obj_t.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(), nedges, obj_Remb.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), obj_r.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(), obj_arr38.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    return obj_arr38;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &add_sub_index_Eemb_h_index_Eemb_t_index_Remb_r);
}