import torch
torch.manual_seed(0)

from torch.utils.cpp_extension import load_inline

# ------------------------------
# CUDA kernel + C++ wrapper
# ------------------------------
cuda_source = r'''
__global__ void generate_anchors_kernel(const float* base_anchors,  // (A, 4)
                                        int num_base_anchors,
                                        int feat_h, int feat_w,
                                        float stride_h, float stride_w,
                                        float* output_anchors)  // (feat_h * feat_w * A, 4)
{
    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int A = num_base_anchors;
    int total = feat_h * feat_w * A;

    if (grid_idx >= total) return;

    // Indexing
    int a_idx = grid_idx % A;
    int loc_idx = grid_idx / A;
    int y_idx = loc_idx / feat_w;
    int x_idx = loc_idx % feat_w;

    // Shift
    float shift_x = x_idx * stride_w;
    float shift_y = y_idx * stride_h;

    // Anchor
    const float* anchor = &base_anchors[a_idx * 4];

    float x1 = anchor[0] + shift_x;
    float y1 = anchor[1] + shift_y;
    float x2 = anchor[2] + shift_x;
    float y2 = anchor[3] + shift_y;

    // Write output
    float* out = &output_anchors[grid_idx * 4];
    out[0] = x1;
    out[1] = y1;
    out[2] = x2;
    out[3] = y2;
}

torch::Tensor generate_anchors(torch::Tensor base_anchors,
                               int feat_h, int feat_w,
                               float stride_h, float stride_w) {
    const int A = base_anchors.size(0);
    const int total = feat_h * feat_w * A;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(base_anchors.device());
    auto output = torch::empty({total, 4}, options);

    dim3 threads(256);
    dim3 blocks((total + threads.x - 1) / threads.x);

    generate_anchors_kernel<<<blocks, threads>>>(
        base_anchors.data_ptr<float>(), A, feat_h, feat_w, stride_h, stride_w, output.data_ptr<float>()
    );

    return output;
}
'''

cpp_source = "torch::Tensor generate_anchors(torch::Tensor base_anchors, int feat_h, int feat_w, float stride_h, float stride_w);"

# ------------------------------
# Load inline CUDA extension
# ------------------------------
anchor_ext = load_inline(
    name="anchor_tiling_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["generate_anchors"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)

# ------------------------------
# Prepare Inputs
# ------------------------------
# 3 scales × 3 aspect ratios = 9 base anchors
scales = [32, 64, 128]
ratios = [0.5, 1.0, 2.0]
base_anchors = []
for scale in scales:
    for ratio in ratios:
        h = scale * (ratio ** 0.5)
        w = scale / (ratio ** 0.5)
        base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])
base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device='cuda')  # (9, 4)

# Feature map config
feat_h, feat_w = 4, 4  # 4x4 grid
stride_h, stride_w = 16.0, 16.0

# ------------------------------
# Run CUDA kernel
# ------------------------------
anchors = anchor_ext.generate_anchors(base_anchors, feat_h, feat_w, stride_h, stride_w)
print("Output anchors:", anchors.shape)
print(anchors[:5])  # Show first few anchors

