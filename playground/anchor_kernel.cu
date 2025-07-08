extern "C" __global__
void generate_anchors_kernel(const float* base_anchors,  // (A, 4)
                             int num_base_anchors,
                             int feat_h, int feat_w,
                             float stride_h, float stride_w,
                             float* output_anchors)  // (feat_h * feat_w * A, 4)
{
    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int A = num_base_anchors;
    int total = feat_h * feat_w * A;

    if (grid_idx >= total) return;

    // Global anchor index
    int a_idx = grid_idx % A;
    int loc_idx = grid_idx / A;
    int y_idx = loc_idx / feat_w;
    int x_idx = loc_idx % feat_w;

    // Compute shift
    float shift_x = x_idx * stride_w;
    float shift_y = y_idx * stride_h;

    // Get base anchor
    const float* anchor = &base_anchors[a_idx * 4];

    // Apply shift
    float x1 = anchor[0] + shift_x;
    float y1 = anchor[1] + shift_y;
    float x2 = anchor[2] + shift_x;
    float y2 = anchor[3] + shift_y;

    // Write result
    float* out = &output_anchors[grid_idx * 4];
    out[0] = x1;
    out[1] = y1;
    out[2] = x2;
    out[3] = y2;
}

