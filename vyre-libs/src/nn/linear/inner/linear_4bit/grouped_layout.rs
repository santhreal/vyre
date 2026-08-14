//! Workgroup geometry and buffer identity shared by every grouped INT4
//! lowering strategy.

pub(super) const AFFINE_GROUPED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
pub(super) const AFFINE_GROUPED_LANES_PER_OUTPUT: u32 = 32;
pub(super) const AFFINE_GROUPED_WARPS_PER_WORKGROUP: u32 =
    AFFINE_GROUPED_WORKGROUP_SIZE[0] / AFFINE_GROUPED_LANES_PER_OUTPUT;
pub(super) const AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP: u32 = AFFINE_GROUPED_WARPS_PER_WORKGROUP;
pub(super) const AFFINE_GROUPED_OP_ID: &str = "vyre-libs::nn::linear_4bit_affine_grouped";
pub(super) const AFFINE_GROUPED_WEIGHT_TILE: &str = "linear_4bit_weight_tile";
