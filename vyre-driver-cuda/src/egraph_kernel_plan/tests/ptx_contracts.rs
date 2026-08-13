use super::*;

#[test]
fn structural_equivalence_kernel_ptx_rejects_invalid_sm_target() {
    assert_eq!(
        cuda_egraph_structural_equivalence_kernel_ptx(0)
            .expect_err("sm_0 is not a valid CUDA PTX target"),
        CudaEGraphKernelPlanError::InvalidPtxTarget { target_sm: 0 }
    );
}

#[test]

fn signature_bucket_planner_rejects_mismatched_image_and_view() {
    let image = GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "lit", &[][..])])
        .try_pack_device_image()
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
    let mismatched_view = synthetic_view(1, 0, 1);

    assert_eq!(
        plan_cuda_egraph_signature_buckets(
            &image,
            mismatched_view,
            CudaEGraphKernelLaunchConfig::default(),
        )
        .expect_err("image/view row mismatch must be rejected"),
        CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "row count",
            image: 2,
            view: 1,
        }
    );
}
