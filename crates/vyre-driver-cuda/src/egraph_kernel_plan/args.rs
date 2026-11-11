//! CUDA e-graph kernel argument table builders.

use crate::backend::staging_reserve::reserve_smallvec;
use smallvec::SmallVec;
use vyre_driver::BackendError;

use super::{
    CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_PARAM_COUNT,
};

pub(super) struct EGraphStructuralKernelArgs {
    pub(super) row_eclass_ids_ptr: u64,
    pub(super) row_language_op_ids_ptr: u64,
    pub(super) row_children_offsets_ptr: u64,
    pub(super) row_children_lens_ptr: u64,
    pub(super) row_signatures_ptr: u64,
    pub(super) children_ptr: u64,
    pub(super) bucket_words_ptr: u64,
    pub(super) bucket_rows_ptr: u64,
    pub(super) output_pairs_ptr: u64,
    pub(super) output_count_ptr: u64,
    pub(super) bucket_index: u32,
    pub(super) first_pair: u64,
    pub(super) pair_count: u64,
}

macro_rules! impl_kernel_args {
    ($type:ty, $count:expr, $context:literal, [$($field:ident),+ $(,)?]) => {
        impl $type {
            pub(super) fn write_kernel_args_into(
                &mut self,
                args: &mut SmallVec<[*mut std::ffi::c_void; 8]>,
            ) -> Result<(), BackendError> {
                reserve_egraph_kernel_args(args, $count, $context)?;
                $(
                    args.push(&mut self.$field as *mut _ as *mut std::ffi::c_void);
                )+
                Ok(())
            }
        }
    };
}

impl_kernel_args!(
    EGraphStructuralKernelArgs,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_PARAM_COUNT,
    "structural-equivalence",
    [
        row_eclass_ids_ptr,
        row_language_op_ids_ptr,
        row_children_offsets_ptr,
        row_children_lens_ptr,
        row_signatures_ptr,
        children_ptr,
        bucket_words_ptr,
        bucket_rows_ptr,
        output_pairs_ptr,
        output_count_ptr,
        bucket_index,
        first_pair,
        pair_count,
    ]
);

pub(super) struct EGraphCanonicalRewriteKernelArgs {
    pub(super) row_eclass_ids_ptr: u64,
    pub(super) children_ptr: u64,
    pub(super) rewrite_words_ptr: u64,
    pub(super) rewrite_count: u32,
    pub(super) row_count: u32,
    pub(super) child_count: u32,
    pub(super) first_item: u64,
}

impl_kernel_args!(
    EGraphCanonicalRewriteKernelArgs,
    CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_PARAM_COUNT,
    "canonical-rewrite",
    [
        row_eclass_ids_ptr,
        children_ptr,
        rewrite_words_ptr,
        rewrite_count,
        row_count,
        child_count,
        first_item,
    ]
);

pub(super) struct EGraphSignatureRefreshKernelArgs {
    pub(super) row_language_op_ids_ptr: u64,
    pub(super) row_children_offsets_ptr: u64,
    pub(super) row_children_lens_ptr: u64,
    pub(super) row_signatures_ptr: u64,
    pub(super) children_ptr: u64,
    pub(super) row_count: u32,
    pub(super) first_row: u64,
}

impl_kernel_args!(
    EGraphSignatureRefreshKernelArgs,
    CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_PARAM_COUNT,
    "signature-refresh",
    [
        row_language_op_ids_ptr,
        row_children_offsets_ptr,
        row_children_lens_ptr,
        row_signatures_ptr,
        children_ptr,
        row_count,
        first_row,
    ]
);

fn reserve_egraph_kernel_args(
    args: &mut SmallVec<[*mut std::ffi::c_void; 8]>,
    arg_count: usize,
    context: &'static str,
) -> Result<(), BackendError> {
    args.clear();
    reserve_smallvec(args, arg_count, context)
}
