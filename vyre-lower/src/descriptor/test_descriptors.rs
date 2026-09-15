//! The one-buffer descriptor fixture every descriptor and analysis unit test builds on.
//!
//! WHY: the same struct literal was written out in `descriptor/` and again in
//! `analyses/op_histogram/`, so a field added to `KernelDescriptor` had to be
//! chased through both. It is stated once here, through the public builder, so a
//! test that needs a different shape says which field differs instead of
//! restating the whole descriptor.

use vyre_foundation::ir::DataType;

use crate::descriptor_builder::{body, descriptor, global_rw};
use crate::{KernelBody, KernelDescriptor, KernelOp, LiteralValue};

/// One read-write global `u32` buffer named `buf`, 64 threads, literal pool `[7]`.
pub(crate) fn build(ops: Vec<KernelOp>, child_bodies: Vec<KernelBody>) -> KernelDescriptor {
    descriptor("k")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(7)])
                .ops(ops)
                .children(child_bodies),
        )
        .build()
}
