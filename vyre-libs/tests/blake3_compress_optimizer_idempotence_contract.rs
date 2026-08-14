//! Optimizer idempotence contract for BLAKE3 compression IR.

mod support;

use support::optimizer::assert_optimizer_is_idempotent;
use vyre_libs::hash::blake3_compress;

#[test]
fn blake3_compress_pre_lowering_optimizer_is_idempotent() {
    assert_optimizer_is_idempotent(blake3_compress("cv_in", "msg", "params", "cv_out"));
}
