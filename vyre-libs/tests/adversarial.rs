//! Hostile-input and edge-case tests for vyre-libs (`.internals/skills/testing` **adversarial** category).
//!
//! Full coverage is split across focused binaries for faster iteration:
//! - [`f32_adversarial`](./f32_adversarial.rs)  -  float edge cases
//! - [`op_boundaries`](./op_boundaries.rs)  -  op argument bounds
//! - [`overflow_guards`](./overflow_guards.rs)  -  numeric wrap / reject paths
//!
//! Run the suite:
//! `cargo test -p vyre-libs --test adversarial --test f32_adversarial --test op_boundaries --test overflow_guards`
//!
//! This file is the canonical `--test adversarial` entry so `tests/SKILL.md` and
//! `../../.internals/skills/testing/SKILL.md` align with a named binary.

use vyre_libs::math::linalg::Matmul;
use vyre_libs::TensorRef;

/// The named aggregate always executes a hostile boundary even when the
/// feature-focused sibling binaries are filtered by their own contracts.
#[test]
fn adversarial_entry_rejects_matrix_element_count_overflow() {
    let error = Matmul::new(
        TensorRef::u32_2d("a", 1 << 16, 1 << 16),
        TensorRef::u32_2d("b", 1 << 16, 1),
        TensorRef::u32_2d("out", 1 << 16, 1),
    )
    .build()
    .expect_err("Fix: matrix element-count overflow must be rejected before program creation");

    assert_eq!(
        error.to_string(),
        "TensorRef `a` element-count overflows u32 for shape [65536, 65536]. Fix: reduce dimensions below the u32 boundary."
    );
}
