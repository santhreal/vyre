//! Generic-semiring matrix multiply  -  the spine of the LEGO substrate.
//!
//! `semiring_gemm` is one Program builder parameterized by a closed semiring
//! choice. It emits IR specialized to that semiring at build time  -  the
//! emitted body contains zero runtime branches over the semiring tag, so
//! Tensor Cores and subgroup-mat-mul intrinsics see the same shape they
//! would for a standard `(×, +)` gemm.
//!
//! # Why this primitive is dual-use
//!
//! Same Program is consumed by dialect callers in this crate AND
//! by vyre's own substrate (`vyre-foundation::transform`):
//!
//! | Semiring | User-dialect consumer | vyre-self consumer |
//! |---|---|---|
//! | `Real` (×, +) | every numeric workload | dispatch-cost matrix products |
//! | `MinPlus` (+, min) | shortest-path graphs in `vyre-libs::security` | dependency-graph longest-path for polyhedral fusion |
//! | `MaxPlus` (+, max) | scheduling, rate analysis | critical-path of dispatch graph for megakernel scheduler |
//! | `BoolOr` (∧, ∨) | reachability in `vyre-libs::dataflow` | Region-tree reachability for dataflow fixpoint |
//! | `MaxTimes` (×, max) | Viterbi/HMM forward in ML consumers | rule-conflict probability resolution |
//! | `Provenance` | `vyre-libs::scallop_join` | rule provenance tracking in external analyzer |
//! | `Gf2` (∧, ⊕) | crypto / linear-code dialects | bitset adjacency under XOR closure |
//!
//! Six self-consumers, six user-dialect consumers  -  clears the recursion-thesis
//! bar from day 1.
//!
//! # Algorithm
//!
//! ```text
//! C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
//! ```
//!
//! where `⊕` is the additive (accumulate) op, `⊗` is the multiplicative
//! (combine) op, and the accumulator initializes to the additive identity.
//! The flat invocation `t = i*N + j` covers `M*N` output cells; the inner
//! `k` loop runs serially per lane.
//!
//! # Variant Boundaries
//!
//! Block-tiled, sparse-adjacency, and user-supplied combine/accumulate
//! forms are distinct registered ops. This module's contract is the
//! dense enum-specialized semiring GEMM over the seven well-known
//! semirings.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};
pub use vyre_spec::Semiring;

use crate::builder::gemm::ContractionComposer;
use crate::plumbing::operand::tensor_ref::TensorRef;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::math::semiring_gemm";

mod wide;
pub use wide::{semiring_gemm_wide, SEMIRING_GEMM_WIDE_WORKGROUP_SIZE};

/// Emit a generic-semiring `M × K · K × N → M × N` matmul Program.
///
/// `a` is laid out row-major with stride `k` (`A[i, kk] = a[i*k + kk]`).
/// `b` is laid out row-major with stride `n` (`B[kk, j] = b[kk*n + j]`).
/// `c` is laid out row-major with stride `n` (`C[i, j] = c[i*n + j]`).
/// All buffers are `u32`. For non-integer semirings, callers encode their
/// own fixed-point scaling on top.
///
/// # Panics
///
/// Panics if any of `m`, `n`, `k` is zero.
#[must_use]
pub fn semiring_gemm(
    a: &str,
    b: &str,
    c: &str,
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Program {
    if m == 0 {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm requires m > 0, got {m}."),
        );
    }
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm requires n > 0, got {n}."),
        );
    }
    if k == 0 {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm requires k > 0, got {k}."),
        );
    }

    if m.checked_mul(n).is_none() {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm output cells overflow u32: m={m}, n={n}."),
        );
    }
    if m.checked_mul(k).is_none() {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm A buffer cells overflow u32: m={m}, k={k}."),
        );
    }
    if k.checked_mul(n).is_none() {
        return trap_program(
            OP_ID,
            Some((c, DataType::U32)),
            format!("Fix: semiring_gemm B buffer cells overflow u32: k={k}, n={n}."),
        );
    }

    let a_ref = TensorRef::u32_2d(a, m, k);
    let b_ref = TensorRef::u32_2d(b, k, n);
    let c_ref = TensorRef::u32_2d(c, m, n);

    ContractionComposer::semiring_2d(OP_ID, a_ref, b_ref, c_ref, m, k, n, semiring)
        .with_region_generator(OP_ID)
        .build()
        .unwrap_or_else(|err| trap_program(OP_ID, Some((c, DataType::U32)), format!("Fix: {err}")))
}

/// CPU reference - exact byte-for-byte target the GPU dispatch must hit.
#[must_use]
#[cfg(test)]
pub(crate) fn semiring_gemm_cpu(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Vec<u32> {
    let mut c = Vec::new();
    try_semiring_gemm_cpu_into(a, b, m, n, k, semiring, &mut c)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - semiring_gemm_cpu failed: invalid GEMM shape");
    c
}

/// CPU reference using a caller-owned output buffer.
#[cfg(test)]
pub(crate) fn semiring_gemm_cpu_into(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    c: &mut Vec<u32>,
) {
    try_semiring_gemm_cpu_into(a, b, m, n, k, semiring, c)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - semiring_gemm_cpu_into failed: invalid GEMM shape");
}

/// Fallible CPU reference using a caller-owned output buffer.
#[cfg(test)]
pub(crate) fn try_semiring_gemm_cpu_into(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    c: &mut Vec<u32>,
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err(format!(
            "semiring_gemm CPU oracle requires non-zero dimensions, got m={m}, n={n}, k={k}."
        ));
    }
    let m_usize =
        usize::try_from(m).map_err(|_| format!("semiring_gemm m={m} does not fit usize."))?;
    let n_usize =
        usize::try_from(n).map_err(|_| format!("semiring_gemm n={n} does not fit usize."))?;
    let k_usize =
        usize::try_from(k).map_err(|_| format!("semiring_gemm k={k} does not fit usize."))?;
    let cell_count = m_usize
        .checked_mul(n_usize)
        .ok_or_else(|| format!("semiring_gemm CPU oracle output cells overflow: m={m}, n={n}."))?;
    m_usize.checked_mul(k_usize).ok_or_else(|| {
        format!("semiring_gemm CPU oracle A buffer cells overflow: m={m}, k={k}.")
    })?;
    k_usize.checked_mul(n_usize).ok_or_else(|| {
        format!("semiring_gemm CPU oracle B buffer cells overflow: k={k}, n={n}.")
    })?;
    if cell_count > c.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            c,
            cell_count - c.len(),
            "semiring GEMM CPU oracle",
            "output matrix",
        )?;
    }
    vyre_reference::composition_witness::semiring_gemm_witness_into(
        a, b, m_usize, n_usize, k_usize, semiring, c,
    );
    Ok(())
}

fn fixture_u32(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || semiring_gemm("a", "b", "c", 2, 2, 2, Semiring::Real),
        Some(|| vec![vec![
            fixture_u32(&[1, 2, 3, 4]),
            fixture_u32(&[5, 6, 7, 8]),
        ]]),
        Some(|| vec![vec![crate::fixture_bytes::MATMUL_2X2_EXPECTED_BYTES.to_vec()]]),
    )
    .with_laws(&["distributive"])
}

#[cfg(test)]
#[path = "../../../tests/internal/math/semiring_gemm/mod.rs"]
mod tests;
