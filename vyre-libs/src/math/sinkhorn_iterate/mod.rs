//! Full iterative Sinkhorn balance.
//!
//! Alternates row-normalize and column-normalize until converged.
//! Composes `sinkhorn_scale` + `semiring_gemm` + a persistent fixpoint
//! harness.

mod program;
mod reference;
mod reference_f64;

#[cfg(test)]
mod f64_tests;
#[cfg(test)]
mod tests;

pub use program::sinkhorn_iterate;
#[cfg(any(test, feature = "cpu-parity"))]
pub use reference::{cpu_ref, cpu_ref_into, try_cpu_ref, try_cpu_ref_into};
#[cfg(any(test, feature = "cpu-parity"))]
pub use reference_f64::{
    sinkhorn_col_residual, sinkhorn_iterate_f64, sinkhorn_iterate_f64_into, sinkhorn_row_residual,
    try_sinkhorn_iterate_f64, try_sinkhorn_iterate_f64_into,
};

/// Stable registry id for the iterative Sinkhorn primitive.
pub const OP_ID: &str = "vyre-primitives::math::sinkhorn_iterate";

/// The ten buffer bindings one iterative-Sinkhorn program declares.
///
/// Every one of them is a `&str`, so a positional call of the ten accepted any
/// permutation of them, and one caller took that offer: the crate's own IR
/// parity test passed the names in BINDING order rather than parameter order,
/// so the emitted program named its kernel matrix `u_curr`, its `u` ping-pong
/// half `k_t`, and its convergence flag `kv`. Nothing noticed, because that
/// test feeds `reference_eval` by binding index, where a name is only a label.
/// A consumer that binds by name reads the scaling vector as the kernel.
/// Naming each binding at the construction site is what makes a transposition a
/// diff instead of a silent argument swap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkhornBuffers<'a> {
    /// `m x n` kernel matrix.
    pub k: &'a str,
    /// `n x m` transposed kernel matrix.
    pub k_t: &'a str,
    /// `m` target marginals.
    pub a: &'a str,
    /// `n` target marginals.
    pub b: &'a str,
    /// `m` elements, current half of the `u` ping-pong.
    pub u_curr: &'a str,
    /// `m` elements, next half of the `u` ping-pong.
    pub u_next: &'a str,
    /// `n` elements, current state for `v`.
    pub v: &'a str,
    /// `m` elements of `K v` scratch.
    pub kv: &'a str,
    /// `n` elements of `K_T u` scratch.
    pub ktu: &'a str,
    /// Convergence flag. One element on the single-workgroup form,
    /// `max_iterations` elements on the grid form; see the form note on
    /// [`sinkhorn_iterate`].
    pub changed: &'a str,
}

impl SinkhornBuffers<'static> {
    /// The canonical binding names for a Sinkhorn program.
    ///
    /// A caller that has no naming of its own gets one here instead of
    /// inventing ten strings, and every program built from it declares the
    /// same bindings in the same order, which is what makes two such programs
    /// comparable byte for byte.
    pub const CANONICAL: Self = Self {
        k: "k",
        k_t: "kt",
        a: "a",
        b: "b",
        u_curr: "uc",
        u_next: "un",
        v: "v",
        kv: "kv",
        ktu: "ktu",
        changed: "c",
    };
}

/// The problem extents and iteration cap one iterative-Sinkhorn program is
/// built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkhornExtents {
    /// Row count of the kernel matrix, and the length of `a`, `u_curr`,
    /// `u_next` and `kv`.
    pub m: u32,
    /// Column count of the kernel matrix, and the length of `b`, `v` and `ktu`.
    pub n: u32,
    /// Hard cap on iterations.
    pub max_iterations: u32,
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || {
            sinkhorn_iterate(
                SinkhornBuffers::CANONICAL,
                SinkhornExtents {
                    m: 2,
                    n: 2,
                    max_iterations: 5,
                },
            )
        },
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[65536, 65536]), // u_curr
                to_bytes(&[0, 0]), // u_next
                to_bytes(&[0]), // changed
                to_bytes(&[65536, 65536, 65536, 65536]), // k
                to_bytes(&[65536, 65536, 65536, 65536]), // k_t
                to_bytes(&[32768, 32768]), // a
                to_bytes(&[32768, 32768]), // b
                to_bytes(&[65536, 65536]), // v
                to_bytes(&[0, 0]), // kv
                to_bytes(&[0, 0]), // ktu
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[32768, 32768]), // u_curr
                to_bytes(&[32768, 32768]), // u_next
                to_bytes(&[0]),            // changed
                to_bytes(&[32768, 32768]), // v
                to_bytes(&[0, 0]),         // kv
                to_bytes(&[0, 0]),         // ktu
            ]]
        }),
    )
}
