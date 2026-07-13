//! Sheaf Laplacian eigenvalue primitive (#P-PRIM-9).
//!
//! Extracts the dominant eigenpair of the **diagonal** sheaf Laplacian.
//!
//! The restriction is supplied as a diagonal (`restriction_diag`), so the operator this primitive
//! represents is `diag(r)`: its eigenvalues are exactly the diagonal entries `r[i]` and its
//! eigenvectors are the standard basis vectors `e_i`. The dominant eigenpair is therefore the
//! CLOSED FORM
//!
//! ```text
//!   lambda = max_i r[i]
//!   v      = e_argmax   (unit indicator at the arg-max index; first arg-max on ties)
//! ```
//!
//! This is exact, no power iteration and no square root are needed (a power iteration on a
//! diagonal operator converges to this same eigenpair, immediately). The `iterations` parameter is
//! retained for interface stability but the answer is iteration-independent. All values are 16.16
//! fixed point on the GPU/IR path (`r`, `v`) and `f64` on the CPU reference path.

use std::sync::Arc;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::sheaf_laplacian_eigenvalue";
/// Inner scan-phase op id.
///
/// The `power_iteration_phase` suffix is a LEGACY-STABLE identity string: it is pinned in the
/// generated conformance evidence (`release/evidence/conformance/*.json`), the primitive catalog,
/// and the generated op docs, so it is retained verbatim as an identifier rather than renamed. The
/// phase itself no longer runs a power iteration, it performs the exact single-pass diagonal
/// max/arg-max scan below (the diagonal operator's dominant eigenpair is closed-form).
const POWER_ITERATION_PHASE_OP_ID: &str =
    "vyre-primitives::math::sheaf_laplacian_eigenvalue::power_iteration_phase";

/// Build a sheaf Laplacian eigenvalue Program.
///
/// Inputs:
/// - `restriction_diag`: `n * d` diagonal sheaf Laplacian.
/// - `v`: `n * d` vector; OVERWRITTEN with the dominant eigenvector `e_argmax` (the initial
///   contents are ignored (the diagonal eigenvector does not depend on a starting vector)).
/// - `lambda`: 1-element output eigenvalue = `max_i r[i]`.
///
/// The running max and arg-max are loop-carried locals (`let` + `assign`), not storage scratch
/// buffers, so the Program's only writable outputs are exactly `v` and `lambda`: no internal
/// scratch leaks across the dispatch boundary.
#[must_use]
pub fn sheaf_laplacian_eigenvalue(
    restriction_diag: &str,
    v: &str,
    lambda: &str,
    n_nodes: u32,
    d: u32,
    iterations: u32,
) -> Program {
    // `iterations` is accepted for interface stability. The dominant eigenpair of a diagonal
    // operator is the closed form below regardless of iteration count (a power iteration converges
    // to it immediately), so it does not influence the emitted program.
    let _ = iterations;
    if n_nodes == 0 || d == 0 {
        return crate::invalid_output_program(
            OP_ID,
            lambda,
            DataType::U32,
            format!(
                "Fix: sheaf_laplacian_eigenvalue requires n_nodes > 0 and d > 0, got n_nodes={n_nodes}, d={d}."
            ),
        );
    }
    let Some(cells) = n_nodes.checked_mul(d) else {
        return crate::invalid_output_program(
            OP_ID,
            lambda,
            DataType::U32,
            format!(
                "Fix: sheaf_laplacian_eigenvalue n_nodes*d overflows vector cell count for n_nodes={n_nodes}, d={d}; shard the sheaf spectrum before GPU dispatch."
            ),
        );
    };

    // Closed-form dominant eigenpair of diag(r): serial single-lane scan for the max diagonal entry
    // and its arg-max index, then write `lambda = max r` and the unit eigenvector `e_argmax`.
    //
    // `eig_max` carries the running max (16.16) and `eig_argmax` the running arg-max index, both are
    // loop-carried locals (`assign`), NOT storage scratch, so nothing internal leaks as a program
    // output. Ties keep the FIRST index (strict `>`), matching the CPU reference. The arg-max is
    // updated BEFORE the max within an iteration so both `>` compares read the same pre-update
    // running max. `one_fp_buf[0]` is 1.0 in 16.16 (the single non-zero eigenvector entry).
    let one_fp = Expr::load("one_fp_buf", Expr::u32(0));
    let nodes = vec![
        Node::let_bind("eig_max", Expr::u32(0)),
        Node::let_bind("eig_argmax", Expr::u32(0)),
        Node::loop_for(
            "eig_scan_i",
            Expr::u32(0),
            Expr::u32(cells),
            vec![
                Node::let_bind(
                    "eig_ri",
                    Expr::load(restriction_diag, Expr::var("eig_scan_i")),
                ),
                Node::assign(
                    "eig_argmax",
                    Expr::select(
                        Expr::gt(Expr::var("eig_ri"), Expr::var("eig_max")),
                        Expr::var("eig_scan_i"),
                        Expr::var("eig_argmax"),
                    ),
                ),
                Node::assign(
                    "eig_max",
                    Expr::select(
                        Expr::gt(Expr::var("eig_ri"), Expr::var("eig_max")),
                        Expr::var("eig_ri"),
                        Expr::var("eig_max"),
                    ),
                ),
            ],
        ),
        Node::store(lambda, Expr::u32(0), Expr::var("eig_max")),
        Node::loop_for(
            "eig_write_j",
            Expr::u32(0),
            Expr::u32(cells),
            vec![Node::store(
                v,
                Expr::var("eig_write_j"),
                Expr::select(
                    Expr::eq(Expr::var("eig_write_j"), Expr::var("eig_argmax")),
                    one_fp.clone(),
                    Expr::u32(0),
                ),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(restriction_diag, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(v, 1, BufferAccess::ReadWrite, DataType::U32).with_count(cells),
            BufferDecl::storage(lambda, 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("one_fp_buf", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::Region {
                generator: Ident::from(POWER_ITERATION_PHASE_OP_ID),
                source_region: Some(GeneratorRef {
                    name: OP_ID.to_string(),
                }),
                // The scan is single-threaded (no per-lane work partitioning). The reference/GPU
                // infers the dispatch grid from buffer shapes, so a count-`cells` vector spawns
                // `cells` invocations. The running-max scratch is a plain (non-atomic) accumulator,
                // so redundant invocations would each recompute it, the last write wins and every
                // lane computes the SAME max, but guarding the whole body to `InvocationId == 0`
                // keeps the answer unambiguously grid-invariant with a single writer (the canonical
                // GPU serial-region idiom (cf. matroid, path_reconstruct)).
                body: Arc::new(vec![Node::if_then(
                    Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    nodes,
                )]),
            }]),
        }],
    )
}

/// CPU reference: dominant eigenpair of the diagonal sheaf Laplacian.
///
/// Returns `(max_i r[i], e_argmax)` over the first `v_init.len()` diagonal entries. `iterations`
/// is accepted for interface stability but the closed-form answer is iteration-independent, and the
/// starting vector `v_init` is used only to size the output eigenvector.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(restriction_diag: &[f64], v_init: &[f64], iterations: u32) -> (f64, Vec<f64>) {
    let mut v = Vec::new();
    let mut v_next = Vec::new();
    let lambda = try_cpu_ref_into(restriction_diag, v_init, iterations, &mut v, &mut v_next)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - sheaf_laplacian_eigenvalue cpu_ref failed: invalid CPU buffers");
    (lambda, v)
}

/// Fallible CPU reference.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(
    restriction_diag: &[f64],
    v_init: &[f64],
    iterations: u32,
) -> Result<(f64, Vec<f64>), String> {
    let mut v = Vec::new();
    let mut v_next = Vec::new();
    let lambda = try_cpu_ref_into(restriction_diag, v_init, iterations, &mut v, &mut v_next)?;
    Ok((lambda, v))
}

/// CPU reference writing the eigenvector into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_into(
    restriction_diag: &[f64],
    v_init: &[f64],
    iterations: u32,
    v: &mut Vec<f64>,
    v_next: &mut Vec<f64>,
) -> f64 {
    try_cpu_ref_into(restriction_diag, v_init, iterations, v, v_next)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - sheaf_laplacian_eigenvalue cpu_ref_into failed: invalid CPU buffers")
}

/// Fallible CPU reference writing the eigenvector into caller-owned storage.
///
/// `v_next` is retained as a caller-owned scratch for interface stability (the closed form needs no
/// intermediate vector); it is truncated to the output length so stale tails cannot leak.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into(
    restriction_diag: &[f64],
    v_init: &[f64],
    iterations: u32,
    v: &mut Vec<f64>,
    v_next: &mut Vec<f64>,
) -> Result<f64, String> {
    let _ = iterations;
    if restriction_diag.len() < v_init.len() {
        return Err(format!(
            "sheaf_laplacian_eigenvalue CPU oracle restriction_diag too short: got {}, need {}.",
            restriction_diag.len(),
            v_init.len()
        ));
    }
    let len = v_init.len();
    reserve_eigen_tmp(v, len, "eigenvector output")?;
    reserve_eigen_tmp(v_next, len, "next-vector scratch")?;
    v.clear();
    v.resize(len, 0.0);
    v_next.clear();
    v_next.resize(len, 0.0);

    // Dominant eigenpair of diag(r): the max diagonal entry and its (first) arg-max index. Running
    // max starts at 0.0, matching the unsigned 16.16 IR path where all diagonal entries are >= 0.
    let mut max_r = 0.0f64;
    let mut argmax = 0usize;
    for (i, &ri) in restriction_diag.iter().take(len).enumerate() {
        if ri > max_r {
            max_r = ri;
            argmax = i;
        }
    }
    if len > 0 {
        v[argmax] = 1.0;
    }
    Ok(max_r)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn reserve_eigen_tmp(out: &mut Vec<f64>, len: usize, name: &str) -> Result<(), String> {
    if len > out.capacity() {
        crate::graph::scratch::reserve_graph_items(
            out,
            len - out.len(),
            "sheaf Laplacian eigenvalue CPU oracle",
            name,
        )?;
    }
    Ok(())
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || sheaf_laplacian_eigenvalue("r", "v", "l", 4, 1, 4),
        Some(|| {
            let to_bytes = |words: &[u32]| crate::wire::pack_u32_slice(words);
            vec![vec![
                to_bytes(&[0; 4]),       // r
                to_bytes(&[0; 4]),       // v
                to_bytes(&[0]),          // l
                to_bytes(&[1u32 << 16]), // one_fp_buf
            ]]
        }),
        Some(|| {
            let to_bytes = |words: &[u32]| crate::wire::pack_u32_slice(words);
            // All-zero diagonal: max is 0 at arg-max index 0, so lambda = 0 and the eigenvector is
            // e_0 = [1.0, 0, 0, 0] in 16.16. The only writable outputs are `v` and `lambda`: the
            // running max/arg-max are loop-carried locals, not storage buffers.
            vec![vec![
                to_bytes(&[1u32 << 16, 0, 0, 0]), // v = e_0
                to_bytes(&[0]),                   // l = max r = 0
            ]]
        }),
    )
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        POWER_ITERATION_PHASE_OP_ID,
        || {
            Program::wrapped(
                vec![
                    BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::Region {
                    generator: Ident::from(POWER_ITERATION_PHASE_OP_ID),
                    source_region: None,
                    body: Arc::new(vec![Node::store(
                        "out",
                        Expr::u32(0),
                        Expr::load("input", Expr::u32(0)),
                    )]),
                }],
            )
        },
        Some(|| {
            let to_bytes = |words: &[u32]| crate::wire::pack_u32_slice(words);
            vec![vec![to_bytes(&[11]), to_bytes(&[0])]]
        }),
        Some(|| {
            let to_bytes = |words: &[u32]| crate::wire::pack_u32_slice(words);
            vec![vec![to_bytes(&[11])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_diagonal_max() {
        let r = vec![1.0, 2.0, 5.0, 3.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let (lambda, vec_final) = cpu_ref(&r, &v, 20);
        // Dominant eigenvalue is the max diagonal entry 5.0; eigenvector is e_2.
        assert_eq!(lambda, 5.0);
        assert_eq!(vec_final, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn cpu_ref_uniform() {
        let r = vec![2.0, 2.0];
        let v = vec![1.0, 0.0];
        let (lambda, vec_final) = cpu_ref(&r, &v, 5);
        assert_eq!(lambda, 2.0);
        // Ties keep the first arg-max index: e_0.
        assert_eq!(vec_final, vec![1.0, 0.0]);
    }

    #[test]
    fn cpu_ref_zero() {
        let r = vec![0.0, 0.0];
        let v = vec![1.0, 1.0];
        let (lambda, vec_final) = cpu_ref(&r, &v, 5);
        assert_eq!(lambda, 0.0);
        // No entry exceeds the 0.0 running max, so the arg-max stays at index 0.
        assert_eq!(vec_final, vec![1.0, 0.0]);
    }

    #[test]
    fn cpu_ref_single() {
        let r = vec![42.0];
        let v = vec![1.0];
        let (lambda, vec_final) = cpu_ref(&r, &v, 1);
        assert_eq!(lambda, 42.0);
        assert_eq!(vec_final, vec![1.0]);
    }

    #[test]
    fn cpu_ref_asymmetric() {
        let r = vec![1.0, 10.0, 0.1];
        let v = vec![1.0, 1.0, 1.0];
        let (lambda, vec_final) = cpu_ref(&r, &v, 10);
        assert_eq!(lambda, 10.0);
        assert_eq!(vec_final, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn cpu_ref_is_iteration_independent() {
        // The closed-form diagonal eigenpair does not depend on the iteration count.
        let r = vec![1.0, 7.0, 3.0, 2.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let (lambda_1, vec_1) = cpu_ref(&r, &v, 1);
        let (lambda_50, vec_50) = cpu_ref(&r, &v, 50);
        assert_eq!(lambda_1, 7.0);
        assert_eq!(lambda_1, lambda_50);
        assert_eq!(vec_1, vec_50);
        assert_eq!(vec_1, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn cpu_ref_into_reuses_vectors_and_truncates_stale_tail() {
        let r = vec![1.0, 2.0, 5.0, 3.0];
        let init = vec![1.0, 1.0, 1.0, 1.0];
        let mut v = Vec::with_capacity(8);
        let mut next = Vec::with_capacity(8);
        v.extend([99.0; 8]);
        next.extend([99.0; 8]);
        let v_ptr = v.as_ptr();
        let next_ptr = next.as_ptr();

        let lambda = try_cpu_ref_into(&r, &init, 20, &mut v, &mut next).unwrap();

        assert_eq!(lambda, 5.0);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(v.len(), init.len());
        assert_eq!(next.len(), init.len());
        assert_eq!(v.as_ptr(), v_ptr);
        assert_eq!(next.as_ptr(), next_ptr);
    }

    #[test]
    fn generated_cpu_ref_matches_independent_arg_max() {
        for case in 0..48 {
            let n = 1 + (case % 8);
            let restriction: Vec<f64> = (0..n)
                .map(|idx| 0.5 + ((idx * 7 + case * 3) % 11) as f64 * 0.25)
                .collect();
            let init: Vec<f64> = vec![1.0; n];
            let iterations = 1 + (case % 8) as u32;
            let mut v = Vec::with_capacity(n + 3);
            let mut next = Vec::with_capacity(n + 3);

            let lambda =
                try_cpu_ref_into(&restriction, &init, iterations, &mut v, &mut next).unwrap();
            let (expected_lambda, expected_arg) = independent_diagonal_dominant(&restriction);

            assert_eq!(lambda, expected_lambda, "case {case}: lambda");
            for (idx, &value) in v.iter().enumerate() {
                let want = if idx == expected_arg { 1.0 } else { 0.0 };
                assert_eq!(value, want, "case {case} idx {idx}: eigenvector entry");
            }
        }
    }

    #[test]
    fn try_cpu_ref_rejects_short_restriction_diag() {
        let err = try_cpu_ref(&[1.0], &[1.0, 2.0], 1).unwrap_err();
        assert!(err.contains("restriction_diag too short"), "{err}");
    }

    #[test]
    fn program_buffer_count() {
        let p = sheaf_laplacian_eigenvalue("r", "v", "l", 4, 1, 4);
        // restriction_diag(RO) + v(RW) + lambda(RW) + one_fp_buf(RO): the running max/arg-max are
        // loop-carried locals, so there are no scratch storage buffers.
        assert_eq!(p.buffers.len(), 4);
        let writable = p
            .buffers
            .iter()
            .filter(|b| b.access() == BufferAccess::ReadWrite)
            .count();
        assert_eq!(writable, 2, "only v and lambda are writable outputs");
    }

    /// Independent dominant-eigenpair oracle for diag(r): the first arg-max of the diagonal, with a
    /// 0.0 running-max floor (matching the unsigned fixed-point path).
    fn independent_diagonal_dominant(restriction_diag: &[f64]) -> (f64, usize) {
        let mut max_r = 0.0f64;
        let mut arg = 0usize;
        for (idx, &value) in restriction_diag.iter().enumerate() {
            if value > max_r {
                max_r = value;
                arg = idx;
            }
        }
        (max_r, arg)
    }
}
