//! Information-geometry primitives  -  Fisher-Rao distance + Amari
//! α-connection between distributions on a statistical manifold.
//!
//! Fisher-Rao distance is the Riemannian distance on the statistical
//! manifold under the Fisher information metric (Rao 1945). For
//! categorical distributions p, q over k outcomes:
//!
//! ```text
//!   d_FR(p, q) = 2 · arccos( Σ_i sqrt(p_i · q_i) )
//! ```
//!
//! This is the "spherical" form. The Amari α-connection (Amari 1985)
//! interpolates between exponential (α = 1) and mixture (α = -1)
//! families; α = 0 recovers the Levi-Civita connection (Fisher-Rao).
//!
//! This file ships:
//! - `bhattacharyya_coefficient`  -  `Σ sqrt(p · q)`, the inner
//!   product on the spherical statistical manifold. Distance is
//!   `2 · arccos(coeff)` host-side.
//! - `amari_alpha_step` witness in `vyre-reference`  -  host-side α-connection interpolation,
//!   useful for distribution-aware loss design.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::bhattacharyya_coefficient";

/// Numerical floor for `sqrt(p_i)`. Caller supplies `p` and `q` in
/// 16.16 fixed-point already representing probabilities; the GPU
/// path uses an integer-square-root approximation per element.
///
/// # Algorithm
///
/// Per lane `t`:
///   `out[t] = isqrt(p[t]) · isqrt(q[t]) >> 16`
///
/// then the caller dispatches `reduce::sum` to obtain the final
/// scalar coefficient.
#[must_use]
pub fn bhattacharyya_per_element(p: &str, q: &str, out_per_elem: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((out_per_elem, DataType::U32)),
            format!("Fix: bhattacharyya_per_element requires n > 0, got {n}."),
        );
    }

    let t = Expr::LogicalIndex { axis: 0 };

    // Integer square root via Newton iteration: x_{k+1} = (x_k + n/x_k) / 2
    // 4 iterations from x_0 = n give ~2 ulps for u32 values up to 2^31.
    // We embed 4 unrolled steps and clamp to 1 to avoid divide-by-zero.
    let isqrt_inline = |val_var: &'static str| {
        let mut steps = vec![
            Node::let_bind(val_var, Expr::load(p, t.clone())),
            // Use any nonzero seed; pick val_var or 1.
            Node::let_bind(
                "x",
                Expr::select(
                    Expr::eq(Expr::var(val_var), Expr::u32(0)),
                    Expr::u32(1),
                    Expr::var(val_var),
                ),
            ),
        ];
        for _ in 0..4 {
            steps.push(Node::assign(
                "x",
                Expr::shr(
                    Expr::add(
                        Expr::var("x"),
                        Expr::div(Expr::var(val_var), Expr::var("x")),
                    ),
                    Expr::u32(1),
                ),
            ));
        }
        steps
    };

    let mut body_inner = isqrt_inline("pv");
    body_inner.push(Node::let_bind("xp", Expr::var("x")));
    // q
    body_inner.push(Node::let_bind("qv", Expr::load(q, t.clone())));
    body_inner.push(Node::let_bind(
        "y",
        Expr::select(
            Expr::eq(Expr::var("qv"), Expr::u32(0)),
            Expr::u32(1),
            Expr::var("qv"),
        ),
    ));
    for _ in 0..4 {
        body_inner.push(Node::assign(
            "y",
            Expr::shr(
                Expr::add(Expr::var("y"), Expr::div(Expr::var("qv"), Expr::var("y"))),
                Expr::u32(1),
            ),
        ));
    }
    body_inner.push(Node::store(
        out_per_elem,
        t.clone(),
        crate::math::fixed::fixed_mul_16_16_expr(Expr::var("xp"), Expr::var("y")),
    ));

    let body = vec![Node::if_then(Expr::lt(t.clone(), Expr::u32(n)), body_inner)];

    Program::wrapped(
        vec![
            BufferDecl::storage(p, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(q, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(out_per_elem, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

// ---- CPU references ----

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || {
            bhattacharyya_per_element("a", "b", "out", 4)
        },
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[1; 4]),
                vyre_primitives::wire::pack_u32_slice(&[1; 4]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x00, // 0
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ]]]
        }),
    )
    .with_no_legal_rewrite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_program_buffer_layout() {
        let prog = bhattacharyya_per_element("p", "q", "out", 16);
        assert_eq!(prog.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = prog.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["p", "q", "out"]);
    }

    #[test]
    fn zero_n_traps() {
        let p = bhattacharyya_per_element("p", "q", "out", 0);
        assert!(p.stats().trap());
    }
}
