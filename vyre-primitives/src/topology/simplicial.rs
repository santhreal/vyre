//! Simplicial neural network message-passing primitive (#32).
//!
//! Simplicial NNs (Bodnar-Frasca 2021, Yang-Sala 2023) generalize GNNs
//! from edges to higher-order simplices (triangles, tetrahedra). The
//! boundary operator ∂ is the substrate: for a triangle (i, j, k),
//! `∂ = (j, k) - (i, k) + (i, j)`  -  alternating-sign sum of faces.
//!
//! This file ships the **2-simplex (triangle) message aggregation
//! step** primitive  -  given an edge-feature buffer and a list of
//! triangles, compute per-triangle messages by summing alternating-
//! sign face features. Composes with `level_wave_program` for a
//! full simplicial-complex pass.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | `vyre-libs::ml::scnn` consumers | hypergraph + mesh learning |
//! | `vyre-libs::sci::topology_features` consumers | topological-feature ML |
//! | `vyre-foundation::transform` conflict analysis | 3-way Region conflicts in vyre's dispatch graph become 2-simplices; same primitive scores them |

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::topology::simplicial_triangle_message";

/// Emit the per-triangle message Program.
///
/// Inputs:
/// - `edge_features`: `n_edges * d` u32 (16.16)  -  per-edge `d`-dim
///   feature vector.
/// - `triangle_edges`: `n_triangles * 3` u32  -  for each triangle, the
///   three edge indices `(e_jk, e_ik, e_ij)` in canonical order.
///
/// Output:
/// - `triangle_messages`: `n_triangles * d` u32  -  per-triangle message
///   computed as `∂(triangle) = e_jk - e_ik + e_ij`.
#[must_use]
pub fn simplicial_triangle_message(
    edge_features: &str,
    triangle_edges: &str,
    triangle_messages: &str,
    n_edges: u32,
    n_triangles: u32,
    d: u32,
) -> Program {
    if n_edges == 0 {
        return crate::invalid_output_program(
            OP_ID,
            triangle_messages,
            DataType::U32,
            "Fix: simplicial_triangle_message requires n_edges > 0, got 0.".to_string(),
        );
    }
    if n_triangles == 0 {
        return crate::invalid_output_program(
            OP_ID,
            triangle_messages,
            DataType::U32,
            "Fix: simplicial_triangle_message requires n_triangles > 0, got 0.".to_string(),
        );
    }
    if d == 0 {
        return crate::invalid_output_program(
            OP_ID,
            triangle_messages,
            DataType::U32,
            "Fix: simplicial_triangle_message requires d > 0, got 0.".to_string(),
        );
    }

    let cells = n_triangles * d;
    let t = Expr::InvocationId { axis: 0 };
    let tri = Expr::div(t.clone(), Expr::u32(d));
    let dim = Expr::rem(t.clone(), Expr::u32(d));

    // edge indices: triangle_edges[tri * 3 + 0..2]
    let tri_base = Expr::mul(tri, Expr::u32(3));
    let e_jk = Expr::load(triangle_edges, tri_base.clone());
    let e_ik = Expr::load(triangle_edges, Expr::add(tri_base.clone(), Expr::u32(1)));
    let e_ij = Expr::load(triangle_edges, Expr::add(tri_base, Expr::u32(2)));

    let load_edge_feat = |e: Expr| {
        Expr::load(
            edge_features,
            Expr::add(Expr::mul(e, Expr::u32(d)), dim.clone()),
        )
    };

    let value = Expr::add(
        Expr::sub(
            load_edge_feat(Expr::var("e_jk")),
            load_edge_feat(Expr::var("e_ik")),
        ),
        load_edge_feat(Expr::var("e_ij")),
    );

    // Skip a triangle that references an out-of-range edge, matching the CPU
    // reference `simplicial_triangle_message_cpu` (`if e_* >= n_edges { continue }`,
    // leaving that triangle's message at zero). `triangle_edges` is unvalidated
    // input; without this gate the GPU gathers edge_features[e*d+dim] OOB (UB on
    // CUDA) and writes a garbage message, diverging from the CPU which leaves it 0.
    // Bind each edge index once (avoids re-loading it three times) and only store
    // when all three are in range. Transparent to well-formed meshes.
    let edges_in_range = Expr::and(
        Expr::lt(Expr::var("e_jk"), Expr::u32(n_edges)),
        Expr::and(
            Expr::lt(Expr::var("e_ik"), Expr::u32(n_edges)),
            Expr::lt(Expr::var("e_ij"), Expr::u32(n_edges)),
        ),
    );

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(cells)),
        vec![
            Node::let_bind("e_jk", e_jk),
            Node::let_bind("e_ik", e_ik),
            Node::let_bind("e_ij", e_ij),
            Node::if_then(
                edges_in_range,
                vec![Node::store(triangle_messages, t, value)],
            ),
        ],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(edge_features, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_edges * d),
            BufferDecl::storage(triangle_edges, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n_triangles * 3),
            BufferDecl::storage(triangle_messages, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
        ],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

/// CPU reference: `triangle_messages = ∂(edge_features)` per triangle.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn simplicial_triangle_message_cpu(
    edge_features: &[f64],
    triangle_edges: &[u32],
    n_edges: u32,
    n_triangles: u32,
    d: u32,
) -> Vec<f64> {
    let n_edges = n_edges as usize;
    let n_triangles = n_triangles as usize;
    let d = d as usize;

    let mut out = vec![0.0; n_triangles * d];
    for tri in 0..n_triangles {
        let Some(&e_jk) = triangle_edges.get(tri * 3) else {
            continue;
        };
        let Some(&e_ik) = triangle_edges.get(tri * 3 + 1) else {
            continue;
        };
        let Some(&e_ij) = triangle_edges.get(tri * 3 + 2) else {
            continue;
        };
        let e_jk = e_jk as usize;
        let e_ik = e_ik as usize;
        let e_ij = e_ij as usize;
        if e_jk >= n_edges || e_ik >= n_edges || e_ij >= n_edges {
            continue;
        }
        for k in 0..d {
            let Some(&jk) = edge_features.get(e_jk * d + k) else {
                continue;
            };
            let Some(&ik) = edge_features.get(e_ik * d + k) else {
                continue;
            };
            let Some(&ij) = edge_features.get(e_ij * d + k) else {
                continue;
            };
            out[tri * d + k] = jk - ik + ij;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_zero_features_zero_messages() {
        let edges = vec![0.0; 9]; // 3 edges, d=3
        let tris = vec![0u32, 1, 2];
        let out = simplicial_triangle_message_cpu(&edges, &tris, 3, 1, 3);
        for v in out {
            assert!(approx_eq(v, 0.0));
        }
    }

    #[test]
    fn cpu_alternating_sign_decoded() {
        // e_jk = 5, e_ik = 3, e_ij = 1 → message = 5 - 3 + 1 = 3
        let edges = vec![1.0, 3.0, 5.0]; // edge 0 = 1, edge 1 = 3, edge 2 = 5
        let tris = vec![2u32, 1, 0]; // (e_jk=2, e_ik=1, e_ij=0)
        let out = simplicial_triangle_message_cpu(&edges, &tris, 3, 1, 1);
        assert!(approx_eq(out[0], 3.0));
    }

    #[test]
    fn cpu_two_triangles_independent() {
        // 4 edges, 2 triangles
        let edges = vec![10.0, 20.0, 30.0, 40.0];
        let tris = vec![0u32, 1, 2, 1, 2, 3];
        // tri 0: e_jk=0 (10), e_ik=1 (20), e_ij=2 (30) → 10-20+30 = 20
        // tri 1: e_jk=1 (20), e_ik=2 (30), e_ij=3 (40) → 20-30+40 = 30
        let out = simplicial_triangle_message_cpu(&edges, &tris, 4, 2, 1);
        assert!(approx_eq(out[0], 20.0));
        assert!(approx_eq(out[1], 30.0));
    }

    #[test]
    fn cpu_d_dim_features_propagate_independently() {
        // 2-D features per edge, 1 triangle.
        let edges = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let tris = vec![2u32, 1, 0];
        // dim 0: 3 - 2 + 1 = 2
        // dim 1: 30 - 20 + 10 = 20
        let out = simplicial_triangle_message_cpu(&edges, &tris, 3, 1, 2);
        assert!(approx_eq(out[0], 2.0));
        assert!(approx_eq(out[1], 20.0));
    }

    #[test]
    fn cpu_malformed_triangle_inputs_leave_zero_messages() {
        let short_edges = vec![1.0];
        let short_tris = vec![0u32, 1];
        let out = simplicial_triangle_message_cpu(&short_edges, &short_tris, 3, 2, 1);
        assert_eq!(out, vec![0.0, 0.0]);

        let bad_edge = simplicial_triangle_message_cpu(&[1.0, 2.0], &[0, 9, 1], 2, 1, 1);
        assert_eq!(bad_edge, vec![0.0]);
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = simplicial_triangle_message("e", "te", "tm", 8, 3, 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        assert_eq!(p.buffers[0].count(), 32);
        assert_eq!(p.buffers[1].count(), 9);
        assert_eq!(p.buffers[2].count(), 12);
    }

    #[test]
    fn zero_edges_traps() {
        let p = simplicial_triangle_message("e", "te", "tm", 0, 1, 1);
        assert!(p.stats().trap());
    }

    #[test]
    fn ir_message_skips_malformed_triangle_matching_cpu_zero() {
        use vyre_reference::value::Value;
        // The GPU IR had NO `e < n_edges` check while the CPU reference skips
        // triangles that reference an out-of-range edge (leaving the message 0,
        // locked by `cpu_malformed_triangle_inputs_leave_zero_messages`). This is a
        // GPU/CPU parity regression LOCK.
        let n_edges = 3u32;
        let n_triangles = 2u32;
        let d = 1u32;
        // OVER-LENGTH edge_features: a 4th element (999) past the declared
        // n_edges*d = 3, so the pre-fix OOB gather reads a NONZERO value the
        // interpreter would store. A zero-init buffer alone cannot distinguish the
        // fix because an OOB read zero-fills to 0 (→ value 0 → stored 0 either way).
        let edge_features = [10u32, 20, 30, 999];
        // tri 0 valid (edges 0,1,2); tri 1 malformed (e_jk = 3 == n_edges, OOR).
        let triangle_edges = [0u32, 1, 2, 3, 0, 0];

        let program = simplicial_triangle_message("e", "te", "tm", n_edges, n_triangles, d);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(crate::wire::pack_u32_slice(&edge_features)),
                Value::from(crate::wire::pack_u32_slice(&triangle_edges)),
                Value::from(crate::wire::pack_u32_slice(&vec![
                    0u32;
                    (n_triangles * d) as usize
                ])),
            ],
        )
        .expect("Fix: simplicial_triangle_message must reference-evaluate");
        let tm_idx = vyre_reference::output_index(&program, "tm")
            .expect("Fix: triangle_messages must be a reference output");
        let messages = crate::wire::decode_u32_le_bytes_all(&outputs[tm_idx].to_bytes());

        // tri 0 (valid): 10 - 20 + 30 = 20 (u32 wrapping). tri 1 (malformed): must be
        // 0 (skipped like the CPU), NOT the 999 the pre-fix OOB read would store.
        assert_eq!(
            messages,
            vec![20u32, 0u32],
            "Fix: a malformed triangle (edge index >= n_edges) must be skipped (message 0) like the CPU reference, not OOB-gathered into a garbage message"
        );
    }
}
