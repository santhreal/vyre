//! Independent known-answer tests for composition witnesses.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::composition_witness::{csr_bfs_witness, prefix_scan_witness, semiring_gemm_witness};
use vyre_reference::{reference_eval, value::Value};
use vyre_spec::Semiring;

#[test]
fn prefix_scan_witness_known_answers() {
    let input = vec![1, 2, 3, 4, 5];

    // Inclusive sum: [1, 3, 6, 10, 15]
    let inc = prefix_scan_witness(&input, true, |a, b| a + b, 0);
    assert_eq!(inc, vec![1, 3, 6, 10, 15]);

    // Exclusive sum: [0, 1, 3, 6, 10]
    let exc = prefix_scan_witness(&input, false, |a, b| a + b, 0);
    assert_eq!(exc, vec![0, 1, 3, 6, 10]);

    // Prefix max: [3, 3, 5, 5, 8]
    let input2 = vec![3, 1, 5, 2, 8];
    let pmax = prefix_scan_witness(&input2, true, u32::max, 0);
    assert_eq!(pmax, vec![3, 3, 5, 5, 8]);
}

#[test]
fn semiring_gemm_witness_known_answers() {
    // 2x2 identity matrix times 2x2 matrix under Real
    let eye = vec![1, 0, 0, 1];
    let m = vec![3, 7, 2, 5];
    let prod = semiring_gemm_witness(&eye, &m, 2, 2, 2, Semiring::Real);
    assert_eq!(prod, vec![3, 7, 2, 5]);

    // Boolean Or-And matrix multiplication (transitive reachability step)
    let a = vec![1, 1, 0, 1];
    let b = vec![0, 1, 1, 0];
    let bool_prod = semiring_gemm_witness(&a, &b, 2, 2, 2, Semiring::BoolOr);
    // [ (1&0)|(1&1)=1, (1&1)|(1&0)=1 ]
    // [ (0&0)|(1&1)=1, (0&1)|(1&0)=0 ]
    assert_eq!(bool_prod, vec![1, 1, 1, 0]);

    // MinPlus (Tropical / Shortest path step)
    let dist_a = vec![0, 3, 7, 0];
    let dist_b = vec![0, 5, 2, 0];
    let min_plus = semiring_gemm_witness(&dist_a, &dist_b, 2, 2, 2, Semiring::MinPlus);
    // c[0][0] = min(0+0, 3+2) = 0
    // c[0][1] = min(0+5, 3+0) = 3
    // c[1][0] = min(7+0, 0+2) = 2
    // c[1][1] = min(7+5, 0+0) = 0
    assert_eq!(min_plus, vec![0, 3, 2, 0]);
}

#[test]
fn csr_bfs_witness_known_graph_topologies() {
    // Triangle graph: 0 -> 1 -> 2 -> 0
    let row_offsets = vec![0, 1, 2, 3];
    let col_indices = vec![1, 2, 0];

    let dists = csr_bfs_witness(3, &row_offsets, &col_indices, 0);
    assert_eq!(dists, vec![0, 1, 2]);

    // Disconnected graph: node 0 -> 1, node 2 isolated
    let row_offsets_disc = vec![0, 1, 1, 1];
    let col_indices_disc = vec![1];
    let dists_disc = csr_bfs_witness(3, &row_offsets_disc, &col_indices_disc, 0);
    assert_eq!(dists_disc, vec![0, 1, u32::MAX]);
}

#[test]
fn interpreter_matches_independent_witness_on_matrix_vector() {
    // Program computing a 2x2 matrix-vector product
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("mat", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("vec", 1, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::output("out", 2, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            // out[0] = mat[0]*vec[0] + mat[1]*vec[1]
            Node::store(
                "out",
                Expr::u32(0),
                Expr::add(
                    Expr::mul(Expr::load("mat", Expr::u32(0)), Expr::load("vec", Expr::u32(0))),
                    Expr::mul(Expr::load("mat", Expr::u32(1)), Expr::load("vec", Expr::u32(1))),
                ),
            ),
            // out[1] = mat[2]*vec[0] + mat[3]*vec[1]
            Node::store(
                "out",
                Expr::u32(1),
                Expr::add(
                    Expr::mul(Expr::load("mat", Expr::u32(2)), Expr::load("vec", Expr::u32(0))),
                    Expr::mul(Expr::load("mat", Expr::u32(3)), Expr::load("vec", Expr::u32(1))),
                ),
            ),
        ],
    );

    let mat = vec![2u32, 3, 4, 5];
    let v = vec![10u32, 20];

    let outputs = reference_eval(
        &program,
        &[
            Value::Bytes(bytemuck_slice(&mat)),
            Value::Bytes(bytemuck_slice(&v)),
        ],
    )
    .expect("reference evaluation must succeed");

    // Independent witness calculation:
    let witness = semiring_gemm_witness(&mat, &v, 2, 1, 2, Semiring::Real);
    // [ 2*10 + 3*20 = 80, 4*10 + 5*20 = 140 ]
    assert_eq!(witness, vec![80, 140]);

    let out_bytes = outputs[0].to_bytes();
    let mut out_u32s = vec![0u32; 2];
    out_u32s[0] = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap());
    out_u32s[1] = u32::from_le_bytes(out_bytes[4..8].try_into().unwrap());

    assert_eq!(out_u32s, witness);
}

fn bytemuck_slice(u32s: &[u32]) -> std::sync::Arc<[u8]> {
    let mut bytes = Vec::with_capacity(u32s.len() * 4);
    for &x in u32s {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.into()
}
