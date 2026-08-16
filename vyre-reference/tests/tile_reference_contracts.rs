//! Contract tests for reference interpreter execution of Tile operations.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Layout, Node, Program, Residency, SubgroupReduceOp,
    Tile,
};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn reference_eval_tile_matmul_2x2() {
    // A = [[1.0, 2.0], [3.0, 4.0]]
    // B = [[5.0, 6.0], [7.0, 8.0]]
    // C = A x B = [[19.0, 22.0], [43.0, 50.0]]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    let tile_a = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );
    let tile_b = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );
    let tile_c = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 2, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_decl("c", tile_c),
            Node::tile_load(
                "t_a",
                tile_a,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_load(
                "t_b",
                tile_b,
                "b",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_matmul("c", "t_a", "t_b"),
            Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "c"),
        ],
    );

    let outputs = reference_eval(
        &prog,
        &[
            Value::from(encode_f32(&a_data)),
            Value::from(encode_f32(&b_data)),
            Value::from(vec![0u8; 16]),
        ],
    )
    .expect("reference_eval failed");

    let out_f32 = decode_f32(&outputs[0].to_bytes());
    assert_eq!(out_f32, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn reference_eval_tile_reduce_and_elementwise() {
    let a_data = vec![1.0f32, 5.0, 2.0, 8.0];
    let tile_a = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 1, DataType::F32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load(
                "t_a",
                tile_a,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_reduce("max_per_row", "t_a", SubgroupReduceOp::Max, 1),
            Node::tile_store("out", vec![Expr::u32(0)], "max_per_row"),
        ],
    );

    let outputs = reference_eval(
        &prog,
        &[Value::from(encode_f32(&a_data)), Value::from(vec![0u8; 8])],
    )
    .expect("reference_eval failed");

    let out_f32 = decode_f32(&outputs[0].to_bytes());
    assert_eq!(out_f32, vec![5.0, 8.0]);
}

#[test]
fn reference_eval_tile_elementwise_scaling() {
    let a_data = vec![2.0f32, 4.0, 6.0, 8.0];
    let tile_a = Tile::new(
        DataType::F32,
        vec![4],
        Layout::RowMajor,
        Residency::Register,
    );

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 1, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load("t_a", tile_a, "a", vec![Expr::u32(0)], Layout::RowMajor),
            Node::tile_elementwise(
                "scaled",
                vec![vyre_foundation::ir::Ident::from("t_a")],
                vec![Node::let_bind(
                    "scaled",
                    Expr::mul(Expr::var("t_a"), Expr::f32(3.0)),
                )],
            ),
            Node::tile_store("out", vec![Expr::u32(0)], "scaled"),
        ],
    );

    let outputs = reference_eval(
        &prog,
        &[Value::from(encode_f32(&a_data)), Value::from(vec![0u8; 16])],
    )
    .expect("reference_eval failed");

    let out_f32 = decode_f32(&outputs[0].to_bytes());
    assert_eq!(out_f32, vec![6.0, 12.0, 18.0, 24.0]);
}

#[test]
fn reference_eval_tile_column_major_layout() {
    // Source 2x2: [[1.0, 2.0], [3.0, 4.0]] in memory: [1.0, 2.0, 3.0, 4.0]
    // Loaded with ColumnMajor into tile extents [2, 2]:
    // (0, 0) -> local linear index (0*1 + 0*2) = 0 => 1.0
    // (0, 1) -> local linear index (0*1 + 1*2) = 2 => 2.0
    // (1, 0) -> local linear index (1*1 + 0*2) = 1 => 3.0
    // (1, 1) -> local linear index (1*1 + 1*2) = 3 => 4.0
    // In tile array elements: [1.0, 3.0, 2.0, 4.0]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tile_col = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::ColumnMajor,
        Residency::Register,
    );

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 1, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load(
                "t_col",
                tile_col,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::ColumnMajor,
            ),
            Node::tile_store("out", vec![Expr::u32(0)], "t_col"),
        ],
    );

    let outputs = reference_eval(
        &prog,
        &[Value::from(encode_f32(&a_data)), Value::from(vec![0u8; 16])],
    )
    .expect("reference_eval failed");

    let out_f32 = decode_f32(&outputs[0].to_bytes());
    assert_eq!(out_f32, vec![1.0, 3.0, 2.0, 4.0]);
}
