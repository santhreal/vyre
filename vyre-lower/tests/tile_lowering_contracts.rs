//! Contract tests proving tile program lowerings match reference execution.
//!
//! Every tile operation lowers to neutral kernel descriptor ops whose simulated
//! evaluation reproduces the reference oracle index for index.

use std::collections::HashMap;
use vyre_foundation::ir::stats::{
    NODE_KIND_TILE_DECL, NODE_KIND_TILE_ELEMENTWISE, NODE_KIND_TILE_LOAD, NODE_KIND_TILE_MATMUL,
    NODE_KIND_TILE_REDUCE, NODE_KIND_TILE_STORE,
};
use vyre_foundation::ir::{
    BinOp, BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency,
    SubgroupReduceOp, Tile, UnOp,
};
use vyre_lower::{lower, verify, KernelBody, KernelDescriptor, KernelOpKind, LiteralValue};

fn simulate_descriptor(
    desc: &KernelDescriptor,
    input_buffers: &[Vec<f32>],
    output_len: usize,
) -> Vec<f32> {
    let mut buffers: HashMap<u32, Vec<f32>> = HashMap::new();
    for slot_info in &desc.bindings.slots {
        if let Some(buf) = input_buffers.get(slot_info.slot as usize) {
            buffers.insert(slot_info.slot, buf.clone());
        } else {
            let count = slot_info.element_count.unwrap_or(output_len as u32) as usize;
            buffers.insert(slot_info.slot, vec![0.0f32; count.max(output_len)]);
        }
    }

    let mut values: HashMap<u32, f32> = HashMap::new();

    fn exec_body(
        body: &KernelBody,
        buffers: &mut HashMap<u32, Vec<f32>>,
        values: &mut HashMap<u32, f32>,
    ) {
        for op in &body.ops {
            match &op.kind {
                KernelOpKind::Region { .. } | KernelOpKind::StructuredBlock => {
                    let child_idx = op.operands[0] as usize;
                    if let Some(child_body) = body.child_bodies.get(child_idx) {
                        exec_body(child_body, buffers, values);
                    }
                }
                KernelOpKind::Literal => {
                    let lit_idx = op.operands[0] as usize;
                    let val = match body.literals[lit_idx] {
                        LiteralValue::F32(v) => v,
                        LiteralValue::U32(v) => v as f32,
                        LiteralValue::I32(v) => v as f32,
                        LiteralValue::Bool(v) => {
                            if v {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    if let Some(res) = op.result {
                        values.insert(res, val);
                    }
                }
                KernelOpKind::LoadGlobal
                | KernelOpKind::LoadShared
                | KernelOpKind::LoadConstant => {
                    let slot = op.operands[0];
                    let idx_id = op.operands[1];
                    let idx = values.get(&idx_id).copied().unwrap_or(0.0) as usize;
                    let val = buffers
                        .get(&slot)
                        .and_then(|b| b.get(idx))
                        .copied()
                        .unwrap_or(0.0);
                    if let Some(res) = op.result {
                        values.insert(res, val);
                    }
                }
                KernelOpKind::StoreGlobal | KernelOpKind::StoreShared => {
                    let slot = op.operands[0];
                    let idx_id = op.operands[1];
                    let val_id = op.operands[2];
                    let idx = values.get(&idx_id).copied().unwrap_or(0.0) as usize;
                    let val = values.get(&val_id).copied().unwrap_or(0.0);
                    if let Some(buf) = buffers.get_mut(&slot) {
                        if idx < buf.len() {
                            buf[idx] = val;
                        }
                    }
                }
                KernelOpKind::BinOpKind(bin_op) => {
                    let left = values.get(&op.operands[0]).copied().unwrap_or(0.0);
                    let right = values.get(&op.operands[1]).copied().unwrap_or(0.0);
                    let res = match bin_op {
                        BinOp::Add => left + right,
                        BinOp::Sub => left - right,
                        BinOp::Mul => left * right,
                        BinOp::Div => left / right,
                        BinOp::Min => left.min(right),
                        BinOp::Max => left.max(right),
                        BinOp::BitAnd => ((left as u32) & (right as u32)) as f32,
                        BinOp::BitOr => ((left as u32) | (right as u32)) as f32,
                        BinOp::BitXor => ((left as u32) ^ (right as u32)) as f32,
                        _ => left + right,
                    };
                    if let Some(res_id) = op.result {
                        values.insert(res_id, res);
                    }
                }
                KernelOpKind::UnOpKind(un_op) => {
                    let operand = values.get(&op.operands[0]).copied().unwrap_or(0.0);
                    let res = match un_op {
                        UnOp::Exp => operand.exp(),
                        UnOp::Negate => -operand,
                        UnOp::Sqrt => operand.sqrt(),
                        _ => operand,
                    };
                    if let Some(res_id) = op.result {
                        values.insert(res_id, res);
                    }
                }
                KernelOpKind::Copy => {
                    let operand = values.get(&op.operands[0]).copied().unwrap_or(0.0);
                    if let Some(res_id) = op.result {
                        values.insert(res_id, operand);
                    }
                }
                _ => {}
            }
        }
    }

    exec_body(&desc.body, &mut buffers, &mut values);

    let out_slot = desc
        .bindings
        .slots
        .iter()
        .find(|s| s.name == "out")
        .map(|s| s.slot)
        .unwrap_or(desc.bindings.slots.last().map(|s| s.slot).unwrap_or(0));

    buffers
        .get(&out_slot)
        .map(|b| b[..output_len.min(b.len())].to_vec())
        .unwrap_or_default()
}

fn case_load_store_roundtrip() -> (Program, Vec<Vec<f32>>, Vec<f32>) {
    let tile = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 1, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load(
                "t",
                tile,
                "in",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "t"),
        ],
    );
    let inputs = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
    let expected = vec![1.0f32, 2.0, 3.0, 4.0];
    (prog, inputs, expected)
}

fn case_matmul_2x2() -> (Program, Vec<Vec<f32>>, Vec<f32>) {
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
    let tile_acc = Tile::new(
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
            Node::tile_decl("c", tile_acc),
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
    let inputs = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![5.0f32, 6.0, 7.0, 8.0],
    ];
    let expected = vec![19.0f32, 22.0, 43.0, 50.0];
    (prog, inputs, expected)
}

fn case_reduce_axis_1() -> (Program, Vec<Vec<f32>>, Vec<f32>) {
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
    let inputs = vec![vec![1.0f32, 5.0, 2.0, 8.0]];
    let expected = vec![5.0f32, 8.0];
    (prog, inputs, expected)
}

fn case_broadcast_elementwise() -> (Program, Vec<Vec<f32>>, Vec<f32>) {
    let tile_a = Tile::new(
        DataType::F32,
        vec![2, 2],
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
            Node::tile_load(
                "t_a",
                tile_a,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_reduce("row_max", "t_a", SubgroupReduceOp::Max, 1),
            Node::tile_elementwise(
                "diff",
                vec![Ident::from("t_a"), Ident::from("row_max")],
                vec![Node::let_bind(
                    "diff",
                    Expr::sub(Expr::var("t_a"), Expr::var("row_max")),
                )],
            ),
            Node::tile_store("out", vec![Expr::u32(0)], "diff"),
        ],
    );
    let inputs = vec![vec![10.0f32, 20.0, 30.0, 40.0]];
    let expected = vec![-10.0f32, 0.0, -10.0, 0.0];
    (prog, inputs, expected)
}

#[test]
fn tile_lowering_suite_covers_all_tile_node_kinds_and_matches_reference() {
    let cases = [
        ("load_store_roundtrip", case_load_store_roundtrip()),
        ("matmul_2x2", case_matmul_2x2()),
        ("reduce_axis_1", case_reduce_axis_1()),
        ("broadcast_elementwise", case_broadcast_elementwise()),
    ];

    let mut covered_kinds = 0u32;

    for (name, (prog, inputs, expected)) in cases {
        let stats = prog.stats();
        covered_kinds |= stats.node_kinds_present;

        let desc = lower(&prog).unwrap_or_else(|e| panic!("case {name} lowering failed: {e}"));
        verify(&desc).unwrap_or_else(|e| panic!("case {name} verify failed: {e:?}"));

        let actual = simulate_descriptor(&desc, &inputs, expected.len());
        assert_eq!(
            actual, expected,
            "case {name} simulated output did not match expected"
        );
    }

    let all_tile_kinds = NODE_KIND_TILE_DECL
        | NODE_KIND_TILE_LOAD
        | NODE_KIND_TILE_STORE
        | NODE_KIND_TILE_MATMUL
        | NODE_KIND_TILE_REDUCE
        | NODE_KIND_TILE_ELEMENTWISE;

    assert_eq!(
        covered_kinds & all_tile_kinds,
        all_tile_kinds,
        "Fix: every tile node kind must be covered in lowering contract suite"
    );
}
