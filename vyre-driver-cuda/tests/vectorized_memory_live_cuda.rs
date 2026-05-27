//! Live CUDA/reference coverage for PTX vectorized memory chains.

mod common;

use common::{assert_u32_output_lanes, cuda_reference_outputs, live_backend, u32_bytes};
use vyre::DispatchConfig;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

const VECTOR_LANE_COUNT: usize = 2048;
const VECTOR_GROUP_COUNT: usize = VECTOR_LANE_COUNT / 4;
const WORKGROUP_SIZE_X: u32 = 128;

#[test]
fn vectorized_u32_copy_emits_packed_ptx_and_matches_reference_on_live_cuda() {
    let program = vectorized_copy_program();
    let ptx = vyre_driver_cuda::codegen::program_to_ptx(&program, &DispatchConfig::default())
        .expect("Fix: CUDA PTX emission must support unit-stride vectorized memory programs.");
    assert!(
        ptx.contains("ld.global.v4.u32") || ptx.contains("ld.global.nc.v4.u32"),
        "Fix: CUDA release PTX must fuse four adjacent u32 loads into a packed v4 global load.\n{ptx}"
    );
    assert!(
        ptx.contains("st.global.v4.u32"),
        "Fix: CUDA release PTX must fuse four adjacent u32 stores into st.global.v4.u32.\n{ptx}"
    );

    let backend = live_backend();
    let input = generated_u32_values();
    let outputs = cuda_reference_outputs(
        &backend,
        &program,
        &[u32_bytes(&input)],
        "vectorized_u32_copy",
    );
    let checked = assert_u32_output_lanes(
        "vectorized_u32_copy direct",
        VECTOR_LANE_COUNT,
        &outputs.direct_cuda,
        &outputs.reference,
    ) + assert_u32_output_lanes(
        "vectorized_u32_copy compiled",
        VECTOR_LANE_COUNT,
        &outputs.compiled_cuda,
        &outputs.reference,
    );

    assert_eq!(
        checked,
        VECTOR_LANE_COUNT * 2,
        "Fix: live CUDA vectorized memory test must compare every lane on direct and compiled paths."
    );
}

fn vectorized_copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(VECTOR_LANE_COUNT as u32),
            BufferDecl::output("out", 1, DataType::U32).with_count(VECTOR_LANE_COUNT as u32),
        ],
        [WORKGROUP_SIZE_X, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(VECTOR_GROUP_COUNT as u32)),
            vec![
                    Node::let_bind("base", Expr::mul(Expr::gid_x(), Expr::u32(4))),
                    Node::let_bind("i0", Expr::var("base")),
                    Node::let_bind("i1", Expr::add(Expr::var("base"), Expr::u32(1))),
                    Node::let_bind("i2", Expr::add(Expr::var("base"), Expr::u32(2))),
                    Node::let_bind("i3", Expr::add(Expr::var("base"), Expr::u32(3))),
                    Node::let_bind("v0", Expr::load("input", Expr::var("i0"))),
                    Node::let_bind("v1", Expr::load("input", Expr::var("i1"))),
                    Node::let_bind("v2", Expr::load("input", Expr::var("i2"))),
                    Node::let_bind("v3", Expr::load("input", Expr::var("i3"))),
                    Node::store("out", Expr::var("i0"), Expr::var("v0")),
                    Node::store("out", Expr::var("i1"), Expr::var("v1")),
                    Node::store("out", Expr::var("i2"), Expr::var("v2")),
                    Node::store("out", Expr::var("i3"), Expr::var("v3")),
            ],
        )],
    )
}

fn generated_u32_values() -> Vec<u32> {
    (0..VECTOR_LANE_COUNT)
        .map(|lane| {
            let lane = lane as u32;
            lane.wrapping_mul(0x9e37_79b9)
                .rotate_left((lane & 31) + 1)
                ^ 0xa5a5_5a5a_u32.rotate_right(lane & 31)
        })
        .collect()
}
