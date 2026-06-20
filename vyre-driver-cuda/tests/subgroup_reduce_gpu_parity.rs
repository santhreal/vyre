//! Live-GPU parity for `Expr::subgroup_reduce` on the CUDA/PTX backend.
//!
//! The IR/WGSL/oracle contract for a subgroup reduction is *all-lane broadcast*:
//! every active lane receives the full reduction (matching `naga`'s
//! `subgroupAdd`, PTX `redux.sync` on integers, and the reference interpreter).
//! These tests store each lane's reduction result into a distinct output slot
//! and assert *every* slot holds the full reduction.
//!
//! BLOCKED ON AN NVIDIA DRIVER JIT BUG (not a vyre bug) — see
//! `docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md`. On the Blackwell driver JIT
//! (570.211.01, sm_120 `cuModuleLoadData`) BOTH the f32 `shfl.sync` butterfly
//! AND the integer `redux.sync` reduction of a globally-loaded value are
//! silently miscompiled — every lane gets its own value instead of the full
//! reduction. The SAME instructions compiled ahead-of-time by `ptxas` 12.8
//! (and the equivalent `__shfl_sync` / `__reduce_add_sync` built by
//! `nvcc -arch=sm_120`) are CORRECT — proven by the probes in the bug doc.
//! vyre emits correct PTX; the fix is AOT cubin compilation (also a JIT-cost
//! win). All four reduce tests here are `#[ignore]`d until that lands; the
//! PTX-emit unit tests (`vyre-emit-ptx`) and the reference oracle already prove
//! the all-lane semantics on the paths vyre controls.

#![cfg(test)]

mod common;

use common::with_live_backend;
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre::DispatchConfig;

const LANES: u32 = 32;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// `out[gid] = subgroup_reduce(op, in[gid])` over a single 32-lane subgroup.
fn subgroup_reduce_program(reduce: fn(Expr) -> Expr) -> Program {
    let body = vec![
        Node::let_bind("gid", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("gid"), Expr::buf_len("out")),
            vec![Node::store(
                "out",
                Expr::var("gid"),
                reduce(Expr::load("in", Expr::var("gid"))),
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::read_write("in", 0, DataType::F32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::F32).with_count(LANES),
        ],
        [LANES, 1, 1],
        vec![Node::Region {
            generator: "test::subgroup_reduce_parity".into(),
            source_region: None,
            body: std::sync::Arc::new(body),
        }],
    )
}

fn run_subgroup_reduce(reduce: fn(Expr) -> Expr, input: &[f32]) -> Vec<f32> {
    assert_eq!(input.len(), LANES as usize);
    let program = subgroup_reduce_program(reduce);
    // `out` is an output buffer the dispatch allocates and reads back, so only
    // the `in` buffer is provided as a dispatch input.
    let inputs = vec![f32_bytes(input)];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = with_live_backend("subgroup reduce parity", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA subgroup reduce dispatch failed: {error}"))
    });
    bytes_f32(&outputs[0])
}

#[test]
#[ignore = "blocked: NVIDIA driver 570.x JIT miscompiles f32 shfl on Blackwell sm_120 \
            (AOT/ptxas is correct) — see docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md; \
            re-enable when AOT cubin compilation lands"]
fn cuda_subgroup_max_f32_broadcasts_full_reduction_to_every_lane() {
    let input: Vec<f32> = (0..LANES).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let expected = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let out = run_subgroup_reduce(Expr::subgroup_max, &input);

    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(
            value, expected,
            "subgroup_max must broadcast the full max ({expected}) to lane {lane}, got {value} \
             (a reduce-to-lane-0 butterfly leaves later lanes with partial results)"
        );
    }
}

#[test]
#[ignore = "blocked: NVIDIA driver 570.x JIT miscompiles f32 shfl on Blackwell sm_120 \
            (AOT/ptxas is correct) — see docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md; \
            re-enable when AOT cubin compilation lands"]
fn cuda_subgroup_add_f32_broadcasts_full_reduction_to_every_lane() {
    let input: Vec<f32> = (0..LANES).map(|i| (i as f32) + 1.0).collect();
    let expected: f32 = input.iter().sum();

    let out = run_subgroup_reduce(Expr::subgroup_add, &input);

    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(
            value, expected,
            "subgroup_add must broadcast the full sum ({expected}) to lane {lane}, got {value}"
        );
    }
}

/// `out[gid] = subgroup_reduce(op, in[gid])` over a single 32-lane subgroup, u32.
/// The integer path lowers to a single all-lane-uniform `redux.sync.{op}.u32`
/// hardware instruction (verified well-formed: offline `ptxas` accepts it and an
/// AOT `__reduce_add_sync` of the same instruction returns the full reduction on
/// the 5090). It is STILL miscompiled by the Blackwell driver JIT — every lane
/// gets its own value, same as the f32 `shfl` path — so these tests are also
/// `#[ignore]`d on the JIT path. See `docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md`: the
/// driver JIT miscompiles cross-lane ops (`redux.sync` AND f32-bitcast `shfl`)
/// on a globally-loaded value; AOT/`ptxas` is correct for all of them.
fn run_subgroup_reduce_u32(reduce: fn(Expr) -> Expr, input: &[u32]) -> Vec<u32> {
    assert_eq!(input.len(), LANES as usize);
    let body = vec![
        Node::let_bind("gid", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("gid"), Expr::buf_len("out")),
            vec![Node::store(
                "out",
                Expr::var("gid"),
                reduce(Expr::load("in", Expr::var("gid"))),
            )],
        ),
    ];
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("in", 0, DataType::U32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        [LANES, 1, 1],
        vec![Node::Region {
            generator: "test::subgroup_reduce_parity_u32".into(),
            source_region: None,
            body: std::sync::Arc::new(body),
        }],
    );
    let inputs: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = with_live_backend("subgroup reduce parity u32", |backend| {
        backend
            .dispatch(&program, &[inputs.clone()], &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA u32 subgroup reduce dispatch failed: {error}"))
    });
    outputs[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "blocked: NVIDIA driver 570.x JIT miscompiles cross-lane redux.sync on \
            Blackwell sm_120 (AOT/ptxas is correct) — see \
            docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md; re-enable when AOT cubin compilation lands"]
fn cuda_subgroup_add_u32_broadcasts_full_reduction_to_every_lane() {
    let input: Vec<u32> = (0..LANES).map(|i| i + 1).collect();
    let expected: u32 = input.iter().sum();
    assert_eq!(expected, 528, "0+1+..+32 sanity");

    let out = run_subgroup_reduce_u32(Expr::subgroup_add, &input);

    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(
            value, expected,
            "u32 subgroup_add must broadcast the full sum ({expected}) to lane {lane}, got {value}"
        );
    }
}

#[test]
#[ignore = "blocked: NVIDIA driver 570.x JIT miscompiles cross-lane redux.sync on \
            Blackwell sm_120 (AOT/ptxas is correct) — see \
            docs/CUDA_DRIVER_JIT_F32_SHFL_BUG.md; re-enable when AOT cubin compilation lands"]
fn cuda_subgroup_max_u32_broadcasts_full_reduction_to_every_lane() {
    // Interleaved so the max (62) is NOT at lane 0 — a reduce-to-lane-0 path
    // would leave most lanes wrong.
    let input: Vec<u32> = (0..LANES).map(|i| (i * 2) ^ 1).collect();
    let expected: u32 = input.iter().copied().max().expect("non-empty");

    let out = run_subgroup_reduce_u32(Expr::subgroup_max, &input);

    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(
            value, expected,
            "u32 subgroup_max must broadcast the full max ({expected}) to lane {lane}, got {value}"
        );
    }
}
