//! `interpreter.bytecode.dispatch.10m`  -  Threaded bytecode interpreter.
//!
//! Executes a 10M-instruction trace of a simple stack-based bytecode VM.
//! GPU kernel interprets opcodes in parallel over independent program instances.
//! CPU baseline uses a hand-tuned switch-dispatch loop.
//!
//! This is deeply CPU-favorable: branch prediction makes CPU interpreters
//! fast despite being serial. The GPU must amortize branch divergence via
//! massive parallelism over independent program instances.

use crate::api::case::{BenchCase, BenchContext, BenchError};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use crate::cases::reference_sample::{
    measure_against_reference, referenced_bytes_touched, referenced_program, HostReferencePayload,
    HostReferenced,
};
use rand::{RngExt, SeedableRng};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Number of independent program instances to run in parallel.
const INSTANCE_COUNT: u32 = 4096;
/// Instructions per instance.
const INSTRS_PER_INSTANCE: u32 = 2500;
/// Total instruction words = INSTANCE_COUNT * INSTRS_PER_INSTANCE = 10M
const TOTAL_INSTRS: u32 = INSTANCE_COUNT * INSTRS_PER_INSTANCE;

// Opcodes (encoded in low 8 bits of u32 instruction word)
const OP_PUSH: u32 = 0;
const OP_ADD: u32 = 1;
const OP_MUL: u32 = 2;
const OP_DUP: u32 = 3;
const OP_SWAP: u32 = 4;

struct BytecodeDispatchPrepared {
    dispatch: HostReferencePayload,
    instrs: Vec<u32>,
}

impl HostReferenced for BytecodeDispatchPrepared {
    fn dispatch(&self) -> &HostReferencePayload {
        &self.dispatch
    }

    fn reference(&self) -> Result<Vec<u8>, BenchError> {
        Ok(cpu_interpret(
            &self.instrs,
            INSTANCE_COUNT as usize,
            INSTRS_PER_INSTANCE as usize,
        ))
    }
}

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "interpreter.bytecode.dispatch.10m",
    "Bytecode Interpreter 10M",
    "Stack-based bytecode VM: 4096 instances × 2500 instructions each",
    &["honest", "branch-heavy", "serial"],
    TOTAL_INSTRS as u64 * 4 + INSTANCE_COUNT as u64 * 4,
    Some(ContractDescription::cpu_sota(
        "Bytecode interpreter",
        "vyre-bench",
        "in-tree scalar Rust match-dispatch interpreter loop (cpu_interpret)",
        3.0,
    )),
);

static OPS: CaseOps<BytecodeDispatchPrepared> = CaseOps {
    build: prepare_bytecode_dispatch,
    measure: measure_against_reference::<BytecodeDispatchPrepared>,
    verify: verify_exact,
    program: referenced_program::<BytecodeDispatchPrepared>,
    fingerprint: None,
    bytes_touched: referenced_bytes_touched::<BytecodeDispatchPrepared>,
};

static CASE: HarnessCase<BytecodeDispatchPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// One instruction trace, one opcode plus immediate per word.
///
/// The stream is seeded, so the dispatched bytes are the same on every sample;
/// generating it once during preparation keeps that host work out of the sampled
/// loop entirely.
fn instruction_trace() -> Vec<u32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_BABE);
    let mut instrs = vec![0u32; TOTAL_INSTRS as usize];
    for instr in &mut instrs {
        let op = rng.random_range(0..5u32);
        let imm = if op == OP_PUSH {
            rng.random_range(1..256u32)
        } else {
            0
        };
        *instr = op | (imm << 8);
    }
    instrs
}

fn prepare_bytecode_dispatch(
    _ctx: &mut BenchContext,
) -> Result<BytecodeDispatchPrepared, BenchError> {
    let instrs = instruction_trace();
    let inputs = vec![vyre_primitives::wire::pack_u32_slice(&instrs)];

    Ok(BytecodeDispatchPrepared {
        dispatch: HostReferencePayload::host_buffers(
            bytecode_program(INSTANCE_COUNT, INSTRS_PER_INSTANCE),
            inputs,
        ),
        instrs,
    })
}

fn bytecode_program(instance_count: u32, instrs_per_instance: u32) -> Program {
    let total_instrs = instance_count.saturating_mul(instrs_per_instance);
    Program::wrapped(
        vec![
            BufferDecl::storage("instrs", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_instrs),
            BufferDecl::output("results", 1, DataType::U32).with_count(instance_count),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(instance_count)),
                vec![
                    Node::let_bind(
                        "base",
                        Expr::mul(Expr::var("tid"), Expr::u32(instrs_per_instance)),
                    ),
                    Node::let_bind("s0", Expr::u32(0)),
                    Node::let_bind("s1", Expr::u32(0)),
                    Node::let_bind("s2", Expr::u32(0)),
                    Node::let_bind("s3", Expr::u32(0)),
                    Node::Loop {
                        var: "pc".into(),
                        from: Expr::u32(0),
                        to: Expr::u32(instrs_per_instance),
                        body: vec![
                            Node::let_bind(
                                "instr",
                                Expr::load("instrs", Expr::add(Expr::var("base"), Expr::var("pc"))),
                            ),
                            Node::let_bind("op", Expr::bitand(Expr::var("instr"), Expr::u32(0xFF))),
                            Node::let_bind("imm", Expr::shr(Expr::var("instr"), Expr::u32(8))),
                            Node::if_then(
                                Expr::eq(Expr::var("op"), Expr::u32(OP_PUSH)),
                                vec![
                                    Node::assign("s3", Expr::var("s2")),
                                    Node::assign("s2", Expr::var("s1")),
                                    Node::assign("s1", Expr::var("s0")),
                                    Node::assign("s0", Expr::var("imm")),
                                ],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("op"), Expr::u32(OP_ADD)),
                                vec![
                                    Node::assign("s0", Expr::add(Expr::var("s0"), Expr::var("s1"))),
                                    Node::assign("s1", Expr::var("s2")),
                                    Node::assign("s2", Expr::var("s3")),
                                    Node::assign("s3", Expr::u32(0)),
                                ],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("op"), Expr::u32(OP_MUL)),
                                vec![
                                    Node::assign("s0", Expr::mul(Expr::var("s0"), Expr::var("s1"))),
                                    Node::assign("s1", Expr::var("s2")),
                                    Node::assign("s2", Expr::var("s3")),
                                    Node::assign("s3", Expr::u32(0)),
                                ],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("op"), Expr::u32(OP_DUP)),
                                vec![
                                    Node::assign("s3", Expr::var("s2")),
                                    Node::assign("s2", Expr::var("s1")),
                                    Node::assign("s1", Expr::var("s0")),
                                ],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("op"), Expr::u32(OP_SWAP)),
                                vec![
                                    Node::let_bind("tmp", Expr::var("s0")),
                                    Node::assign("s0", Expr::var("s1")),
                                    Node::assign("s1", Expr::var("tmp")),
                                ],
                            ),
                        ],
                    },
                    Node::store("results", Expr::var("tid"), Expr::var("s0")),
                ],
            ),
        ],
    )
}

/// CPU interpreter  -  processes bytecode with a simple switch loop.
fn cpu_interpret(instrs: &[u32], instances: usize, instrs_per: usize) -> Vec<u8> {
    let mut results = vec![0u32; instances];
    for instance in 0..instances {
        let base = instance * instrs_per;
        let mut s = [0u32; 4];
        for pc in 0..instrs_per {
            let instr = instrs[base + pc];
            let op = instr & 0xFF;
            let imm = instr >> 8;
            match op {
                0 => {
                    // PUSH
                    s[3] = s[2];
                    s[2] = s[1];
                    s[1] = s[0];
                    s[0] = imm;
                }
                1 => {
                    // ADD
                    s[0] = s[0].wrapping_add(s[1]);
                    s[1] = s[2];
                    s[2] = s[3];
                    s[3] = 0;
                }
                2 => {
                    // MUL
                    s[0] = s[0].wrapping_mul(s[1]);
                    s[1] = s[2];
                    s[2] = s[3];
                    s[3] = 0;
                }
                3 => {
                    // DUP
                    s[3] = s[2];
                    s[2] = s[1];
                    s[1] = s[0];
                }
                4 => {
                    // SWAP
                    s.swap(0, 1);
                }
                _ => {}
            }
        }
        results[instance] = s[0];
    }
    vyre_primitives::wire::pack_u32_slice(&results)
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "device-tests")]
    use std::sync::Mutex;
    use vyre_driver::{DispatchConfig, VyreBackend};

    /// Both device tests dispatch on the one physical adapter, so they take
    /// turns rather than racing for it.
    #[cfg(feature = "device-tests")]
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[cfg(feature = "device-tests")]
    fn stack_carrier_snapshot_instrs() -> Vec<u32> {
        vec![
            OP_SWAP,
            OP_SWAP,
            OP_PUSH | (192 << 8),
            OP_ADD,
            OP_PUSH | (222 << 8),
            OP_SWAP,
            OP_MUL,
        ]
    }

    #[test]
    fn bytecode_program_matches_cpu_reference_on_stack_ops() {
        let instrs = vec![
            OP_PUSH | (2 << 8),
            OP_PUSH | (3 << 8),
            OP_ADD,
            OP_DUP,
            OP_PUSH | (5 << 8),
            OP_SWAP,
            OP_MUL,
        ];
        let inputs = vec![vyre_primitives::wire::pack_u32_slice(&instrs)];
        let program = bytecode_program(1, instrs.len() as u32);
        let outputs = vyre_driver_reference::CpuRefBackend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .expect("Fix: cpu-ref bytecode VM dispatch must succeed");
        let expected = cpu_interpret(&instrs, 1, instrs.len());

        assert_eq!(
            outputs,
            vec![expected],
            "official IR reference semantics must match the bytecode benchmark baseline"
        );
    }

    // Dispatches on a physical GPU: gated on `device-tests` so the hosted CI
    // matrix, which has no Vulkan adapter, does not report the absence of
    // hardware as a parity defect. `gpu-parity.yml` enables the feature.
    #[cfg(feature = "device-tests")]
    #[test]
    fn bytecode_program_wgpu_matches_seeded_cpu_trace() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|error| {
            panic!("benchmark GPU test lock was poisoned: {error}");
        });
        let instrs = stack_carrier_snapshot_instrs();
        let backend = vyre_driver_wgpu::WgpuBackend::new()
            .expect("Fix: wgpu backend must initialize on the release GPU machine");
        let inputs = vec![vyre_primitives::wire::pack_u32_slice(&instrs)];
        let program = bytecode_program(1, instrs.len() as u32);
        let outputs = backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .expect("Fix: wgpu bytecode VM dispatch must succeed");
        let expected = cpu_interpret(&instrs, 1, instrs.len());

        assert_eq!(
            outputs,
            vec![expected],
            "wgpu must snapshot stack carriers during SWAP instead of aliasing later carrier writes"
        );
    }

    // vyre-driver-cuda is a dependency only under cfg(not(target_os = "macos")),
    // so the crate is not nameable on macOS and the test cannot be written there.
    // `device-tests` gates the live dispatch for the same reason as the wgpu test
    // above: a hosted runner has no CUDA driver to load.
    #[cfg(all(not(target_os = "macos"), feature = "device-tests"))]
    #[test]
    fn bytecode_program_cuda_matches_seeded_cpu_trace() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|error| {
            panic!("benchmark GPU test lock was poisoned: {error}");
        });
        let instrs = stack_carrier_snapshot_instrs();
        let backend = vyre_driver_cuda::CudaBackend::acquire()
            .expect("Fix: CUDA backend must initialize on the release GPU machine");
        let inputs = vec![vyre_primitives::wire::pack_u32_slice(&instrs)];
        let program = bytecode_program(1, instrs.len() as u32);
        let outputs = backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .expect("Fix: CUDA bytecode VM dispatch must succeed");
        let expected = cpu_interpret(&instrs, 1, instrs.len());

        assert_eq!(
            outputs,
            vec![expected],
            "CUDA must snapshot stack carriers during SWAP instead of aliasing later carrier writes"
        );
    }
}
