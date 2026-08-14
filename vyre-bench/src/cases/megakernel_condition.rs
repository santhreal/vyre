use super::byte_pack::{gb_per_second, rate_per_second_x1000, read_word, write_word};
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRun, Correctness,
    DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    dispatch_artifact_timed, input_bytes_total, transfer_accounting, ResidentInputPool,
};
use crate::api::suite::SuiteKind;
use rayon::prelude::*;
use std::sync::Arc;
use vyre_foundation::ir::{Expr, Node};
use vyre_runtime::resident_work_queue::protocol::{
    ARG0_WORD, OPCODE_WORD, PRIORITY_WORD, STATUS_WORD, TENANT_WORD,
};
use vyre_runtime::resident_work_queue::{self, control, slot, OpcodeHandler, SLOT_WORDS};

/// Names this case's buffers in word codec errors.
const WORD_CONTEXT: &str = "megakernel condition output";

pub struct MegakernelCondition;

const SLOT_COUNT: u32 = 65_536;
const WORKGROUP_SIZE: u32 = 256;
const CONDITION_OPCODE: u32 = 0x1000;
const CONDITION_FIRED_WORD: u32 = control::METRICS_BASE;
const SUITES: &[SuiteKind] = &[SuiteKind::Release, SuiteKind::Gpu, SuiteKind::Deep];
const RESIDENT_SAMPLE_SETS: usize = 64;

struct MegakernelConditionPrepared {
    program: Arc<vyre_foundation::ir::Program>,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    expected_fired: u32,
    resident: Option<ResidentInputPool>,
}

impl BenchCase for MegakernelCondition {
    fn id(&self) -> BenchId {
        BenchId("runtime.megakernel.condition.64k".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Megakernel Condition Opcode 64k Slots".to_string(),
            description:
                "Custom megakernel opcode evaluating branchy rule predicates from slot payloads"
                    .to_string(),
            tags: vec![
                "runtime".to_string(),
                "megakernel".to_string(),
                "condition".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Runtime,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-runtime".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        SUITES
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_100x(
            "resident megakernel condition evaluation",
            "vyre-runtime",
            "optimized CPU condition evaluator baseline",
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let handler = condition_opcode_handler();
        let program = resident_work_queue::build_program_sharded_once_slots_control_report_shared(
            WORKGROUP_SIZE,
            SLOT_COUNT,
            &[handler],
        );
        let control_bytes = resident_work_queue::encode_control(false, 1, 0)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let mut expected_fired = 0u32;
        let ring_bytes = condition_ring(SLOT_COUNT, &mut expected_fired)?;
        let debug_bytes = resident_work_queue::encode_empty_debug_log(
            resident_work_queue::debug::RECORD_CAPACITY,
        )
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let io_bytes = resident_work_queue::io::try_encode_empty_io_queue(
            resident_work_queue::io::IO_SLOT_COUNT,
        )
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let inputs = vec![control_bytes, ring_bytes, debug_bytes, io_bytes];
        let input_bytes_total = input_bytes_total(&inputs);
        let resident = ResidentInputPool::upload_optional(
            ctx,
            &inputs,
            RESIDENT_SAMPLE_SETS,
            "megakernel condition bench",
        )?;
        Ok(Box::new(MegakernelConditionPrepared {
            program,
            inputs,
            input_bytes_total,
            expected_fired,
            resident,
        }))
    }

    fn program<'a>(&self, _prepared: &'a PreparedCase) -> Option<&'a vyre_foundation::ir::Program> {
        None
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_mut::<MegakernelConditionPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "megakernel condition prepared payload type mismatch".to_string(),
                )
            })?;
        let mut config = ctx.dispatch_config.clone();
        config.grid_override = Some([SLOT_COUNT.div_ceil(WORKGROUP_SIZE), 1, 1]);

        let dispatch = dispatch_artifact_timed(
            ctx,
            prepared.program.as_ref(),
            prepared.resident.as_mut(),
            &prepared.inputs,
            &config,
        )?;
        let resident_used = dispatch.resident_used;
        let elapsed = dispatch.timed.wall_ns;
        let dispatch_ns = dispatch.timed.device_ns;
        let outputs = dispatch.timed.outputs;
        let output_bytes_total = outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting = transfer_accounting(
            prepared.input_bytes_total,
            output_bytes_total,
            resident_used,
        );
        let device_ns = dispatch_ns.unwrap_or(elapsed);
        let start_ref = std::time::Instant::now();
        let baseline_outputs = simulate_condition_outputs(&prepared.inputs)?;
        let baseline_ns = start_ref.elapsed().as_nanos() as u64;
        let baseline_output_bytes = baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let baseline_bytes_touched = prepared
            .input_bytes_total
            .saturating_add(baseline_output_bytes);

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(elapsed),
                dispatch_ns,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes_total),
                bytes_touched: Some(accounting.bytes_touched),
                bytes_read: Some(accounting.bytes_read),
                bytes_written: Some(accounting.bytes_written),
                atomic_op_count: Some(u64::from(SLOT_COUNT + prepared.expected_fired)),
                wall_throughput_gb_s: Some(gb_per_second(accounting.bytes_touched, elapsed)),
                device_throughput_gb_s: Some(gb_per_second(accounting.bytes_touched, device_ns)),
                custom: vec![
                    MetricPoint {
                        name: "megakernel_condition_slots".to_string(),
                        value: u64::from(SLOT_COUNT),
                    },
                    MetricPoint {
                        name: "megakernel_condition_fired".to_string(),
                        value: u64::from(prepared.expected_fired),
                    },
                    MetricPoint {
                        name: "megakernel_condition_slots_per_sec_x1000".to_string(),
                        value: rate_per_second_x1000(u64::from(SLOT_COUNT), device_ns),
                    },
                    MetricPoint {
                        name: "megakernel_resident_input_pool_sets".to_string(),
                        value: if resident_used {
                            RESIDENT_SAMPLE_SETS as u64
                        } else {
                            0
                        },
                    },
                ],
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(baseline_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(baseline_output_bytes),
                bytes_touched: Some(baseline_bytes_touched),
                bytes_read: Some(prepared.input_bytes_total),
                bytes_written: Some(baseline_output_bytes),
                ..Default::default()
            }),
            outputs,
            baseline_outputs: Some(baseline_outputs),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()?;
        verify_condition_outputs(&run.outputs)
    }
}

fn condition_opcode_handler() -> OpcodeHandler {
    OpcodeHandler {
        opcode: CONDITION_OPCODE,
        body: vec![
            Node::let_bind(
                "condition_count",
                Expr::bitand(Expr::var("arg1"), Expr::u32(0xFFFF)),
            ),
            Node::let_bind(
                "condition_threshold",
                Expr::shr(Expr::var("arg1"), Expr::u32(16)),
            ),
            Node::let_bind(
                "condition_offset",
                Expr::bitand(Expr::var("arg2"), Expr::u32(0xFFFF)),
            ),
            Node::let_bind(
                "condition_limit",
                Expr::shr(Expr::var("arg2"), Expr::u32(16)),
            ),
            Node::let_bind(
                "condition_literals",
                Expr::eq(
                    Expr::bitand(Expr::var("arg0"), Expr::u32(0b11)),
                    Expr::u32(0b11),
                ),
            ),
            Node::let_bind(
                "condition_count_ok",
                Expr::ge(
                    Expr::var("condition_count"),
                    Expr::var("condition_threshold"),
                ),
            ),
            Node::let_bind(
                "condition_offset_ok",
                Expr::le(Expr::var("condition_offset"), Expr::var("condition_limit")),
            ),
            Node::let_bind(
                "condition_fired",
                Expr::and(
                    Expr::var("condition_literals"),
                    Expr::and(
                        Expr::var("condition_count_ok"),
                        Expr::var("condition_offset_ok"),
                    ),
                ),
            ),
            Node::if_then(
                Expr::var("condition_fired"),
                vec![Node::let_bind(
                    "condition_fired_prev",
                    Expr::atomic_add("control", Expr::u32(CONDITION_FIRED_WORD), Expr::u32(1)),
                )],
            ),
        ],
    }
}

fn condition_ring(slot_count: u32, expected_fired: &mut u32) -> Result<Vec<u8>, BenchError> {
    let mut ring = resident_work_queue::encode_empty_ring(slot_count)
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    for slot_index in 0..slot_count {
        let flags = condition_flags(slot_index);
        let count = condition_count(slot_index);
        let threshold = condition_threshold(slot_index);
        let offset = condition_offset(slot_index);
        let limit = condition_limit(slot_index);
        if condition_matches(flags, count, threshold, offset, limit) {
            *expected_fired = expected_fired.checked_add(1).ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "megakernel expected fired count overflowed".to_string(),
                )
            })?;
        }
        let base = slot_index.saturating_mul(SLOT_WORDS);
        write_word(
            &mut ring,
            base.saturating_add(STATUS_WORD),
            slot::PUBLISHED,
            WORD_CONTEXT,
        )?;
        write_word(
            &mut ring,
            base.saturating_add(OPCODE_WORD),
            CONDITION_OPCODE,
            WORD_CONTEXT,
        )?;
        write_word(&mut ring, base.saturating_add(TENANT_WORD), 0, WORD_CONTEXT)?;
        write_word(
            &mut ring,
            base.saturating_add(PRIORITY_WORD),
            slot::PRIORITY_NORMAL,
            WORD_CONTEXT,
        )?;
        write_word(
            &mut ring,
            base.saturating_add(ARG0_WORD),
            flags,
            WORD_CONTEXT,
        )?;
        write_word(
            &mut ring,
            base.saturating_add(ARG0_WORD + 1),
            count | (threshold << 16),
            WORD_CONTEXT,
        )?;
        write_word(
            &mut ring,
            base.saturating_add(ARG0_WORD + 2),
            offset | (limit << 16),
            WORD_CONTEXT,
        )?;
    }
    Ok(ring)
}

fn condition_flags(slot_index: u32) -> u32 {
    let literal_a = u32::from(slot_index % 2 != 0);
    let literal_b = u32::from(slot_index % 5 != 0) << 1;
    literal_a | literal_b
}

fn condition_count(slot_index: u32) -> u32 {
    slot_index.wrapping_mul(17).wrapping_add(11) & 0x3F
}

fn condition_threshold(slot_index: u32) -> u32 {
    8 + (slot_index.wrapping_mul(7) & 0x1F)
}

fn condition_offset(slot_index: u32) -> u32 {
    slot_index.wrapping_mul(19).wrapping_add(3) & 0xFF
}

fn condition_limit(slot_index: u32) -> u32 {
    32 + (slot_index.wrapping_mul(13) & 0x7F)
}

fn condition_matches(flags: u32, count: u32, threshold: u32, offset: u32, limit: u32) -> bool {
    flags & 0b11 == 0b11 && count >= threshold && offset <= limit
}

fn simulate_condition_outputs(inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, BenchError> {
    if inputs.len() != 4 {
        return Err(BenchError::ExecutionFailed(format!(
            "megakernel condition baseline received {} buffers, expected 4",
            inputs.len()
        )));
    }
    let mut control = inputs[0].clone();
    let ring = &inputs[1];
    let slot_bytes = usize::try_from(SLOT_WORDS)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| {
            BenchError::ExecutionFailed(
                "megakernel condition slot byte size overflowed usize".to_string(),
            )
        })?;
    let expected_ring_bytes = usize::try_from(SLOT_COUNT)
        .ok()
        .and_then(|slots| slots.checked_mul(slot_bytes))
        .ok_or_else(|| {
            BenchError::ExecutionFailed(
                "megakernel condition ring byte size overflowed usize".to_string(),
            )
        })?;
    if ring.len() < expected_ring_bytes {
        return Err(BenchError::ExecutionFailed(format!(
            "megakernel condition CPU baseline received {} ring bytes, expected at least {expected_ring_bytes}",
            ring.len()
        )));
    }
    let fired = ring[..expected_ring_bytes]
        .par_chunks_exact(slot_bytes)
        .map(|slot| {
            let flags = slot_word(slot, ARG0_WORD);
            let packed_count = slot_word(slot, ARG0_WORD + 1);
            let packed_offset = slot_word(slot, ARG0_WORD + 2);
            let count = packed_count & 0xFFFF;
            let threshold = packed_count >> 16;
            let offset = packed_offset & 0xFFFF;
            let limit = packed_offset >> 16;
            u32::from(condition_matches(flags, count, threshold, offset, limit))
        })
        .sum::<u32>();

    write_word(&mut control, control::DONE_COUNT, SLOT_COUNT, WORD_CONTEXT)?;
    write_word(&mut control, CONDITION_FIRED_WORD, fired, WORD_CONTEXT)?;
    Ok(vec![control, Vec::new(), Vec::new(), Vec::new()])
}

fn slot_word(slot: &[u8], word_index: u32) -> u32 {
    let offset = word_index as usize * 4;
    u32::from_le_bytes([
        slot[offset],
        slot[offset + 1],
        slot[offset + 2],
        slot[offset + 3],
    ])
}

fn verify_condition_outputs(outputs: &[Vec<u8>]) -> Result<Correctness, BenchError> {
    if outputs.len() != 4 {
        return Err(BenchError::CorrectnessViolation(format!(
            "megakernel condition returned {} buffers, expected 4",
            outputs.len()
        )));
    }
    let done_count = read_word(&outputs[0], control::DONE_COUNT, WORD_CONTEXT)?;
    if done_count != SLOT_COUNT {
        return Err(BenchError::CorrectnessViolation(format!(
            "megakernel condition DONE_COUNT was {done_count}, expected {SLOT_COUNT}"
        )));
    }
    let fired = read_word(&outputs[0], CONDITION_FIRED_WORD, WORD_CONTEXT)?;
    if fired == 0 {
        return Err(BenchError::CorrectnessViolation(
            "megakernel condition opcode produced zero fired predicates".to_string(),
        ));
    }
    for buffer_index in 1..outputs.len() {
        if !outputs[buffer_index].is_empty() {
            return Err(BenchError::CorrectnessViolation(format!(
                "megakernel condition hot path returned {} bytes for non-control buffer {buffer_index}",
                outputs[buffer_index].len()
            )));
        }
    }
    Ok(Correctness::Exact)
}

inventory::submit! {
    &MegakernelCondition as &'static dyn BenchCase
}
