use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRun, Correctness,
    DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{dispatch_artifact_timed, ResidentInputPool};
use crate::cases::resident_queue::{
    account, queue_buffers, resident_pool_sets_metric, timed_reference,
};
use std::sync::Arc;
use vyre_runtime::resident_work_queue::{self, protocol, ResidentWorkItem};

pub struct MegakernelTruth;

const WORK_ITEM_COUNT: usize = 1024;
const WORKER_COUNT: u32 = 256;
const RESIDENT_SAMPLE_SETS: usize = 8;
const SUITES: &[crate::api::suite::SuiteKind] = &[
    crate::api::suite::SuiteKind::Release,
    crate::api::suite::SuiteKind::Gpu,
    crate::api::suite::SuiteKind::Deep,
];

struct MegakernelTruthPrepared {
    program: Arc<vyre_foundation::ir::Program>,
    work_items: Vec<ResidentWorkItem>,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    resident: Option<ResidentInputPool>,
}

impl BenchCase for MegakernelTruth {
    fn id(&self) -> BenchId {
        BenchId("runtime.megakernel.truth.1024".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Megakernel Truth 1024 WorkItems".to_string(),
            description:
                "Actual megakernel dispatcher path with queue planning, publication, and backend timing"
                    .to_string(),
            tags: vec![
                "runtime".to_string(),
                "megakernel".to_string(),
                "truth".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Runtime,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-runtime".to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        SUITES
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let work_items = make_work_items(WORK_ITEM_COUNT)?;
        let slot_count = u32::try_from(WORK_ITEM_COUNT).map_err(|source| {
            BenchError::ExecutionFailed(format!(
                "megakernel truth work item count cannot fit u32: {source}"
            ))
        })?;
        let program = resident_work_queue::build_program_sharded_once_slots_control_report_shared(
            WORKER_COUNT,
            slot_count,
            &[],
        );
        let mut ring_words = Vec::new();
        vyre_runtime::resident_work_queue::ResidentWorkQueue::encode_work_items_ring_words_into(
            slot_count,
            0,
            &work_items,
            &mut ring_words,
        )
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let mut ring_bytes = Vec::with_capacity(ring_words.len().saturating_mul(4));
        for word in &ring_words {
            ring_bytes.extend_from_slice(&word.to_le_bytes());
        }
        let queue = queue_buffers(
            ctx,
            ring_bytes,
            RESIDENT_SAMPLE_SETS,
            "megakernel truth bench",
        )?;
        Ok(Box::new(MegakernelTruthPrepared {
            program,
            work_items,
            inputs: queue.inputs,
            input_bytes_total: queue.input_bytes_total,
            resident: queue.resident,
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
            .downcast_mut::<MegakernelTruthPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "megakernel truth prepared payload type mismatch".to_string(),
                )
            })?;

        let mut dispatch_config = ctx.dispatch_config.clone();
        dispatch_config.grid_override =
            Some([(WORK_ITEM_COUNT as u32).div_ceil(WORKER_COUNT), 1, 1]);
        let dispatch = dispatch_artifact_timed(
            ctx,
            prepared.program.as_ref(),
            prepared.resident.as_mut(),
            &prepared.inputs,
            &dispatch_config,
        )?;
        let sample = account(dispatch, prepared.input_bytes_total);
        let done_count = read_done_count(&sample.outputs)?;
        let (baseline_processed, baseline_ns) =
            timed_reference(|| simulate_cpu_drain(&prepared.work_items));

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(sample.wall_ns),
                dispatch_ns: Some(sample.wall_ns),
                kernel_queue_submit_ns: Some(0),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(sample.output_bytes_total),
                bytes_read: Some(sample.accounting.bytes_read),
                bytes_written: Some(sample.accounting.bytes_written),
                bytes_touched: Some(sample.accounting.bytes_touched),
                atomic_op_count: Some((WORK_ITEM_COUNT as u64).saturating_mul(2)),
                custom: vec![
                    MetricPoint {
                        name: "megakernel_backend_dispatch_ns".to_string(),
                        value: sample.wall_ns,
                    },
                    MetricPoint {
                        name: "megakernel_published_items".to_string(),
                        value: WORK_ITEM_COUNT as u64,
                    },
                    MetricPoint {
                        name: "megakernel_items_processed".to_string(),
                        value: done_count,
                    },
                    MetricPoint {
                        name: "megakernel_items_remaining".to_string(),
                        value: (WORK_ITEM_COUNT as u64).saturating_sub(done_count),
                    },
                    MetricPoint {
                        name: "megakernel_kernel_launches".to_string(),
                        value: 1,
                    },
                    resident_pool_sets_metric(sample.resident_used, RESIDENT_SAMPLE_SETS),
                    MetricPoint {
                        name: "megakernel_backend_neutral_cuda_path".to_string(),
                        value: u64::from(ctx.preferred_backend.id() == "cuda"),
                    },
                ],
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(baseline_ns),
                cpu_ns: Some(baseline_ns),
                input_bytes: Some(prepared.input_bytes_total),
                bytes_read: Some(prepared.input_bytes_total),
                bytes_touched: Some(prepared.input_bytes_total),
                custom: vec![MetricPoint {
                    name: "megakernel_items_processed".to_string(),
                    value: baseline_processed,
                }],
                ..Default::default()
            }),
            outputs: vec![done_count.to_le_bytes().to_vec()],
            baseline_outputs: Some(vec![baseline_processed.to_le_bytes().to_vec()]),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<MegakernelTruthPrepared>()
            .map(|prepared| (prepared.input_bytes_total, 0))
            .unwrap_or((0, 0))
    }
}

fn read_done_count(outputs: &[Vec<u8>]) -> Result<u64, BenchError> {
    let control = outputs.first().ok_or_else(|| {
        BenchError::CorrectnessViolation(
            "megakernel truth dispatch produced no control output".to_string(),
        )
    })?;
    let done = vyre_runtime::resident_work_queue::try_read_done_count(control)
        .map_err(|error| BenchError::CorrectnessViolation(error.to_string()))?;
    Ok(u64::from(done))
}

fn make_work_items(count: usize) -> Result<Vec<ResidentWorkItem>, BenchError> {
    let mut items = Vec::with_capacity(count);
    for index in 0..count {
        let word = u32::try_from(index).map_err(|_| {
            BenchError::ExecutionFailed(
                "megakernel truth work item index exceeded u32::MAX".to_string(),
            )
        })?;
        items.push(ResidentWorkItem {
            op_handle: protocol::opcode::NOP,
            input_handle: word,
            output_handle: word,
            param: word,
        });
    }
    Ok(items)
}

fn simulate_cpu_drain(items: &[ResidentWorkItem]) -> u64 {
    items.iter().fold(0_u64, |count, item| {
        count.saturating_add(if item.op_handle == protocol::opcode::NOP {
            1
        } else {
            0
        })
    })
}

inventory::submit! {
    &MegakernelTruth as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_drain_counts_nop_items() {
        let items = make_work_items(8).expect("Fix: fixture");

        assert_eq!(simulate_cpu_drain(&items), 8);
    }
}
