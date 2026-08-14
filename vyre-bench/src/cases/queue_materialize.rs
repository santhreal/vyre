//! What a queue-materialize benchmark carries, and how it takes one sample.
//!
//! Two cases materialize a packed frontier into a sparse active queue and then
//! traverse that queue: the CSR frontier family and the IFDS family. They differ
//! in fixture, CPU oracle, and reported metric points. The payload they carry
//! between prepare and run, the workgroup they validate, the resident-or-host
//! choice of dispatch path, the identity they fingerprint, and the run they
//! assemble are one fact each, stated here.

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::ResidentInputSet;
use crate::cases::queue_stage::{
    dispatch_host_queue_sequence, dispatch_resident_queue_sequence, HostQueueSequenceSpec,
    QueueSequenceRun, ResidentQueueSequenceSpec, QUEUE_RESET_GRID,
};
use vyre_foundation::ir::Program;

/// Frontier bitset width a fixture's stats record.
///
/// The queue-build grid is sized from frontier words rather than nodes, and that
/// is the only stats field the shared sampling path reads.
pub(crate) trait FrontierWords {
    fn frontier_words(&self) -> u32;
}

/// Everything a queue-materialize case builds during prepare.
///
/// `S` is the family's own stats record, which the case reports and the shared
/// path only reads `frontier_words` from.
pub(crate) struct QueueMaterializePrepared<S> {
    pub(crate) reset_program: Program,
    pub(crate) queue_program: Program,
    pub(crate) traverse_program: Program,
    pub(crate) traverse_grid: [u32; 3],
    pub(crate) row_strided_traverse: bool,
    pub(crate) split_high_degree_traverse: bool,
    pub(crate) high_traverse_program: Option<Program>,
    pub(crate) high_traverse_grid: [u32; 3],
    pub(crate) high_degree_queue_capacity: u32,
    pub(crate) traverse_logical_lanes: u64,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) baseline_output: Vec<u8>,
    pub(crate) baseline_wall_ns: u64,
    pub(crate) stats: S,
    pub(crate) queue_capacity: u32,
    pub(crate) resident: Option<ResidentInputSet>,
}

/// The workgroup every stage of the sequence runs at.
///
/// The sequence programs are built together, so the queue program's workgroup is
/// the sequence's workgroup and a caller override can only disagree with it.
/// `label` names the case in both errors, e.g. `"skewed CSR queue"`.
pub(crate) fn queue_materialize_workgroup<S>(
    ctx: &BenchContext,
    prepared: &QueueMaterializePrepared<S>,
    label: &str,
) -> Result<[u32; 3], BenchError> {
    let workgroup = prepared.queue_program.workgroup_size();
    if workgroup.contains(&0) {
        return Err(BenchError::ExecutionFailed(format!(
            "{label} benchmark received invalid workgroup {workgroup:?}. Fix: use positive dispatch dimensions."
        )));
    }
    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "{label} resident sequence uses program workgroup {workgroup:?}, but received override {override_workgroup:?}. Fix: run the queue sequence without a workgroup override or rebuild all sequence programs."
            )));
        }
    }
    Ok(workgroup)
}

/// Run the reset-then-materialize-then-traverse sequence.
///
/// Resident buffers keep every stage on the device; without them the same stages
/// run as separate host dispatches. `context` is the family's noun inside every
/// stage label, e.g. `"IFDS"`.
pub(crate) fn dispatch_queue_materialize_sequence<S: FrontierWords>(
    ctx: &BenchContext,
    prepared: &QueueMaterializePrepared<S>,
    workgroup: [u32; 3],
    context: &'static str,
) -> Result<QueueSequenceRun, BenchError> {
    match prepared.resident.as_ref() {
        Some(resident) => dispatch_resident_queue_sequence(
            ctx,
            ResidentQueueSequenceSpec {
                reset_program: &prepared.reset_program,
                queue_program: &prepared.queue_program,
                traverse_program: &prepared.traverse_program,
                high_traverse_program: prepared.high_traverse_program.as_ref(),
                frontier_words: prepared.stats.frontier_words(),
                traverse_grid: prepared.traverse_grid,
                high_traverse_grid: prepared.high_traverse_grid,
                baseline_output_len: prepared.baseline_output.len(),
                context,
            },
            resident,
            workgroup,
        ),
        None => dispatch_host_queue_sequence(
            ctx,
            HostQueueSequenceSpec {
                inputs: &prepared.inputs,
                reset_program: &prepared.reset_program,
                queue_program: &prepared.queue_program,
                traverse_program: &prepared.traverse_program,
                high_traverse_program: prepared.high_traverse_program.as_ref(),
                frontier_words: prepared.stats.frontier_words(),
                traverse_grid: prepared.traverse_grid,
                high_traverse_grid: prepared.high_traverse_grid,
                context,
            },
            workgroup,
        ),
    }
}

/// Workload identity of the whole sequence rather than of one program.
///
/// A single traverse fingerprint would call two different sequences the same
/// workload whenever they happen to share their traversal kernel, so the reset
/// and queue-build programs, every grid, and the family's own discriminating
/// values go in too.
pub(crate) fn queue_materialize_sequence_fingerprint<S>(
    domain: &[u8],
    prepared: &QueueMaterializePrepared<S>,
    extra_values: &[u32],
) -> [u8; 32] {
    crate::cases::queue_stage::queue_materialize_sequence_fingerprint(
        domain,
        [
            &prepared.reset_program,
            &prepared.queue_program,
            &prepared.traverse_program,
        ],
        prepared.high_traverse_program.as_ref(),
        [
            QUEUE_RESET_GRID,
            prepared.queue_program.workgroup_size(),
            prepared.traverse_grid,
            prepared.high_traverse_grid,
        ],
        extra_values,
    )
}

/// Assemble the run a queue-materialize case reports.
///
/// The sequence accounts its own transfers, because a resident sequence moves
/// only the final readback across the bus while a host sequence moves every
/// stage's buffers both ways.
pub(crate) fn queue_materialize_run<S>(
    prepared: &QueueMaterializePrepared<S>,
    sequence: QueueSequenceRun,
    custom: Vec<MetricPoint>,
    baseline_custom: Vec<MetricPoint>,
) -> BenchRun {
    let output_bytes = sequence.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(sequence.wall_ns),
            dispatch_ns: sequence.dispatch_ns,
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(output_bytes),
            bytes_read: Some(sequence.bytes_read),
            bytes_written: Some(sequence.bytes_written),
            bytes_touched: Some(sequence.bytes_read.saturating_add(sequence.bytes_written)),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(prepared.baseline_wall_ns),
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(prepared.baseline_output.len() as u64),
            custom: baseline_custom,
            ..Default::default()
        }),
        outputs: sequence.outputs,
        baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
    }
}

/// Bytes a queue-materialize sample reads and writes.
pub(crate) fn queue_materialize_bytes_touched<S>(
    prepared: &QueueMaterializePrepared<S>,
) -> (u64, u64) {
    (
        prepared.input_bytes_total,
        prepared.baseline_output.len() as u64,
    )
}
