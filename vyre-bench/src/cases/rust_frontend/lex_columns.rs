//! The four-column lexer dispatch shared by the single-source and batched Rust
//! lexer cases.
//!
//! Both cases run one IR lexer plan that writes token kinds, starts, lengths
//! and counts as live-out columns, then report the same wall, device and byte
//! accounting against a CPU lexer baseline. Only the plan, the fixture and the
//! custom metric names are per case.

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;

pub(super) const LEX_SUITES: &[SuiteKind] = &[
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

/// The prepared lexer dispatch and the CPU baseline it is checked against.
pub(super) struct LexColumns {
    pub(super) program: vyre::ir::Program,
    pub(super) inputs: Vec<Vec<u8>>,
    /// Source bytes handed to the plan, reported as wire bytes.
    pub(super) source_bytes: u64,
    pub(super) baseline_outputs: Vec<Vec<u8>>,
    pub(super) baseline_wall_ns: u64,
    pub(super) token_count: usize,
}

/// How a case names its plan and its live-out columns when the dispatch returns
/// the wrong number of them.
#[derive(Clone, Copy)]
pub(super) struct LexColumnsContract {
    pub(super) plan: &'static str,
    pub(super) columns: &'static str,
}

/// One measured lexer dispatch.
pub(super) struct LexSample {
    pub(super) wall_ns: u64,
    pub(super) device_ns: Option<u64>,
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) input_bytes: u64,
    pub(super) output_bytes: u64,
}

/// Dispatch the lexer plan once and check that it returned all four columns.
pub(super) fn lex_columns_sample(
    ctx: &BenchContext,
    lex: &LexColumns,
    grid: Option<[u32; 3]>,
    contract: LexColumnsContract,
) -> Result<LexSample, BenchError> {
    let mut dispatch_config = ctx.dispatch_config.clone();
    if let Some(grid) = grid {
        dispatch_config.grid_override = Some(grid);
    }
    let timed = ctx
        .dispatch_timed(&lex.program, &lex.inputs, &dispatch_config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    if timed.outputs.len() != 4 {
        let LexColumnsContract { plan, columns } = contract;
        return Err(BenchError::BackendFailed(format!(
            "{plan} IR must return 4 live-out columns {columns}, got {}",
            timed.outputs.len()
        )));
    }

    Ok(LexSample {
        input_bytes: lex.inputs.iter().map(Vec::len).sum::<usize>() as u64,
        output_bytes: timed.outputs.iter().map(Vec::len).sum::<usize>() as u64,
        wall_ns: timed.wall_ns,
        device_ns: timed.device_ns,
        outputs: timed.outputs,
    })
}

/// Assemble the run record shared by both lexer cases, adding the per-case
/// custom metric points.
pub(super) fn lex_columns_run(
    ctx: &BenchContext,
    lex: &LexColumns,
    sample: LexSample,
    custom: Vec<MetricPoint>,
) -> BenchRun {
    let baseline_bytes = lex.baseline_bytes();
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(sample.wall_ns),
            dispatch_ns: Some(sample.wall_ns),
            kernel_execute_ns: sample.device_ns.filter(|ns| *ns > 0),
            input_bytes: Some(sample.input_bytes),
            output_bytes: Some(sample.output_bytes),
            bytes_read: Some(sample.input_bytes),
            bytes_written: Some(sample.output_bytes),
            wire_bytes: Some(lex.source_bytes),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(lex.baseline_wall_ns),
            input_bytes: Some(sample.input_bytes),
            output_bytes: Some(baseline_bytes),
            bytes_read: Some(lex.source_bytes),
            bytes_written: Some(baseline_bytes),
            wire_bytes: Some(lex.source_bytes),
            ..Default::default()
        }),
        outputs: sample.outputs,
        baseline_outputs: ctx
            .include_baseline_outputs
            .then(|| lex.baseline_outputs.clone()),
    }
}

/// Bytes one lexer sample reads and writes.
pub(super) fn lex_columns_bytes_touched(lex: &LexColumns) -> (u64, u64) {
    (
        lex.inputs.iter().map(Vec::len).sum::<usize>() as u64,
        lex.baseline_bytes(),
    )
}

impl LexColumns {
    fn baseline_bytes(&self) -> u64 {
        self.baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64
    }
}

/// The live-out wire format both lexer plans write: three token columns then
/// the token counts.
pub(super) fn lex_baseline_columns(
    kinds: &[u32],
    starts: &[u32],
    lens: &[u32],
    counts: &[u32],
) -> Vec<Vec<u8>> {
    vec![
        u32s_to_bytes(kinds),
        u32s_to_bytes(starts),
        u32s_to_bytes(lens),
        u32s_to_bytes(counts),
    ]
}

pub(super) fn u32s_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[cfg(test)]
pub(super) fn decode_u32_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
        .collect()
}
