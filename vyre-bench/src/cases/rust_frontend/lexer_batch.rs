use super::lex_columns::{
    lex_baseline_columns, lex_columns_bytes_touched, lex_columns_run, lex_columns_sample,
    u32s_to_bytes, LexColumns, LexColumnsContract, LexSample, LEX_SUITES,
};
use super::rust_source_words;
use crate::api::case::{BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, WorkloadClass};
use crate::api::metric::{elapsed_ns, MetricPoint};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use vyre_foundation::ir::Program;
use vyre_frontend_rust::lex::lexer::cpu_lexer::lex as lex_cpu;
use vyre_frontend_rust::lex::lexer::plan::rust_lexer_batch;

const RUST_LEXER_BATCH_SOURCES: usize = 2048;
const WORKGROUP_SIZE: u32 = 256;

const CONTRACT: LexColumnsContract = LexColumnsContract {
    plan: "Rust batch lexer",
    columns: "[types, starts, lens, counts]",
};

struct RustLexerBatchPrepared {
    lex: LexColumns,
    source_count: u32,
    token_stride: usize,
}

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "frontend.rust.lexer.batch_ir_execute",
    name: "Batched Rust GPU Lexer IR Execute",
    summary: "Many small Rust nano-subset sources packed into one GPU lexer dispatch with exact per-source CPU lexer column parity",
    tags: &[
        "frontend-rust",
        "gpu-lexer",
        "lexer",
        "batch",
        "many-source",
        "tokenization",
        "ir-lexer",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-frontend-rust",
    suites: LEX_SUITES,
    min_input_bytes: Some((RUST_LEXER_BATCH_SOURCES * 192) as u64),
    feature_set: &["rust-frontend", "gpu-lexer", "batched-lexer", "ir-lexer"],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<RustLexerBatchPrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: batch_program,
    fingerprint: None,
    bytes_touched: batch_bytes_touched,
};

static CASE: HarnessCase<RustLexerBatchPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn batch_program(prepared: &RustLexerBatchPrepared) -> Option<&Program> {
    Some(&prepared.lex.program)
}

fn batch_bytes_touched(prepared: &RustLexerBatchPrepared) -> (u64, u64) {
    lex_columns_bytes_touched(&prepared.lex)
}

fn build_case(_ctx: &mut BenchContext) -> Result<RustLexerBatchPrepared, BenchError> {
    let sources = rust_lexer_batch_sources();
    let layout = RustLexerBatchLayout::from_sources(&sources)?;
    let source_count = u32::try_from(sources.len()).map_err(|_| {
        BenchError::ExecutionFailed(
            "Rust lexer batch source count exceeds u32-addressable plan limit".to_string(),
        )
    })?;
    let haystack_len = u32::try_from(layout.packed_source.len()).map_err(|_| {
        BenchError::ExecutionFailed(
            "Rust lexer batch source bytes exceed u32-addressable plan limit".to_string(),
        )
    })?;
    let token_stride = u32::try_from(layout.token_stride).map_err(|_| {
        BenchError::ExecutionFailed(
            "Rust lexer batch token stride exceeds u32-addressable plan limit".to_string(),
        )
    })?;
    let program = rust_lexer_batch(
        "haystack",
        "source_offsets",
        "source_lens",
        "out_tok_types",
        "out_tok_starts",
        "out_tok_lens",
        "out_counts",
        haystack_len,
        source_count,
        token_stride,
    );
    let inputs = layout.inputs();

    let baseline_start = std::time::Instant::now();
    let (baseline_outputs, token_count) =
        rust_lexer_batch_baseline_outputs(&sources, layout.token_stride)?;
    let baseline_wall_ns = elapsed_ns(baseline_start);

    Ok(RustLexerBatchPrepared {
        lex: LexColumns {
            program,
            inputs,
            source_bytes: layout.packed_source.len() as u64,
            baseline_outputs,
            baseline_wall_ns,
            token_count,
        },
        source_count,
        token_stride: layout.token_stride,
    })
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut RustLexerBatchPrepared,
) -> Result<BenchRun, BenchError> {
    let grid = [prepared.source_count.div_ceil(WORKGROUP_SIZE).max(1), 1, 1];
    let sample = lex_columns_sample(ctx, &prepared.lex, Some(grid), CONTRACT)?;
    let custom = batch_metric_points(prepared, &sample);
    Ok(lex_columns_run(ctx, &prepared.lex, sample, custom))
}

fn batch_metric_points(prepared: &RustLexerBatchPrepared, sample: &LexSample) -> Vec<MetricPoint> {
    vec![
        MetricPoint {
            name: "rust_frontend_gpu_lexer_batch_speedup_x1000".to_string(),
            value: super::speedup_x1000(prepared.lex.baseline_wall_ns, sample.wall_ns),
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_batch_tokens".to_string(),
            value: prepared.lex.token_count as u64,
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_batch_source_bytes".to_string(),
            value: prepared.lex.source_bytes,
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_batch_sources".to_string(),
            value: u64::from(prepared.source_count),
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_batch_token_stride".to_string(),
            value: prepared.token_stride as u64,
        },
    ]
}

struct RustLexerBatchLayout {
    packed_source: Vec<u8>,
    offsets: Vec<u32>,
    lens: Vec<u32>,
    token_stride: usize,
}

impl RustLexerBatchLayout {
    fn from_sources(sources: &[Vec<u8>]) -> Result<Self, BenchError> {
        let mut packed_source = Vec::new();
        let mut offsets = Vec::with_capacity(sources.len());
        let mut lens = Vec::with_capacity(sources.len());
        let mut token_stride = 1usize;
        for (source_idx, source) in sources.iter().enumerate() {
            offsets.push(u32::try_from(packed_source.len()).map_err(|_| {
                BenchError::ExecutionFailed(format!(
                    "Rust lexer batch byte offset for source {source_idx} exceeds u32"
                ))
            })?);
            lens.push(u32::try_from(source.len()).map_err(|_| {
                BenchError::ExecutionFailed(format!(
                    "Rust lexer batch source {source_idx} length exceeds u32"
                ))
            })?);
            token_stride = token_stride.max(source.len().saturating_add(1).max(1));
            packed_source.extend_from_slice(source);
        }
        Ok(Self {
            packed_source,
            offsets,
            lens,
            token_stride,
        })
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        let token_slots = self.offsets.len().max(1) * self.token_stride;
        let zero_tokens = vec![0u8; token_slots * std::mem::size_of::<u32>()];
        vec![
            u32s_to_bytes(&rust_source_words(&self.packed_source)),
            u32s_to_bytes(&self.offsets),
            u32s_to_bytes(&self.lens),
            zero_tokens.clone(),
            zero_tokens.clone(),
            zero_tokens,
            vec![0u8; self.offsets.len().max(1) * std::mem::size_of::<u32>()],
        ]
    }
}

fn rust_lexer_batch_sources() -> Vec<Vec<u8>> {
    let mut sources = Vec::with_capacity(RUST_LEXER_BATCH_SOURCES);
    for idx in 0..RUST_LEXER_BATCH_SOURCES {
        sources.push(
            format!(
                "fn stress_file_{idx}(n: i32, flag: bool) -> i32 {{
    let mut acc: i32 = {};
    /* per-file block comment to force bounded scanning */
    for i in -{}..n {{
        // line comment with branch-heavy token stream
        if i <= {} && flag != false {{
            acc += i * {};
        }} else {{
            acc -= i % {};
        }};
    }}
    return acc;
}}
",
                idx % 31,
                (idx % 7) + 1,
                (idx % 13) + 1,
                (idx % 5) + 2,
                (idx % 11) + 2
            )
            .into_bytes(),
        );
    }
    sources
}

fn rust_lexer_batch_baseline_outputs(
    sources: &[Vec<u8>],
    token_stride: usize,
) -> Result<(Vec<Vec<u8>>, usize), BenchError> {
    let token_slots = sources.len().max(1) * token_stride;
    let mut kinds = vec![0u32; token_slots];
    let mut starts = vec![0u32; token_slots];
    let mut lens = vec![0u32; token_slots];
    let mut counts = vec![0u32; sources.len().max(1)];
    let mut total_tokens = 0usize;

    for (source_idx, source) in sources.iter().enumerate() {
        let tokens = lex_cpu(source).map_err(|offset| {
            BenchError::ExecutionFailed(format!(
                "Rust lexer batch CPU baseline rejected source {source_idx} at byte {offset}"
            ))
        })?;
        if tokens.len() > token_stride {
            return Err(BenchError::ExecutionFailed(format!(
                "Rust lexer batch source {source_idx} emitted {} tokens for stride {token_stride}",
                tokens.len()
            )));
        }
        counts[source_idx] = u32::try_from(tokens.len()).map_err(|_| {
            BenchError::ExecutionFailed(format!(
                "Rust lexer batch source {source_idx} token count exceeds u32"
            ))
        })?;
        let base = source_idx * token_stride;
        for (token_idx, token) in tokens.iter().enumerate() {
            let out_idx = base + token_idx;
            kinds[out_idx] = u32::from(token.kind);
            starts[out_idx] = token.start;
            lens[out_idx] = u32::from(token.len);
        }
        total_tokens += tokens.len();
    }

    Ok((
        lex_baseline_columns(&kinds, &starts, &lens, &counts),
        total_tokens,
    ))
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::super::lex_columns::decode_u32_words;
    use super::*;
    use vyre::ir::BufferAccess;
    use vyre_frontend_rust::lex::tokens::{EOF, KW_FN};

    #[test]
    fn rust_lexer_batch_baseline_pads_each_source_window_to_program_shape() {
        let sources = rust_lexer_batch_sources();
        let layout = RustLexerBatchLayout::from_sources(&sources).expect("layout builds");
        let (outputs, token_count) =
            rust_lexer_batch_baseline_outputs(&sources, layout.token_stride)
                .expect("benchmark sources lex");
        let token_slots = sources.len() * layout.token_stride;

        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].len(), token_slots * std::mem::size_of::<u32>());
        assert_eq!(outputs[1].len(), token_slots * std::mem::size_of::<u32>());
        assert_eq!(outputs[2].len(), token_slots * std::mem::size_of::<u32>());
        assert_eq!(outputs[3].len(), sources.len() * std::mem::size_of::<u32>());
        assert!(token_count > sources.len());

        let kinds = decode_u32_words(&outputs[0]);
        let counts = decode_u32_words(&outputs[3]);
        assert_eq!(kinds[0], u32::from(KW_FN));
        for (source_idx, count) in counts.iter().copied().enumerate() {
            assert!(count as usize <= layout.token_stride);
            let last_idx = source_idx * layout.token_stride + count as usize - 1;
            assert_eq!(kinds[last_idx], u32::from(EOF));
        }
    }

    #[test]
    fn rust_lexer_batch_program_declares_packed_inputs_and_four_live_out_columns() {
        let sources = rust_lexer_batch_sources();
        let layout = RustLexerBatchLayout::from_sources(&sources).expect("layout builds");
        let program = rust_lexer_batch(
            "haystack",
            "source_offsets",
            "source_lens",
            "out_tok_types",
            "out_tok_starts",
            "out_tok_lens",
            "out_counts",
            layout.packed_source.len() as u32,
            sources.len() as u32,
            layout.token_stride as u32,
        );
        let buffers = program.buffers();
        assert_eq!(buffers.len(), 7);
        assert_eq!(buffers[0].access(), BufferAccess::ReadOnly);
        assert_eq!(buffers[1].access(), BufferAccess::ReadOnly);
        assert_eq!(buffers[2].access(), BufferAccess::ReadOnly);
        assert!(
            buffers[3..]
                .iter()
                .all(|buffer| buffer.access() == BufferAccess::ReadWrite),
            "token columns and per-source counts must be live-outs"
        );
    }

    #[test]
    fn rust_lexer_batch_sources_stay_inside_cpu_subset() {
        let sources = rust_lexer_batch_sources();
        for (source_idx, source) in sources.iter().enumerate().take(32) {
            let tokens = lex_cpu(source).unwrap_or_else(|offset| {
                panic!("benchmark source {source_idx} must lex, rejected at byte {offset}")
            });
            assert_eq!(tokens.last().map(|token| token.kind), Some(EOF));
        }
    }
}
