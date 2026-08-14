use super::lex_columns::{
    lex_baseline_columns, lex_columns_bytes_touched, lex_columns_run, lex_columns_sample,
    u32s_to_bytes, LexColumns, LexColumnsContract, LexSample, LEX_SUITES,
};
use super::rust_source_words;
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, DeterminismClass, WorkloadClass,
};
use crate::api::metric::MetricPoint;
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use vyre_foundation::ir::Program;
use vyre_frontend_rust::lex::lexer::cpu_lexer::lex as lex_cpu;
use vyre_frontend_rust::lex::lexer::plan::rust_lexer;

const RUST_LEXER_REPEATS: usize = 32;

const CONTRACT: LexColumnsContract = LexColumnsContract {
    plan: "Rust lexer",
    columns: "[types, starts, lens, count]",
};

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "frontend.rust.lexer.ir_execute",
    name: "Rust GPU Lexer IR Execute",
    summary: "Rust nano-subset source tokenized by the Vyre IR lexer on GPU with exact CPU lexer column parity",
    tags: &[
        "frontend-rust",
        "gpu-lexer",
        "lexer",
        "tokenization",
        "ir-lexer",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    determinism: DeterminismClass::Deterministic,
    owner_crate: "vyre-frontend-rust",
    suites: LEX_SUITES,
    needs_gpu: true,
    needs_network: false,
    min_vram_bytes: None,
    min_input_bytes: Some((RUST_LEXER_REPEATS * 512) as u64),
    feature_set: &["rust-frontend", "gpu-lexer", "ir-lexer"],
    contract: None,
};

static OPS: CaseOps<LexColumns> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: lex_program,
    fingerprint: None,
    bytes_touched: lex_columns_bytes_touched,
};

static CASE: HarnessCase<LexColumns> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn lex_program(prepared: &LexColumns) -> Option<&Program> {
    Some(&prepared.program)
}

fn build_case(_ctx: &mut BenchContext) -> Result<LexColumns, BenchError> {
    let source_bytes = rust_lexer_source();
    let haystack_len = u32::try_from(source_bytes.len()).map_err(|_| {
        BenchError::ExecutionFailed(
            "Rust lexer benchmark source exceeds u32-addressable plan limit".to_string(),
        )
    })?;
    let program = rust_lexer(
        "haystack",
        "out_tok_types",
        "out_tok_starts",
        "out_tok_lens",
        "out_counts",
        haystack_len,
    );
    let inputs = rust_lexer_inputs(&source_bytes);

    let baseline_start = std::time::Instant::now();
    let (baseline_outputs, token_count) = rust_lexer_baseline_outputs(&source_bytes)?;
    let baseline_wall_ns = baseline_start.elapsed().as_nanos() as u64;

    Ok(LexColumns {
        program,
        inputs,
        source_bytes: source_bytes.len() as u64,
        baseline_outputs,
        baseline_wall_ns,
        token_count,
    })
}

fn measure(ctx: &mut BenchContext, prepared: &mut LexColumns) -> Result<BenchRun, BenchError> {
    let sample = lex_columns_sample(ctx, prepared, None, CONTRACT)?;
    let custom = lexer_metric_points(prepared, &sample);
    Ok(lex_columns_run(ctx, prepared, sample, custom))
}

fn lexer_metric_points(prepared: &LexColumns, sample: &LexSample) -> Vec<MetricPoint> {
    vec![
        MetricPoint {
            name: "rust_frontend_gpu_lexer_speedup_x1000".to_string(),
            value: super::speedup_x1000(prepared.baseline_wall_ns, sample.wall_ns),
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_tokens".to_string(),
            value: prepared.token_count as u64,
        },
        MetricPoint {
            name: "rust_frontend_gpu_lexer_source_bytes".to_string(),
            value: prepared.source_bytes,
        },
    ]
}

fn rust_lexer_source() -> Vec<u8> {
    let mut source = String::new();
    for idx in 0..RUST_LEXER_REPEATS {
        source.push_str(&format!(
            "fn stress_{idx}(n: i32, flag: bool) -> i32 {{
    let mut acc: i32 = {idx};
    // branchy token stream with comments, booleans, ranges, and compound ops
    for i in -3..n {{
        if i <= 0 && flag != false {{
            acc += i * 2;
        }} else {{
            acc -= i % 3;
        }};
    }}
    return acc;
}}
"
        ));
    }
    source.into_bytes()
}

fn rust_lexer_inputs(source: &[u8]) -> Vec<Vec<u8>> {
    let token_capacity = token_capacity(source);
    let zero_tokens = vec![0u8; token_capacity * std::mem::size_of::<u32>()];
    vec![
        u32s_to_bytes(&rust_source_words(source)),
        zero_tokens.clone(),
        zero_tokens.clone(),
        zero_tokens,
        u32s_to_bytes(&[0]),
    ]
}

fn rust_lexer_baseline_outputs(source: &[u8]) -> Result<(Vec<Vec<u8>>, usize), BenchError> {
    let tokens = lex_cpu(source).map_err(|offset| {
        BenchError::ExecutionFailed(format!(
            "Rust lexer benchmark CPU baseline rejected source at byte {offset}"
        ))
    })?;
    let token_capacity = token_capacity(source);
    if tokens.len() > token_capacity {
        return Err(BenchError::ExecutionFailed(format!(
            "Rust lexer baseline emitted {} tokens for capacity {token_capacity}",
            tokens.len()
        )));
    }

    let mut kinds = vec![0u32; token_capacity];
    let mut starts = vec![0u32; token_capacity];
    let mut lens = vec![0u32; token_capacity];
    for (idx, token) in tokens.iter().enumerate() {
        kinds[idx] = u32::from(token.kind);
        starts[idx] = token.start;
        lens[idx] = u32::from(token.len);
    }
    let count = u32::try_from(tokens.len()).map_err(|_| {
        BenchError::ExecutionFailed(
            "Rust lexer benchmark token count exceeds u32 output count".to_string(),
        )
    })?;

    Ok((
        lex_baseline_columns(&kinds, &starts, &lens, &[count]),
        tokens.len(),
    ))
}

fn token_capacity(source: &[u8]) -> usize {
    source.len().saturating_add(1).max(1)
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::super::lex_columns::decode_u32_words;
    use super::*;
    use vyre::ir::BufferAccess;
    use vyre_frontend_rust::lex::lexer::cpu_lexer::Token;
    use vyre_frontend_rust::lex::tokens::{EOF, KW_FN};

    #[test]
    fn rust_lexer_baseline_pads_live_out_columns_to_program_shape() {
        let source = rust_lexer_source();
        let (outputs, token_count) =
            rust_lexer_baseline_outputs(&source).expect("benchmark source lexes");
        let token_capacity = token_capacity(&source);

        assert_eq!(outputs.len(), 4);
        assert_eq!(
            outputs[0].len(),
            token_capacity * std::mem::size_of::<u32>()
        );
        assert_eq!(
            outputs[1].len(),
            token_capacity * std::mem::size_of::<u32>()
        );
        assert_eq!(
            outputs[2].len(),
            token_capacity * std::mem::size_of::<u32>()
        );
        assert_eq!(outputs[3].len(), std::mem::size_of::<u32>());
        assert!(token_count > RUST_LEXER_REPEATS);

        let kinds = decode_u32_words(&outputs[0]);
        let count = decode_u32_words(&outputs[3])[0] as usize;
        assert_eq!(count, token_count);
        assert_eq!(kinds[0], u32::from(KW_FN));
        assert_eq!(kinds[count - 1], u32::from(EOF));
    }

    #[test]
    fn rust_lexer_program_declares_haystack_plus_four_live_out_columns() {
        let source = rust_lexer_source();
        let program = rust_lexer(
            "haystack",
            "out_tok_types",
            "out_tok_starts",
            "out_tok_lens",
            "out_counts",
            source.len() as u32,
        );
        let buffers = program.buffers();
        assert_eq!(buffers.len(), 5);
        assert_eq!(buffers[0].access(), BufferAccess::ReadOnly);
        assert!(
            buffers[1..]
                .iter()
                .all(|buffer| buffer.access() == BufferAccess::ReadWrite),
            "token columns must be read-write live-outs so CUDA returns all lexer columns"
        );
    }

    #[test]
    fn rust_lexer_source_stays_inside_cpu_subset() {
        let source = rust_lexer_source();
        let tokens = lex_cpu(&source).expect("benchmark source must stay in lexer subset");
        assert_eq!(tokens.last().map(|token: &Token| token.kind), Some(EOF));
    }
}
