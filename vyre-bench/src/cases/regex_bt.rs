//! `regex.backtracking.adversarial`  -  Catastrophic backtracking regex.
//!
//! Pattern `(a+)+b` against hostile inputs of repeated 'a's. CPU regex engines
//! with backtracking go superlinear (O(2^n)). GPU parallelism should dominate
//! by evaluating all NFA states simultaneously.

use crate::api::case::{BenchCase, BenchContext, BenchError};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use crate::cases::reference_sample::{
    measure_against_reference, referenced_bytes_touched, referenced_program, HostReferencePayload,
    HostReferenced,
};
use pcre2::bytes::{Regex, RegexBuilder};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Input size: 1024 bytes of 'a' per instance, 4096 instances.
/// Each GPU thread evaluates whether the pattern matches its input slice.
const INPUT_LEN: u32 = 256; // words (1024 bytes)
const INSTANCE_COUNT: u32 = 4096;
const TOTAL_WORDS: u32 = INPUT_LEN * INSTANCE_COUNT;

struct RegexBacktrackingPrepared {
    dispatch: HostReferencePayload,
    regex: Regex,
    corpus: Vec<u8>,
}

impl HostReferenced for RegexBacktrackingPrepared {
    fn dispatch(&self) -> &HostReferencePayload {
        &self.dispatch
    }

    fn reference(&self) -> Result<Vec<u8>, BenchError> {
        cpu_pcre2_scan(
            &self.regex,
            &self.corpus,
            INSTANCE_COUNT as usize,
            INPUT_LEN as usize * 4,
        )
    }

    /// PCRE2 scans the corpus itself, which is the whole uploaded input set.
    fn reference_input_bytes(&self) -> u64 {
        self.corpus.len() as u64
    }
}

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "regex.backtracking.adversarial",
    "Regex Backtracking Adversarial",
    "Catastrophic backtracking: (a+)+b pattern on hostile 'aaaa...' input",
    &["honest", "regex", "adversarial"],
    (TOTAL_WORDS as u64 + INSTANCE_COUNT as u64) * 4,
    Some(ContractDescription {
        primitive: "Catastrophic backtracking regex",
        baseline_crate: "pcre2",
        baseline_name: "PCRE2 10.44 (backtracking engine)",
        min_speedup_x: 3.0,
    }),
);

static OPS: CaseOps<RegexBacktrackingPrepared> = CaseOps {
    build: prepare_regex_backtracking,
    measure: measure_against_reference::<RegexBacktrackingPrepared>,
    verify: verify_exact,
    program: referenced_program::<RegexBacktrackingPrepared>,
    fingerprint: None,
    bytes_touched: referenced_bytes_touched::<RegexBacktrackingPrepared>,
};

static CASE: HarnessCase<RegexBacktrackingPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// NFA-style parallel state evaluation of `(a+)+b`.
///
/// The NFA has three states: start, inside the `(a+)` group, and accept. Each
/// thread walks its own input slice byte by byte. The hostile corpus is all
/// `'a'` with no `'b'`, so every result is zero: the work is the scan, not the
/// answer. The device does that scan honestly, and its advantage is parallelism
/// across instances rather than an algorithmic shortcut.
fn prepare_regex_backtracking(
    ctx: &mut BenchContext,
) -> Result<RegexBacktrackingPrepared, BenchError> {
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(TOTAL_WORDS),
            BufferDecl::output("results", 1, DataType::U32).with_count(INSTANCE_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(INSTANCE_COUNT)),
                vec![
                    Node::let_bind("base", Expr::mul(Expr::var("tid"), Expr::u32(INPUT_LEN))),
                    // NFA state: 0=start, 1=in_a_group, 2=matched
                    Node::let_bind("state", Expr::u32(0)),
                    Node::let_bind("match_count", Expr::u32(0)),
                    // Scan each word
                    Node::Loop {
                        var: "i".into(),
                        from: Expr::u32(0),
                        to: Expr::u32(INPUT_LEN),
                        body: vec![
                            Node::let_bind(
                                "word",
                                Expr::load("input", Expr::add(Expr::var("base"), Expr::var("i"))),
                            ),
                            // Extract each byte from the word and process
                            // Byte 0
                            Node::let_bind("b0", Expr::bitand(Expr::var("word"), Expr::u32(0xFF))),
                            // 'a' = 0x61, 'b' = 0x62
                            Node::if_then(
                                Expr::eq(Expr::var("b0"), Expr::u32(0x61)),
                                vec![Node::assign("state", Expr::u32(1))],
                            ),
                            Node::if_then(
                                Expr::and(
                                    Expr::eq(Expr::var("b0"), Expr::u32(0x62)),
                                    Expr::eq(Expr::var("state"), Expr::u32(1)),
                                ),
                                vec![
                                    Node::assign(
                                        "match_count",
                                        Expr::add(Expr::var("match_count"), Expr::u32(1)),
                                    ),
                                    Node::assign("state", Expr::u32(0)),
                                ],
                            ),
                            // Byte 1
                            Node::let_bind(
                                "b1",
                                Expr::bitand(
                                    Expr::shr(Expr::var("word"), Expr::u32(8)),
                                    Expr::u32(0xFF),
                                ),
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("b1"), Expr::u32(0x61)),
                                vec![Node::assign("state", Expr::u32(1))],
                            ),
                            Node::if_then(
                                Expr::and(
                                    Expr::eq(Expr::var("b1"), Expr::u32(0x62)),
                                    Expr::eq(Expr::var("state"), Expr::u32(1)),
                                ),
                                vec![
                                    Node::assign(
                                        "match_count",
                                        Expr::add(Expr::var("match_count"), Expr::u32(1)),
                                    ),
                                    Node::assign("state", Expr::u32(0)),
                                ],
                            ),
                            // Byte 2
                            Node::let_bind(
                                "b2",
                                Expr::bitand(
                                    Expr::shr(Expr::var("word"), Expr::u32(16)),
                                    Expr::u32(0xFF),
                                ),
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("b2"), Expr::u32(0x61)),
                                vec![Node::assign("state", Expr::u32(1))],
                            ),
                            Node::if_then(
                                Expr::and(
                                    Expr::eq(Expr::var("b2"), Expr::u32(0x62)),
                                    Expr::eq(Expr::var("state"), Expr::u32(1)),
                                ),
                                vec![
                                    Node::assign(
                                        "match_count",
                                        Expr::add(Expr::var("match_count"), Expr::u32(1)),
                                    ),
                                    Node::assign("state", Expr::u32(0)),
                                ],
                            ),
                            // Byte 3
                            Node::let_bind("b3", Expr::shr(Expr::var("word"), Expr::u32(24))),
                            Node::if_then(
                                Expr::eq(Expr::var("b3"), Expr::u32(0x61)),
                                vec![Node::assign("state", Expr::u32(1))],
                            ),
                            Node::if_then(
                                Expr::and(
                                    Expr::eq(Expr::var("b3"), Expr::u32(0x62)),
                                    Expr::eq(Expr::var("state"), Expr::u32(1)),
                                ),
                                vec![
                                    Node::assign(
                                        "match_count",
                                        Expr::add(Expr::var("match_count"), Expr::u32(1)),
                                    ),
                                    Node::assign("state", Expr::u32(0)),
                                ],
                            ),
                        ],
                    },
                    Node::store("results", Expr::var("tid"), Expr::var("match_count")),
                ],
            ),
        ],
    );
    let regex = RegexBuilder::new()
        .jit(true)
        .build(r"(a+)+b")
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let corpus = vec![0x61u8; TOTAL_WORDS as usize * 4];
    let inputs = vec![corpus.clone()];

    Ok(RegexBacktrackingPrepared {
        dispatch: HostReferencePayload::program_ordered_resident(
            ctx,
            prog,
            inputs,
            "regex backtracking bench",
        )?,
        regex,
        corpus,
    })
}

/// CPU PCRE2 scan  -  the advertised backtracking baseline, not a custom NFA.
fn cpu_pcre2_scan(
    regex: &Regex,
    input: &[u8],
    instances: usize,
    bytes_per: usize,
) -> Result<Vec<u8>, BenchError> {
    let mut results = vec![0u32; instances];
    for instance in 0..instances {
        let base = instance * bytes_per;
        let haystack = &input[base..base + bytes_per];
        let mut matches = regex
            .find_iter(haystack)
            .map(|item| item.map(|_| 1u32))
            .try_fold(0u32, |count, item| {
                item.map(|matched| count + matched)
                    .map_err(|error| BenchError::ExecutionFailed(error.to_string()))
            })?;
        if matches > 0 {
            matches = 1;
        }
        results[instance] = matches;
    }
    Ok(vyre_primitives::wire::pack_u32_slice(&results))
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
