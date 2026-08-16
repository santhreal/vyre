//! Turn a logit row into one decoded token.
//!
//! Sampling is three stages and the middle one already exists. The logits are
//! rescaled per element, the rescaled row is reduced to its `k` largest
//! probabilities, and one of those candidates is drawn against a uniform
//! sample. The middle stage is the mixture-of-experts gate: softmax over a row
//! followed by a top-k selection with normalized weights is the same reduction
//! whether the row is expert scores or vocabulary logits, so this composes
//! `softmax_top_k` rather than restating it against a second scratch layout.
//!
//! The two rescalings are one pass. A repetition penalty and a temperature are
//! both per-logit transforms with no reduction between them; splitting them
//! into two operations would cost a vocabulary-sized intermediate and a
//! grid-wide barrier to hand it over, and buy no reuse, because nothing reads a
//! penalized-but-unscaled logit row.
//!
//! The selection and the draw are serial: the top-k insertion sort is a
//! read-modify-write on `k` slots, and the cumulative mass that top-p needs is
//! a running sum over those slots in descending order. Fusing them under the
//! elementwise stage widens their geometry, so both arms are attributed through
//! [`attribute_serial_child`], which names the invocation they run on.

use thiserror::Error;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::builder::build_indexed_map;
use crate::nn::moe::softmax_top_k::{softmax_top_k, OP_ID as SOFTMAX_TOP_K_OP_ID};
use crate::plumbing::program::attribution::{attribute_child, attribute_serial_child};
use crate::plumbing::program::outputs::demote_intermediate_outputs;

/// Canonical op id of the per-logit rescaling.
pub const LOGIT_ADJUST_OP_ID: &str = "vyre-libs::llm::logit_adjust";
/// Canonical op id of the top-p draw over an already selected candidate set.
pub const NUCLEUS_SELECT_OP_ID: &str = "vyre-libs::llm::nucleus_select";
/// Canonical op id of the composed sampler.
pub const SAMPLE_TOKEN_OP_ID: &str = "vyre-libs::llm::sample_token";

/// Rescaled logit row handed from the elementwise stage to the selection.
const ADJUSTED_BUFFER: &str = "__vyre_llm_sampling_adjusted";
/// Candidate token ids the selection keeps for the draw.
const CANDIDATES_BUFFER: &str = "__vyre_llm_sampling_candidates";
/// Normalized candidate weights the draw accumulates mass over.
const WEIGHTS_BUFFER: &str = "__vyre_llm_sampling_weights";

/// Rejected sampling parameters.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SamplingError {
    /// The vocabulary is empty, so there is nothing to draw from.
    #[error("sampling requires a nonempty vocabulary")]
    EmptyVocabulary,
    /// No candidate is kept, so the draw has no support.
    #[error("sampling requires at least one candidate")]
    NoCandidates,
    /// More candidates were asked for than the vocabulary holds.
    #[error("sampling asked for {candidates} candidates from a vocabulary of {vocabulary}")]
    CandidatesExceedVocabulary {
        /// Requested candidate count.
        candidates: u32,
        /// Vocabulary size the candidates are drawn from.
        vocabulary: u32,
    },
    /// Temperature is not a positive finite number.
    #[error("sampling temperature must be positive and finite; got {temperature}")]
    Temperature {
        /// Rejected temperature.
        temperature: f32,
    },
    /// Repetition penalty is not a positive finite number.
    #[error("repetition penalty must be positive and finite; got {penalty}")]
    Penalty {
        /// Rejected penalty.
        penalty: f32,
    },
    /// Top-p is outside the mass a probability distribution can supply.
    #[error("top-p must lie in (0, 1]; got {top_p}")]
    TopP {
        /// Rejected nucleus mass.
        top_p: f32,
    },
    /// The stages could not be fused into one dispatch.
    #[error("sampler stages did not fuse: {reason}")]
    Fusion {
        /// Fusion failure, as reported by the fuser.
        reason: String,
    },
}

/// One decode step's sampling parameters and the buffers it reads and writes.
///
/// Nine values describe the draw, and a positional call taking all nine reads
/// as a row of bare literals at every call site. The struct is what makes
/// `temperature` and `top_p` distinguishable at the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSampler<'a> {
    /// Raw logit row, one element per vocabulary entry.
    pub logits: &'a str,
    /// Occurrence count per vocabulary entry in the context so far.
    pub counts: &'a str,
    /// Single-element uniform sample in `[0, 1)`.
    pub uniform: &'a str,
    /// Single-element output holding the drawn vocabulary index.
    pub token: &'a str,
    /// Vocabulary size.
    pub vocabulary: u32,
    /// Candidates the nucleus is drawn from.
    pub candidates: u32,
    /// Softmax temperature. Below one sharpens, above one flattens.
    pub temperature: f32,
    /// Repetition penalty. One leaves the logits alone.
    pub repetition_penalty: f32,
    /// Nucleus mass. One draws from the whole candidate set.
    pub top_p: f32,
}

impl TokenSampler<'_> {
    /// Build the fused sampler.
    ///
    /// # Errors
    ///
    /// Returns [`SamplingError`] when a parameter cannot describe a draw, or
    /// when the stages cannot share one dispatch.
    pub fn program(&self) -> Result<Program, SamplingError> {
        self.check()?;
        let adjusted = ADJUSTED_BUFFER;
        let selected = CANDIDATES_BUFFER;
        let weights = WEIGHTS_BUFFER;

        let adjust = attribute_child(
            SAMPLE_TOKEN_OP_ID,
            LOGIT_ADJUST_OP_ID,
            logit_adjust(
                self.logits,
                self.counts,
                adjusted,
                self.vocabulary,
                self.temperature,
                self.repetition_penalty,
            ),
        );
        let select = attribute_serial_child(
            SAMPLE_TOKEN_OP_ID,
            SOFTMAX_TOP_K_OP_ID,
            softmax_top_k(
                adjusted,
                selected,
                weights,
                self.vocabulary,
                self.candidates,
            ),
        );
        let draw = attribute_serial_child(
            SAMPLE_TOKEN_OP_ID,
            NUCLEUS_SELECT_OP_ID,
            nucleus_select(
                selected,
                weights,
                self.uniform,
                self.token,
                self.candidates,
                self.top_p,
            ),
        );

        let fused = fuse_programs(&[adjust, select, draw]).map_err(|error| SamplingError::Fusion {
            reason: error.to_string(),
        })?;
        Ok(demote_intermediate_outputs(fused, self.token))
    }

    fn check(&self) -> Result<(), SamplingError> {
        if self.vocabulary == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        if self.candidates == 0 {
            return Err(SamplingError::NoCandidates);
        }
        if self.candidates > self.vocabulary {
            return Err(SamplingError::CandidatesExceedVocabulary {
                candidates: self.candidates,
                vocabulary: self.vocabulary,
            });
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(SamplingError::Temperature {
                temperature: self.temperature,
            });
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return Err(SamplingError::Penalty {
                penalty: self.repetition_penalty,
            });
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(SamplingError::TopP { top_p: self.top_p });
        }
        Ok(())
    }
}

/// Build `adjusted[i]`, the logit row a sampler draws from.
///
/// A token already in the context is pushed toward zero by `penalty`: a
/// positive logit is divided and a negative logit is multiplied, so the penalty
/// always reduces the score rather than flipping its sign the way a bare
/// division does below zero. The whole row is then divided by `temperature`.
///
/// `counts` holds an occurrence count per vocabulary entry; a zero count leaves
/// the logit untouched, so a caller that wants no penalty passes a penalty of
/// one and the arithmetic is the identity.
#[must_use]
pub fn logit_adjust(
    logits: &str,
    counts: &str,
    adjusted: &str,
    vocabulary: u32,
    temperature: f32,
    penalty: f32,
) -> Program {
    let buffers = vec![
        BufferDecl::storage(logits, 0, BufferAccess::ReadOnly, DataType::F32).with_count(vocabulary),
        BufferDecl::storage(counts, 1, BufferAccess::ReadOnly, DataType::U32).with_count(vocabulary),
        BufferDecl::output(adjusted, 2, DataType::F32).with_count(vocabulary),
    ];

    build_indexed_map(
        LOGIT_ADJUST_OP_ID,
        buffers,
        adjusted,
        vocabulary,
        [64, 1, 1],
        |i| {
            let logit = Expr::load(logits, i.clone());
            let penalized = Expr::select(
                Expr::gt(logit.clone(), Expr::f32(0.0)),
                Expr::div(logit.clone(), Expr::f32(penalty)),
                Expr::mul(logit.clone(), Expr::f32(penalty)),
            );
            let seen = Expr::gt(Expr::load(counts, i.clone()), Expr::u32(0));
            let value = Expr::div(
                Expr::select(seen, penalized, logit),
                Expr::f32(temperature),
            );
            (i, value)
        },
    )
}

/// Build the top-p draw over a descending candidate set.
///
/// `weights` are the candidate probabilities in descending order and
/// `selected` their vocabulary indices, which is exactly what `softmax_top_k`
/// produces. The nucleus is the shortest prefix whose mass reaches `top_p`, or
/// the whole candidate set when it never does, and the draw walks that prefix
/// against `uniform[0]` scaled by the prefix mass. Scaling the sample by the
/// mass is what renormalizes the nucleus without a second pass over it.
///
/// An empty candidate set has no token to draw, so it builds a trap program
/// rather than a load at index `candidates - 1`.
#[must_use]
pub fn nucleus_select(
    selected: &str,
    weights: &str,
    uniform: &str,
    token: &str,
    candidates: u32,
    top_p: f32,
) -> Program {
    if candidates == 0 {
        return vyre_foundation::composition::trap_program(
            NUCLEUS_SELECT_OP_ID,
            Some((token, DataType::U32)),
            "Fix: nucleus_select requires candidates > 0; an empty candidate set has no token to \
             draw."
                .to_string(),
        );
    }
    let buffers = vec![
        BufferDecl::storage(selected, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(candidates),
        BufferDecl::storage(weights, 1, BufferAccess::ReadOnly, DataType::F32)
            .with_count(candidates),
        BufferDecl::storage(uniform, 2, BufferAccess::ReadOnly, DataType::F32).with_count(1),
        BufferDecl::output(token, 3, DataType::U32).with_count(1),
    ];

    Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![vyre_foundation::composition::wrap_anonymous_region(
            NUCLEUS_SELECT_OP_ID,
            nucleus_select_body(selected, weights, uniform, token, candidates, top_p),
        )],
    )
}

fn nucleus_select_body(
    selected: &str,
    weights: &str,
    uniform: &str,
    token: &str,
    candidates: u32,
    top_p: f32,
) -> Vec<Node> {
    let kept = Expr::var("kept");
    let mass = Expr::var("mass");

    // Walk the candidates in descending order until the accumulated mass
    // reaches top_p. `kept` stays zero until the crossing, which both stops the
    // accumulation and records the prefix length.
    let mut body = vec![
        Node::let_bind("kept", Expr::u32(0)),
        Node::let_bind("mass", Expr::f32(0.0)),
        Node::loop_for(
            "j",
            Expr::u32(0),
            Expr::u32(candidates),
            vec![Node::if_then(
                Expr::eq(kept.clone(), Expr::u32(0)),
                vec![
                    Node::assign(
                        "mass",
                        Expr::add(mass.clone(), Expr::load(weights, Expr::var("j"))),
                    ),
                    Node::if_then(
                        Expr::ge(mass.clone(), Expr::f32(top_p)),
                        vec![Node::assign(
                            "kept",
                            Expr::add(Expr::var("j"), Expr::u32(1)),
                        )],
                    ),
                ],
            )],
        ),
        // The candidate set never reached top_p, so the nucleus is all of it
        // and `mass` already holds its total.
        Node::if_then(
            Expr::eq(kept.clone(), Expr::u32(0)),
            vec![Node::assign("kept", Expr::u32(candidates))],
        ),
    ];

    // The sample is scaled by the nucleus mass instead of the candidates being
    // rescaled to sum to one.
    body.push(Node::let_bind(
        "target",
        Expr::mul(Expr::load(uniform, Expr::u32(0)), mass.clone()),
    ));
    // The last kept candidate is the answer when the running sum never reaches
    // the target, which rounding at the end of the prefix can produce.
    body.push(Node::let_bind(
        "chosen",
        Expr::load(selected, Expr::sub(kept.clone(), Expr::u32(1))),
    ));
    body.push(Node::let_bind("found", Expr::u32(0)));
    body.push(Node::let_bind("drawn", Expr::f32(0.0)));
    body.push(Node::loop_for(
        "d",
        Expr::u32(0),
        Expr::u32(candidates),
        vec![Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("found"), Expr::u32(0)),
                Expr::lt(Expr::var("d"), kept.clone()),
            ),
            vec![
                Node::assign(
                    "drawn",
                    Expr::add(Expr::var("drawn"), Expr::load(weights, Expr::var("d"))),
                ),
                Node::if_then(
                    Expr::ge(Expr::var("drawn"), Expr::var("target")),
                    vec![
                        Node::assign("chosen", Expr::load(selected, Expr::var("d"))),
                        Node::assign("found", Expr::u32(1)),
                    ],
                ),
            ],
        )],
    ));
    body.push(Node::store(token, Expr::u32(0), Expr::var("chosen")));
    body
}

/// The vocabulary every registered fixture in this module draws from.
const FIXTURE_VOCABULARY: u32 = 4;
/// The candidate count every registered fixture in this module keeps.
const FIXTURE_CANDIDATES: u32 = 2;
/// Raw logits behind every registered fixture in this module.
const FIXTURE_LOGITS: [f32; FIXTURE_VOCABULARY as usize] = [1.0, 2.0, 3.0, 4.0];
/// Occurrence counts behind every registered fixture in this module. Entry one
/// is already in the context, so the penalty applies to exactly one logit and a
/// build that dropped the count lookup cannot reproduce the row.
const FIXTURE_COUNTS: [u32; FIXTURE_VOCABULARY as usize] = [0, 1, 0, 0];
/// Fixture temperature. Not one, so a build that dropped the division cannot
/// reproduce the row.
const FIXTURE_TEMPERATURE: f32 = 2.0;
/// Fixture repetition penalty.
const FIXTURE_PENALTY: f32 = 2.0;
/// Fixture nucleus mass. Below the leading candidate plus the second, and above
/// the leading candidate alone, so the cutoff keeps both.
const FIXTURE_TOP_P: f32 = 0.7;
/// Fixture uniform sample. High enough in the nucleus that the draw lands on
/// the second candidate, so a build that always answered with the argmax fails.
const FIXTURE_UNIFORM: f32 = 0.9;

fn fixture_f32(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

fn fixture_u32(values: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(values)
}

/// The adjusted logit row, recomputed in the same order the body evaluates it.
fn fixture_adjusted() -> Vec<f32> {
    FIXTURE_LOGITS
        .iter()
        .zip(FIXTURE_COUNTS.iter())
        .map(|(logit, count)| {
            let penalized = if *count > 0 {
                if *logit > 0.0 {
                    logit / FIXTURE_PENALTY
                } else {
                    logit * FIXTURE_PENALTY
                }
            } else {
                *logit
            };
            penalized / FIXTURE_TEMPERATURE
        })
        .collect()
}

/// The candidate set the selection produces: exponentials of the adjusted row
/// relative to its maximum, the two largest in descending order, and their
/// share of the full softmax denominator.
struct FixtureSelection {
    indices: Vec<u32>,
    exponentials: Vec<f32>,
    weights: Vec<f32>,
}

fn fixture_selection() -> FixtureSelection {
    let adjusted = fixture_adjusted();
    let max = adjusted
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |best, value| {
            if value > best {
                value
            } else {
                best
            }
        });
    let exponentials: Vec<f32> = adjusted.iter().map(|value| (value - max).exp()).collect();
    let sum = exponentials
        .iter()
        .fold(0.0f32, |total, value| total + value);

    // The body's insertion sort keeps the first index on a tie, so ordering by
    // a strict comparison over ascending indices reproduces it.
    let mut order: Vec<u32> = (0..FIXTURE_VOCABULARY).collect();
    order.sort_by(|left, right| {
        let left_value = exponentials[*left as usize];
        let right_value = exponentials[*right as usize];
        right_value
            .partial_cmp(&left_value)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    let indices: Vec<u32> = order
        .into_iter()
        .take(FIXTURE_CANDIDATES as usize)
        .collect();
    let kept: Vec<f32> = indices
        .iter()
        .map(|index| exponentials[*index as usize])
        .collect();
    let weights: Vec<f32> = kept.iter().map(|value| value / sum).collect();
    FixtureSelection {
        indices,
        exponentials: kept,
        weights,
    }
}

/// The token the fixture draw lands on.
fn fixture_token(selection: &FixtureSelection) -> u32 {
    let mut kept = 0usize;
    let mut mass = 0.0f32;
    for (position, weight) in selection.weights.iter().enumerate() {
        if kept == 0 {
            mass += *weight;
            if mass >= FIXTURE_TOP_P {
                kept = position + 1;
            }
        }
    }
    if kept == 0 {
        kept = selection.weights.len();
    }
    let target = FIXTURE_UNIFORM * mass;
    let mut drawn = 0.0f32;
    let mut chosen = selection.indices[kept - 1];
    for position in 0..kept {
        drawn += selection.weights[position];
        if drawn >= target {
            chosen = selection.indices[position];
            break;
        }
    }
    chosen
}

fn fixture_sampler() -> TokenSampler<'static> {
    TokenSampler {
        logits: "logits",
        counts: "counts",
        uniform: "uniform",
        token: "token",
        vocabulary: FIXTURE_VOCABULARY,
        candidates: FIXTURE_CANDIDATES,
        temperature: FIXTURE_TEMPERATURE,
        repetition_penalty: FIXTURE_PENALTY,
        top_p: FIXTURE_TOP_P,
    }
}

fn logit_adjust_fixture_program() -> Program {
    logit_adjust(
        "logits",
        "counts",
        "adjusted",
        FIXTURE_VOCABULARY,
        FIXTURE_TEMPERATURE,
        FIXTURE_PENALTY,
    )
}

fn nucleus_select_fixture_program() -> Program {
    nucleus_select(
        "selected",
        "weights",
        "uniform",
        "token",
        FIXTURE_CANDIDATES,
        FIXTURE_TOP_P,
    )
}

fn sample_token_fixture_program() -> Program {
    match fixture_sampler().program() {
        Ok(program) => program,
        Err(error) => vyre_foundation::composition::trap_program(
            SAMPLE_TOKEN_OP_ID,
            None,
            format!("Fix: sample_token fixture must build: {error}"),
        ),
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        LOGIT_ADJUST_OP_ID,
        logit_adjust_fixture_program,
        Some(|| vec![vec![
            fixture_f32(&FIXTURE_LOGITS),
            fixture_u32(&FIXTURE_COUNTS),
        ]]),
        Some(|| vec![vec![fixture_f32(&fixture_adjusted())]]),
    )
    .with_category("llm")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        NUCLEUS_SELECT_OP_ID,
        nucleus_select_fixture_program,
        Some(|| {
            let selection = fixture_selection();
            vec![vec![
                fixture_u32(&selection.indices),
                fixture_f32(&selection.weights),
                fixture_f32(&[FIXTURE_UNIFORM]),
            ]]
        }),
        Some(|| {
            let selection = fixture_selection();
            vec![vec![fixture_u32(&[fixture_token(&selection)])]]
        }),
    )
    .with_category("llm")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SAMPLE_TOKEN_OP_ID,
        sample_token_fixture_program,
        Some(|| {
            const ZERO_F32: [f32; FIXTURE_CANDIDATES as usize] = [0.0; FIXTURE_CANDIDATES as usize];
            const ZERO_U32: [u32; FIXTURE_CANDIDATES as usize] = [0; FIXTURE_CANDIDATES as usize];
            vec![vec![
                fixture_f32(&FIXTURE_LOGITS),
                fixture_u32(&FIXTURE_COUNTS),
                fixture_f32(&ZERO_F32),
                fixture_f32(&ZERO_F32),
                fixture_u32(&ZERO_U32),
                fixture_f32(&[FIXTURE_UNIFORM]),
            ]]
        }),
        Some(|| {
            let selection = fixture_selection();
            vec![vec![
                fixture_f32(&fixture_adjusted()),
                fixture_u32(&selection.indices),
                fixture_f32(&selection.weights),
                fixture_f32(&selection.exponentials),
                fixture_u32(&selection.indices),
                fixture_u32(&[fixture_token(&selection)]),
            ]]
        }),
    )
    .with_category("llm")
}
