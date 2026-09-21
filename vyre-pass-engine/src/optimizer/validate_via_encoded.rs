//! GPU-native limit validator over the encoded arena + ProgramGraph.
//!
//! Cheap parallel reductions over the canonical 5-buffer arena to
//! check the foundation validator's static limits without crossing
//! back to the CPU. Currently checks:
//!
//! - **V019**: total IR statement-node count ≤
//!   `DEFAULT_MAX_NODE_COUNT` (100_000).
//! - **V033**: deepest expression nesting ≤
//!   `DEFAULT_MAX_EXPR_DEPTH` (1024).
//!
//! These are the two limits that map directly to existing arena
//! columns (`expr_count`, `depths`, `node_count`)  -  no per-Node walk
//! required. The other validators (typecheck, uniformity, fusion
//! safety, name-shadowing) need contextual data the substrate
//! doesn't yet build into the arena and stay on the CPU side.
//!
//! Output is a 2-word `violations` bitmap:
//!   - `violations[0]` : V033 (expr-depth overflow); `1` = violation
//!   - `violations[1]` : V019 (node-count overflow); `1` = violation
//!
//! The kernel runs as a single dispatch with an internal level-style
//! reduction: each thread processes its share of `depths`, computes
//! a local max via SeqCst-barrier-coordinated workgroup-shared
//! state, and the first thread compares against the limits.

use std::sync::Arc;

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use vyre_libs::dispatch_buffers::{decode_u32_output_exact, u32_slice_to_le_bytes};

use super::encode::{encode_program, EncodeError};
use super::expr_arena::{encode_expr_arena, ExprArenaEncoding};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// Max expression nesting depth this GPU-native pre-filter flags.
///
/// This is a CHEAP PRE-FILTER, not the authoritative gate: it runs over the
/// encoded arena to reject obviously-oversized programs without a CPU walk.
/// It is intentionally MORE PERMISSIVE than foundation's strict V033 ceiling
/// (`DEFAULT_MAX_EXPR_DEPTH = 128`) so the cheap path never *falsely* rejects
///: anything between 128 and this bound is caught by the strict CPU
/// validator in `vyre_foundation::validate`. The CAS-loop reduction below
/// relies on depths being bounded by this value.
pub const DEFAULT_MAX_EXPR_DEPTH: u32 = 1024;
/// Max accepted statement-node count. Derived from the foundation ceiling so
/// the two CANNOT diverge: for node count the encoded validator and the CPU
/// validator must agree exactly, or a fused bundle between the two ceilings
/// passes one and is rejected by the other (the V019 regression that broke a
/// downstream megakernel-bundle scan). Single source of truth lives
/// in `vyre_foundation::validate::depth`.
pub const DEFAULT_MAX_NODE_COUNT: u32 = vyre_foundation::validate::DEFAULT_MAX_NODE_COUNT as u32;

/// Workgroup size for the limit-validator kernel.
const VALIDATOR_WORKGROUP_X: u32 = 256;

/// Index of the V033 (expr-depth) violation bit in the output buffer.
pub const VIOLATION_INDEX_V033: u32 = 0;
/// Index of the V019 (node-count) violation bit in the output buffer.
pub const VIOLATION_INDEX_V019: u32 = 1;

/// Errors surfaced by `gpu_validate_limits`.
#[derive(Debug)]
pub enum ValidateError {
    /// Encoder did not accept the input shape.
    Encode(EncodeError),
    /// Semantic execution or canonical output decoding failed.
    Semantic(SemanticExecutionError),
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_validate encode error: {err:?}"),
            Self::Semantic(err) => write!(f, "gpu_validate semantic execution error: {err}"),
        }
    }
}

impl std::error::Error for ValidateError {}

/// Run the limit checker through semantic execution.
pub fn gpu_validate_limits(
    program: &Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<[bool; 2], ValidateError> {
    let arena = encode_expr_arena(program).map_err(ValidateError::Encode)?;
    let encoded = encode_program(program).map_err(ValidateError::Encode)?;
    gpu_validate_limits_from_encoding(&arena, encoded.node_count, executor, policy)
}

/// Run the limit checker from precomputed encodings.
pub fn gpu_validate_limits_from_encoding(
    arena: &ExprArenaEncoding,
    node_count: u32,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<[bool; 2], ValidateError> {
    if arena.expr_count == 0 && node_count == 0 {
        return Ok([false, false]);
    }
    let depths_bytes = if arena.depths.is_empty() {
        vec![0u8; 4]
    } else {
        u32_slice_to_le_bytes(&arena.depths)
    };
    let limits_bytes = u32_slice_to_le_bytes(&[
        DEFAULT_MAX_EXPR_DEPTH,
        DEFAULT_MAX_NODE_COUNT,
        node_count,
        arena.expr_count,
    ]);
    let mut execution = vyre_megakernel::execute_single_program(
        executor,
        "validate-limits",
        build_validate_limits_program(arena.expr_count.max(1)),
        &[depths_bytes, limits_bytes],
        policy,
    )
    .map_err(ValidateError::Semantic)?;
    if execution.outputs.len() != 1 {
        return Err(ValidateError::Semantic(SemanticExecutionError::Backend(
            format!(
                "validate-limits semantic execution expected one canonical output, got {}",
                execution.outputs.len()
            ),
        )));
    }
    let mut violations = Vec::with_capacity(2);
    decode_u32_output_exact(
        &execution.outputs.remove(0),
        2,
        "gpu_validate_limits violations",
        &mut violations,
    )
    .map_err(|error| {
        ValidateError::Semantic(SemanticExecutionError::Backend(format!(
            "validate-limits semantic output decoding failed: {error}"
        )))
    })?;
    Ok([violations[0] != 0, violations[1] != 0])
}

/// Build the limit-checker Program. Single workgroup [256, 1, 1].
/// Threads cooperate on a max-reduce of `depths`; thread 0 then
/// compares against the limits and writes the violations bitmap.
///
/// Buffer layout:
///   0: depths    (RO)   -  per-Expr depth (column from `ExprArenaEncoding.depths`)
///   1: limits    (RO)   -  `[max_expr_depth, max_node_count, node_count, expr_count]`
///   2: violations (RW)  -  2 u32 slots; index 0 = V033, index 1 = V019
#[must_use]
pub fn build_validate_limits_program(expr_count: u32) -> Program {
    let buffers = vec![
        BufferDecl::storage("depths", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("limits", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
        BufferDecl::output("violations", 2, DataType::U32).with_count(2),
    ];

    // Per-thread strided max-reduce. Each thread t computes a local
    // max over depths[t, t + WG, t + 2·WG, …]. Then we coalesce via a
    // single-element-per-thread atomic_max-emulated reduction: every
    // thread atomically OR-merges its local_max into a shared `gmax`
    // u32 word in the violations buffer (slot 0 used as scratch
    // before the final compare).
    //
    // Correctness note: we use atomic_or as a max-emulator only when
    // we KNOW the depths fit in the 0..2¹⁶ range (well below the
    // u32 OR-saturation point). For depths outside that, we'd need
    // a CAS loop. V033's limit is 1024 so depths are bounded; we
    // emit a CAS loop anyway for safety.
    let chunk_cap = (expr_count + VALIDATOR_WORKGROUP_X - 1) / VALIDATOR_WORKGROUP_X;

    let body = vec![
        Node::let_bind("local_max", Expr::u32(0)),
        Node::loop_for(
            "chunk",
            Expr::u32(0),
            Expr::u32(chunk_cap.max(1)),
            vec![
                Node::let_bind(
                    "i",
                    Expr::add(
                        Expr::gid_x(),
                        Expr::mul(Expr::var("chunk"), Expr::u32(VALIDATOR_WORKGROUP_X)),
                    ),
                ),
                Node::if_then(
                    Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
                    vec![
                        Node::let_bind("d", Expr::load("depths", Expr::var("i"))),
                        Node::if_then(
                            Expr::lt(Expr::var("local_max"), Expr::var("d")),
                            vec![Node::assign("local_max", Expr::var("d"))],
                        ),
                    ],
                ),
            ],
        ),
        // Thread 0 seeds the global max in violations[0] with this
        // thread's local_max. Every other thread CAS-updates if its
        // local_max is greater. Single workgroup ⇒ every thread sees
        // the seed before the CAS sequence.
        Node::if_then(
            Expr::eq(Expr::gid_x(), Expr::u32(0)),
            vec![Node::store(
                "violations",
                Expr::u32(0),
                Expr::var("local_max"),
            )],
        ),
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        },
        // CAS loop: each thread tries to bump violations[0] up to
        // local_max if its local_max is greater. Bounded by a small
        // retry count to avoid pathological contention.
        Node::loop_for(
            "cas_retry",
            Expr::u32(0),
            Expr::u32(8),
            vec![
                Node::let_bind("cur", Expr::load("violations", Expr::u32(0))),
                Node::if_then(
                    Expr::lt(Expr::var("cur"), Expr::var("local_max")),
                    vec![Node::let_bind(
                        "_cas",
                        Expr::atomic_compare_exchange_ordered(
                            "violations",
                            Expr::u32(0),
                            Expr::var("cur"),
                            Expr::var("local_max"),
                            vyre_foundation::ir::MemoryOrdering::SeqCst,
                        ),
                    )],
                ),
            ],
        ),
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        },
        // Thread 0 reads the final max + the limits and writes the
        // violations bitmap. We deliberately overwrite violations[0]
        // with the V033 bit (1 if max_depth > limit, 0 otherwise),
        // discarding the scratch max value.
        Node::if_then(
            Expr::eq(Expr::gid_x(), Expr::u32(0)),
            vec![
                Node::let_bind("max_depth", Expr::load("violations", Expr::u32(0))),
                Node::let_bind("max_expr_depth_lim", Expr::load("limits", Expr::u32(0))),
                Node::let_bind("max_node_count_lim", Expr::load("limits", Expr::u32(1))),
                Node::let_bind("node_count", Expr::load("limits", Expr::u32(2))),
                // V033: depth > limit
                Node::if_then(
                    Expr::lt(Expr::var("max_expr_depth_lim"), Expr::var("max_depth")),
                    vec![Node::store("violations", Expr::u32(0), Expr::u32(1))],
                ),
                Node::if_then(
                    Expr::le(Expr::var("max_depth"), Expr::var("max_expr_depth_lim")),
                    vec![Node::store("violations", Expr::u32(0), Expr::u32(0))],
                ),
                // V019: node_count > limit
                Node::if_then(
                    Expr::lt(Expr::var("max_node_count_lim"), Expr::var("node_count")),
                    vec![Node::store("violations", Expr::u32(1), Expr::u32(1))],
                ),
                Node::if_then(
                    Expr::le(Expr::var("node_count"), Expr::var("max_node_count_lim")),
                    vec![Node::store("violations", Expr::u32(1), Expr::u32(0))],
                ),
            ],
        ),
    ];

    Program::wrapped(
        buffers,
        [VALIDATOR_WORKGROUP_X, 1, 1],
        vec![Node::Region {
            generator: Ident::from("vyre-pass-engine::optimizer::validate_via_encoded"),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}
