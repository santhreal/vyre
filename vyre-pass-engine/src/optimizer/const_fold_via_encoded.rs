//! Constant folding as a dispatched compute kernel.
//!
//! V1 scope: literals (`LitU32`) plus the integer-arithmetic
//! BinOps (`Add`, `Sub`, `Mul`, `BitAnd`, `BitOr`, `BitXor`) on u32.
//! Larger op coverage is mechanical extension of `build_const_fold_program`.
//!
//! Architecture: the encoder turns every Expr into a `(kind, arg0,
//! arg1, arg2, arg3)` row in the canonical `ExprArenaEncoding`. The
//! `build_const_fold_program` function constructs a vyre `Program`
//! that scans the arena bottom-up, marking each foldable Expr in a
//! `foldable[]` u32 buffer and writing its computed value into a
//! `value[]` u32 buffer. The `ProgramDispatcher` runs that Program
//! on the GPU; the decoder walks the IR and rewrites every foldable
//! Expr into a literal.
//!
//! No host-reference escape in production. Tests parity vs the existing
//! `vyre-foundation` const-fold pass via `CpuOracleDispatcher`-style
//! tests (extension follow-up  -  for V1 we run through a real
//! backend in the driver integration test crates).

use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use vyre_libs::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};

use super::encode::EncodeError;
use super::expr_arena::{encode_expr_arena, expr_kind, ExprArenaEncoding};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

#[derive(Debug, Default)]
struct ConstFoldKernelScratch {
    inputs: Vec<Vec<u8>>,
    current_level: [u32; 1],
}

/// Errors surfaced by `gpu_const_fold`.
#[derive(Debug)]
pub enum ConstFoldError {
    /// Encoder did not accept the input shape.
    Encode(EncodeError),
    /// Dispatcher rejected or failed to run the analysis Program.
    Dispatch(DispatchError),
}

impl std::fmt::Display for ConstFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_const_fold encode error: {err:?}"),
            Self::Dispatch(err) => write!(f, "gpu_const_fold dispatch error: {err}"),
        }
    }
}

impl std::error::Error for ConstFoldError {}

/// Run constant-folding on `program` by encoding its Expr arena,
/// dispatching the bottom-up evaluator Program through `dispatcher`,
/// and rewriting every foldable Expr in the input Program into the
/// computed literal value.
pub fn gpu_const_fold(
    program: Program,
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Program, ConstFoldError> {
    let arena = encode_expr_arena(&program).map_err(ConstFoldError::Encode)?;
    if arena.expr_count == 0 {
        return Ok(program);
    }
    let mut scratch = ConstFoldKernelScratch::default();
    let mut foldable = Vec::with_capacity(arena.expr_count as usize);
    let mut value = Vec::with_capacity(arena.expr_count as usize);
    run_const_fold_kernel_with_scratch_into(
        &arena,
        dispatcher,
        &mut scratch,
        &mut foldable,
        &mut value,
    )
    .map_err(ConstFoldError::Dispatch)?;
    Ok(rewrite_program_with_folded_values(
        program, &foldable, &value,
    ))
}

#[cfg(test)]
fn run_const_fold_kernel_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    foldable: &mut Vec<u32>,
    value: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = ConstFoldKernelScratch::default();
    run_const_fold_kernel_with_scratch_into(arena, dispatcher, &mut scratch, foldable, value)
}

fn run_const_fold_kernel_with_scratch_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    scratch: &mut ConstFoldKernelScratch,
    foldable: &mut Vec<u32>,
    value: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let n = arena.expr_count;
    let analysis = build_const_fold_program(n);
    let words = n as usize;
    let state_bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: const-fold state byte count overflows usize for expr_count={n}."
            ))
        })?;

    // Level-parallel kernel: each dispatch processes all Exprs at one
    // depth level in parallel via gid_x(). The host loops levels
    // 0..=max_depth, updating the `current_level` buffer between
    // dispatches. Foldable + value buffers persist their state across
    // levels (we re-feed the previous output as the next input).
    //
    // Buffer order matches `build_const_fold_program`'s declarations:
    //   0: arena_kinds (RO)
    //   1: arena_arg0 (RO)
    //   2: arena_arg1 (RO)
    //   3: arena_arg2 (RO)
    //   4: arena_depths (RO)  -  per-Expr depth
    //   5: current_level (RO)  -  single u32, varied per dispatch
    //   6: foldable (RW; init zeros, persists across levels)
    //   7: value (RW; init zeros, persists across levels)
    ensure_input_slots(&mut scratch.inputs, 8);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.inputs[3], &arena.arg2);
    write_u32_slice_le_bytes(&mut scratch.inputs[4], &arena.depths);
    write_zero_bytes(&mut scratch.inputs[6], state_bytes);
    write_zero_bytes(&mut scratch.inputs[7], state_bytes);

    let grid_x = (n + WORKGROUP_X - 1) / WORKGROUP_X;

    for level in 0..=arena.max_depth {
        scratch.current_level[0] = level;
        write_u32_slice_le_bytes(&mut scratch.inputs[5], &scratch.current_level);

        let outputs = dispatcher.dispatch(&analysis, &scratch.inputs, Some([grid_x, 1, 1]))?;
        if outputs.len() != 2 {
            return Err(DispatchError::BackendError(format!(
                "Fix: const-fold dispatch expected exactly 2 outputs (foldable, value), got {}.",
                outputs.len()
            )));
        }
        decode_u32_output_exact(&outputs[0], words, "const-fold foldable", foldable)?;
        decode_u32_output_exact(&outputs[1], words, "const-fold value", value)?;
        // Carry RW state forward to the next level dispatch.
        scratch.inputs[6].clear();
        scratch.inputs[6].extend_from_slice(&outputs[0]);
        scratch.inputs[7].clear();
        scratch.inputs[7].extend_from_slice(&outputs[1]);
    }

    Ok(())
}

/// Workgroup size for the level-parallel const-fold kernel.
const WORKGROUP_X: u32 = super::arena_kernel::WORKGROUP_X;

/// Build the FUSED const-fold analysis Program: a single dispatch that
/// internally iterates `level` from 0..=`max_depth`, with a workgroup-
/// scope barrier between levels. Eliminates the per-level host
/// dispatch loop that dominates chain-shaped Programs.
///
/// The level wave itself is
/// `super::arena_kernel::build_fused_level_wave_program`; const-fold
/// contributes the `foldable` / `value` outputs at bindings 6 and 7 and
/// the per-Expr body that writes them.
#[must_use]
pub fn build_const_fold_program_fused(expr_count: u32, max_depth_iter_cap: u32) -> Program {
    let count = expr_count.max(1);
    super::arena_kernel::build_fused_level_wave_program(
        expr_count,
        max_depth_iter_cap,
        vec![
            BufferDecl::storage("foldable", 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(count),
            BufferDecl::storage("value", 7, BufferAccess::ReadWrite, DataType::U32)
                .with_count(count),
        ],
        per_expr_body(),
    )
}

/// Build the const-fold analysis Program. Level-parallel kernel: each
/// GPU thread handles one Expr id via `gid_x()` and acts only when
/// the Expr's depth equals `current_level[0]`. The orchestrator
/// dispatches once per level (0..=max_depth), with foldable + value
/// buffers persisting their state across dispatches.
pub fn build_const_fold_program(expr_count: u32) -> Program {
    let count = expr_count.max(1);
    let mut buffers = super::arena_kernel::arena_row_buffers(expr_count, 0);
    buffers.extend([
        BufferDecl::storage("arena_depths", 4, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
        BufferDecl::storage("current_level", 5, BufferAccess::ReadOnly, DataType::U32)
            .with_count(1),
        BufferDecl::storage("foldable", 6, BufferAccess::ReadWrite, DataType::U32)
            .with_count(count),
        BufferDecl::storage("value", 7, BufferAccess::ReadWrite, DataType::U32).with_count(count),
    ]);

    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("my_depth", Expr::load("arena_depths", Expr::var("i"))),
                Node::let_bind("level", Expr::load("current_level", Expr::u32(0))),
                Node::if_then(
                    Expr::eq(Expr::var("my_depth"), Expr::var("level")),
                    per_expr_body(),
                ),
            ],
        ),
    ];

    Program::wrapped(buffers, [WORKGROUP_X, 1, 1], body)
}

/// Per-Expr-id body of the sequential const-fold scan.
fn per_expr_body() -> Vec<Node> {
    vec![
        // let kind = arena_kinds[i]
        Node::let_bind("kind", Expr::load("arena_kinds", Expr::var("i"))),
        // Literal kinds: foldable=1, value = arena_arg0[i].
        // (V1 covers LitU32 only; LitI32/F32/Bool fold by reinterpret
        //  but the kernel emits the same payload bits, so adding their
        //  kind discriminants follows the same pattern with no new
        //  arithmetic.)
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LIT_U32)),
            vec![
                Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                Node::store(
                    "value",
                    Expr::var("i"),
                    Expr::load("arena_arg0", Expr::var("i")),
                ),
            ],
        ),
        // BIN_OP: only fold if both operands are themselves foldable.
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BIN_OP)),
            bin_op_body(),
        ),
    ]
}

/// Body of the BIN_OP arm: read op tag + child ids, check both
/// children are foldable, compute the result if the op is one of the
/// V1-supported integer arithmetic ops.
fn bin_op_body() -> Vec<Node> {
    vec![
        Node::let_bind("op", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind("l", Expr::load("arena_arg1", Expr::var("i"))),
        Node::let_bind("r", Expr::load("arena_arg2", Expr::var("i"))),
        Node::let_bind("lf", Expr::load("foldable", Expr::var("l"))),
        Node::let_bind("rf", Expr::load("foldable", Expr::var("r"))),
        Node::if_then(
            // lf == 1 && rf == 1
            Expr::and(
                Expr::eq(Expr::var("lf"), Expr::u32(1)),
                Expr::eq(Expr::var("rf"), Expr::u32(1)),
            ),
            vec![
                Node::let_bind("lv", Expr::load("value", Expr::var("l"))),
                Node::let_bind("rv", Expr::load("value", Expr::var("r"))),
                // Add (tag 0x01)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x01)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::add(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Sub (tag 0x02)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x02)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::sub(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Mul (tag 0x03)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x03)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::mul(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // BitAnd (tag 0x06)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x06)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::bitand(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // BitOr (tag 0x07)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x07)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::bitor(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // BitXor (tag 0x08)
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x08)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::bitxor(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Shl (tag 0x09). u32 shift; rv must be in 0..32 to
                // be well-defined. We fold for any rv since the
                // wrapping behaviour matches target shift semantics.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x09)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::shl(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Shr (tag 0x0A)  -  logical shift right.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x0A)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::shr(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Div (tag 0x04)  -  fold only if `rv != 0`. Folding
                // a division by zero would crash the compiler at
                // emit time; the host-side rewriter still emits the
                // original Div which lets the program's own runtime
                // semantics decide.
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("op"), Expr::u32(0x04)),
                        Expr::ne(Expr::var("rv"), Expr::u32(0)),
                    ),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::div(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Mod (tag 0x05)  -  same divide-by-zero guard.
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("op"), Expr::u32(0x05)),
                        Expr::ne(Expr::var("rv"), Expr::u32(0)),
                    ),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::rem(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Min (tag 0x15)  -  `lv if lv < rv else rv`. Folded
                // via a Select gated on Lt; works for u32 directly.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x15)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::lt(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::var("lv")),
                                false_val: Box::new(Expr::var("rv")),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Max (tag 0x16)  -  symmetric.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x16)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::gt(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::var("lv")),
                                false_val: Box::new(Expr::var("rv")),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // AbsDiff (tag 0x14)  -  `|lv - rv|` for u32 = if lv >
                // rv then lv-rv else rv-lv. Always non-negative.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x14)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::gt(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::sub(Expr::var("lv"), Expr::var("rv"))),
                                false_val: Box::new(Expr::sub(Expr::var("rv"), Expr::var("lv"))),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // SaturatingAdd (0x17)  -  clamps to u32::MAX when the
                // unsaturated sum would overflow. Detect overflow by
                // checking if the wrapped sum is less than either
                // operand (carry happened).
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x17)),
                    vec![
                        Node::let_bind("sat_sum", Expr::add(Expr::var("lv"), Expr::var("rv"))),
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::lt(Expr::var("sat_sum"), Expr::var("lv"))),
                                true_val: Box::new(Expr::u32(u32::MAX)),
                                false_val: Box::new(Expr::var("sat_sum")),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // SaturatingSub (0x18)  -  clamps to 0 when rv > lv.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x18)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::ge(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::sub(Expr::var("lv"), Expr::var("rv"))),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // WrappingAdd (0x20)  -  same as `Add` for u32 since
                // backend Add already wraps. Fold straight through.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x20)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::add(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // WrappingSub (0x21)  -  same as `Sub` for u32.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x21)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::sub(Expr::var("lv"), Expr::var("rv")),
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // Comparison ops (Eq=0x0B, Ne=0x0C, Lt=0x0D, Gt=0x0E,
                // Le=0x10, Ge=0x11). Writes 0/1 into `value` because
                // the decoder reconstructs LitU32; downstream
                // dead-branch elimination accepts both LitU32(0|1)
                // and LitBool. CPU const-prop re-types to LitBool
                // when it later substitutes through a Var.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x0B)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::eq(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x0C)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::ne(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x0D)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::lt(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x0E)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::gt(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x10)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::le(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x11)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::ge(Expr::var("lv"), Expr::var("rv"))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::u32(0)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // RotateLeft (0x1E). The backend implements rotate
                // natively; the kernel just emits the BinOp.
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x1E)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::BinOp {
                                op: BinOp::RotateLeft,
                                left: Box::new(Expr::var("lv")),
                                right: Box::new(Expr::var("rv")),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // RotateRight (0x1F).
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x1F)),
                    vec![
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::BinOp {
                                op: BinOp::RotateRight,
                                left: Box::new(Expr::var("lv")),
                                right: Box::new(Expr::var("rv")),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
                // SaturatingMul (0x19). For u32, overflow detection
                // via the inverse-divide identity: if (lv*rv)/lv == rv
                // then no overflow. Guard the divisor with Select to
                // avoid div-by-zero when lv == 0 (in which case the
                // product is trivially 0, no overflow).
                Node::if_then(
                    Expr::eq(Expr::var("op"), Expr::u32(0x19)),
                    vec![
                        Node::let_bind("sm_prod", Expr::mul(Expr::var("lv"), Expr::var("rv"))),
                        Node::let_bind(
                            "sm_divisor",
                            Expr::Select {
                                cond: Box::new(Expr::eq(Expr::var("lv"), Expr::u32(0))),
                                true_val: Box::new(Expr::u32(1)),
                                false_val: Box::new(Expr::var("lv")),
                            },
                        ),
                        Node::let_bind(
                            "sm_quot",
                            Expr::div(Expr::var("sm_prod"), Expr::var("sm_divisor")),
                        ),
                        Node::let_bind(
                            "sm_no_overflow",
                            Expr::or(
                                Expr::eq(Expr::var("lv"), Expr::u32(0)),
                                Expr::eq(Expr::var("sm_quot"), Expr::var("rv")),
                            ),
                        ),
                        Node::store(
                            "value",
                            Expr::var("i"),
                            Expr::Select {
                                cond: Box::new(Expr::var("sm_no_overflow")),
                                true_val: Box::new(Expr::var("sm_prod")),
                                false_val: Box::new(Expr::u32(u32::MAX)),
                            },
                        ),
                        Node::store("foldable", Expr::var("i"), Expr::u32(1)),
                    ],
                ),
            ],
        ),
    ]
}

fn rewrite_program_with_folded_values(
    program: Program,
    foldable: &[u32],
    value: &[u32],
) -> Program {
    super::rewrite_walk::rewrite_program_with_expr_rewriter(&program, |expr, counter| {
        super::rewrite_walk::rewrite_simple_expr_postorder(expr, counter, &mut |rebuilt, id| {
            fold_decision(rebuilt, id, foldable, value)
        })
    })
}

/// Const-fold's rewrite decision for the Expr the postorder walk rebuilt at
/// arena id `id`.
///
/// `BinOp` and `UnOp` collapse to the literal the kernel computed. A leaf
/// collapses too unless it already is a literal: V1 folds to `LitU32`, so an
/// `i32`/`f32`/`bool` literal would lose its type. `Load`, `Select`, and `Fma`
/// are never marked foldable by the kernel and keep their rebuilt children.
fn fold_decision(rebuilt: Expr, id: u32, foldable: &[u32], value: &[u32]) -> Expr {
    if foldable[id as usize] != 1 {
        return rebuilt;
    }
    match rebuilt {
        Expr::Load { .. }
        | Expr::Select { .. }
        | Expr::Fma { .. }
        | Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_) => rebuilt,
        _ => Expr::LitU32(value[id as usize]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_libs::dispatch_buffers::u32_slice_to_le_bytes;

    use super::super::arena_kernel::{
        single_lit_u32_arena as one_expr_arena, FixedOutputDispatcher, GridExpectation,
    };

    fn dispatcher(outputs: Vec<Vec<u8>>) -> FixedOutputDispatcher {
        FixedOutputDispatcher {
            pass: "const-fold",
            expected_inputs: 8,
            grid: GridExpectation::StridedOverArena,
            outputs,
        }
    }

    #[test]
    fn kernel_into_decodes_exact_outputs_into_reused_buffers() {
        let dispatcher = dispatcher(vec![
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[7]),
        ]);
        let mut foldable = Vec::with_capacity(4);
        let mut value = Vec::with_capacity(4);
        let foldable_ptr = foldable.as_ptr();
        let value_ptr = value.as_ptr();
        run_const_fold_kernel_into(&one_expr_arena(), &dispatcher, &mut foldable, &mut value)
            .expect("Fix: dispatch succeeds");
        assert_eq!(foldable, vec![1]);
        assert_eq!(value, vec![7]);
        assert_eq!(foldable.as_ptr(), foldable_ptr);
        assert_eq!(value.as_ptr(), value_ptr);
    }

    #[test]
    fn kernel_with_scratch_reuses_dispatch_state_and_outputs() {
        let dispatcher = dispatcher(vec![
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[7]),
        ]);
        let arena = one_expr_arena();
        let mut scratch = ConstFoldKernelScratch::default();
        let mut foldable = Vec::with_capacity(1);
        let mut value = Vec::with_capacity(1);

        run_const_fold_kernel_with_scratch_into(
            &arena,
            &dispatcher,
            &mut scratch,
            &mut foldable,
            &mut value,
        )
        .expect("Fix: dispatch succeeds");

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let foldable_capacity = foldable.capacity();
        let value_capacity = value.capacity();

        run_const_fold_kernel_with_scratch_into(
            &arena,
            &dispatcher,
            &mut scratch,
            &mut foldable,
            &mut value,
        )
        .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(foldable.capacity(), foldable_capacity);
        assert_eq!(value.capacity(), value_capacity);
        assert_eq!(foldable, vec![1]);
        assert_eq!(value, vec![7]);
    }

    #[test]
    fn kernel_rejects_extra_outputs() {
        let dispatcher = dispatcher(vec![
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[7]),
            u32_slice_to_le_bytes(&[0]),
        ]);
        let mut foldable = Vec::new();
        let mut value = Vec::new();
        let err =
            run_const_fold_kernel_into(&one_expr_arena(), &dispatcher, &mut foldable, &mut value)
                .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn kernel_rejects_trailing_value_bytes() {
        let dispatcher = dispatcher(vec![u32_slice_to_le_bytes(&[1]), vec![7, 0, 0, 0, 1]]);
        let mut foldable = Vec::new();
        let mut value = Vec::new();
        let err =
            run_const_fold_kernel_into(&one_expr_arena(), &dispatcher, &mut foldable, &mut value)
                .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    /// P2 regression: the const-fold dispatcher receives `Some([grid_x, 1, 1])`
    /// where `grid_x = ceil(expr_count / WORKGROUP_X)`. For a 257-expr arena
    /// `grid_x` must be 2. The sibling arena passes each dispatch exactly one
    /// workgroup, and this pass's own test fake once copied their
    /// `Some([1, 1, 1])` assertion, which passed only for the trivial 1-expr
    /// case and left multi-workgroup dispatch unexercised.
    #[test]
    fn const_fold_kernel_sends_correct_multi_workgroup_grid_for_257_exprs() {
        struct GridCapture {
            /// Store the x-grid from each dispatch call.
            grid_x_values: std::cell::RefCell<Vec<u32>>,
            expr_count: u32,
        }
        impl ProgramDispatcher for GridCapture {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                grid_override: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                let gx = grid_override.map(|g| g[0]).unwrap_or(0);
                self.grid_x_values.borrow_mut().push(gx);
                let n = self.expr_count as usize;
                Ok(vec![
                    u32_slice_to_le_bytes(&vec![0u32; n]),
                    u32_slice_to_le_bytes(&vec![0u32; n]),
                ])
            }
        }

        let n: u32 = 257;
        let arena = ExprArenaEncoding {
            expr_count: n,
            kinds: vec![expr_kind::LIT_U32; n as usize],
            arg0: vec![0u32; n as usize],
            arg1: vec![0u32; n as usize],
            arg2: vec![0u32; n as usize],
            depths: vec![0u32; n as usize],
            max_depth: 0,
            ..ExprArenaEncoding::default()
        };
        let dispatcher = GridCapture {
            grid_x_values: std::cell::RefCell::new(Vec::new()),
            expr_count: n,
        };
        let mut foldable = Vec::new();
        let mut value = Vec::new();
        run_const_fold_kernel_into(&arena, &dispatcher, &mut foldable, &mut value)
            .expect("Fix: 257-expr const-fold dispatch succeeds");

        assert_eq!(foldable.len(), n as usize);
        // For n=257 and WORKGROUP_X=256: ceil(257/256) = 2.
        let expected_grid_x = (n + WORKGROUP_X - 1) / WORKGROUP_X;
        assert_eq!(
            expected_grid_x, 2,
            "sanity: expected_grid_x for 257 exprs must be 2"
        );
        for (dispatch_idx, &gx) in dispatcher.grid_x_values.borrow().iter().enumerate() {
            assert_eq!(
                gx, expected_grid_x,
                "dispatch {dispatch_idx}: grid_x must be {expected_grid_x} for expr_count={n}; \
                 asserting a literal Some([1,1,1]) here only works by accident for 1-expr \
                 arenas"
            );
        }
    }
}
