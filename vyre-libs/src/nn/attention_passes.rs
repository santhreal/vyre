//! Reusable attention passes built from the shared `dot_partial` primitive.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use crate::math::dot_partial::{dot_partial, OP_ID as DOT_PARTIAL_OP_ID};
use crate::nn::attention_stability::{
    bounded_exp_arg, bounded_score, direct_score_expr, positive_denominator,
};
use crate::nn::f32_stability::{finite_or, flush_tiny};

/// Stable op id for the max-score pass.
pub const ATTENTION_MAX_PASS_OP_ID: &str = "vyre-libs::nn::attention_max_pass";
/// Stable op id for the normalization-sum pass.
pub const ATTENTION_SUM_PASS_OP_ID: &str = "vyre-libs::nn::attention_sum_pass";
/// Stable op id for the weighted-value write pass.
pub const ATTENTION_WRITE_PASS_OP_ID: &str = "vyre-libs::nn::attention_write_pass";

/// Emit the attention max-reduction pass for one contiguous query row `i`.
#[must_use]
pub fn attention_max_pass(q: &str, k: &str, d: u32, s: u32, scale_expr: Expr) -> Vec<Node> {
    attention_max_pass_bounded(
        q,
        k,
        d,
        Expr::u32(s),
        scale_expr,
        Expr::mul(Expr::var("i"), Expr::u32(d)),
        Expr::u32(0),
    )
}

/// Emit a max-score pass whose causal/cache key limit is an expression.
#[must_use]
pub fn attention_max_pass_bounded(
    q: &str,
    k: &str,
    d: u32,
    key_limit: Expr,
    scale_expr: Expr,
    query_base: Expr,
    key_base: Expr,
) -> Vec<Node> {
    let parent = Ident::from(ATTENTION_MAX_PASS_OP_ID);
    vec![Node::loop_for(
        "j",
        Expr::u32(0),
        key_limit,
        vec![wrap_child_region(
            DOT_PARTIAL_OP_ID,
            parent,
            vec![
                Node::let_bind("dot_val", Expr::f32(0.0)),
                dot_partial(
                    q,
                    k,
                    "dot_val",
                    query_base,
                    Expr::add(key_base, Expr::mul(Expr::var("j"), Expr::u32(d))),
                    d,
                ),
                Node::let_bind(
                    "score",
                    bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr)),
                ),
                Node::let_bind(
                    "finite_score",
                    finite_or(Expr::var("score"), Expr::f32(f32::MIN)),
                ),
                Node::assign(
                    "max_val",
                    Expr::select(
                        Expr::BinOp {
                            op: BinOp::Gt,
                            left: Box::new(Expr::var("finite_score")),
                            right: Box::new(Expr::var("max_val")),
                        },
                        Expr::var("finite_score"),
                        Expr::var("max_val"),
                    ),
                ),
            ],
        )],
    )]
}

/// Standalone max-score pass for query row 0.
#[must_use]
pub fn attention_max_pass_program(q: &str, k: &str, out: &str, s: u32, d: u32) -> Program {
    let scale_expr = Expr::f32(1.0f32 / (d as f32).sqrt());
    if s <= 8 && d <= 16 {
        let mut max_val = Expr::f32(f32::MIN);
        for col in 0..s {
            let score = finite_or(
                direct_score_expr(q, k, 0, col, d, scale_expr.clone()),
                Expr::f32(f32::MIN),
            );
            max_val = Expr::select(Expr::gt(score.clone(), max_val.clone()), score, max_val);
        }
        return Program::wrapped(
            vec![
                BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
                BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(s.saturating_mul(d)),
                BufferDecl::storage(out, 2, BufferAccess::ReadWrite, DataType::F32).with_count(1),
            ],
            [1, 1, 1],
            vec![wrap_anonymous_region(
                ATTENTION_MAX_PASS_OP_ID,
                vec![Node::store(out, Expr::u32(0), max_val)],
            )],
        );
    }
    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(s.saturating_mul(d)),
            BufferDecl::storage(out, 2, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            ATTENTION_MAX_PASS_OP_ID,
            vec![
                Node::let_bind("i", Expr::u32(0)),
                Node::let_bind("max_val", Expr::f32(f32::MIN)),
                Node::Block(attention_max_pass(q, k, d, s, scale_expr)),
                Node::store(out, Expr::u32(0), Expr::var("max_val")),
            ],
        )],
    )
}

/// Emit the attention normalization-sum pass for one contiguous query row `i`.
#[must_use]
pub fn attention_sum_pass(q: &str, k: &str, d: u32, s: u32, scale_expr: Expr) -> Vec<Node> {
    attention_sum_pass_bounded(
        q,
        k,
        d,
        Expr::u32(s),
        scale_expr,
        Expr::mul(Expr::var("i"), Expr::u32(d)),
        Expr::u32(0),
    )
}

/// Emit a normalization-sum pass whose causal/cache key limit is an expression.
#[must_use]
pub fn attention_sum_pass_bounded(
    q: &str,
    k: &str,
    d: u32,
    key_limit: Expr,
    scale_expr: Expr,
    query_base: Expr,
    key_base: Expr,
) -> Vec<Node> {
    let parent = Ident::from(ATTENTION_SUM_PASS_OP_ID);
    vec![Node::loop_for(
        "j",
        Expr::u32(0),
        key_limit,
        vec![wrap_child_region(
            DOT_PARTIAL_OP_ID,
            parent,
            vec![
                Node::let_bind("dot_val", Expr::f32(0.0)),
                dot_partial(
                    q,
                    k,
                    "dot_val",
                    query_base,
                    Expr::add(key_base, Expr::mul(Expr::var("j"), Expr::u32(d))),
                    d,
                ),
                Node::let_bind(
                    "score",
                    bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr)),
                ),
                Node::let_bind(
                    "exp_arg",
                    bounded_exp_arg(Expr::sub(Expr::var("score"), Expr::var("max_val"))),
                ),
                Node::assign(
                    "sum_val",
                    Expr::add(
                        Expr::var("sum_val"),
                        Expr::UnOp {
                            op: UnOp::Exp,
                            operand: Box::new(Expr::var("exp_arg")),
                        },
                    ),
                ),
            ],
        )],
    )]
}

/// Standalone normalization-sum pass for query row 0.
#[must_use]
pub fn attention_sum_pass_program(
    q: &str,
    k: &str,
    max_in: &str,
    out: &str,
    s: u32,
    d: u32,
) -> Program {
    let scale_expr = Expr::f32(1.0f32 / (d as f32).sqrt());
    if s <= 8 && d <= 16 {
        let max_val = Expr::load(max_in, Expr::u32(0));
        let mut sum_val = Expr::f32(0.0);
        for col in 0..s {
            let score = direct_score_expr(q, k, 0, col, d, scale_expr.clone());
            let exp_arg = bounded_exp_arg(Expr::sub(score, max_val.clone()));
            sum_val = Expr::add(
                sum_val,
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(exp_arg),
                },
            );
        }
        return Program::wrapped(
            vec![
                BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
                BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(s.saturating_mul(d)),
                BufferDecl::storage(max_in, 2, BufferAccess::ReadOnly, DataType::F32).with_count(1),
                BufferDecl::storage(out, 3, BufferAccess::ReadWrite, DataType::F32).with_count(1),
            ],
            [1, 1, 1],
            vec![wrap_anonymous_region(
                ATTENTION_SUM_PASS_OP_ID,
                vec![Node::store(out, Expr::u32(0), sum_val)],
            )],
        );
    }
    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(s.saturating_mul(d)),
            BufferDecl::storage(max_in, 2, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage(out, 3, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            ATTENTION_SUM_PASS_OP_ID,
            vec![
                Node::let_bind("i", Expr::u32(0)),
                Node::let_bind("max_val", Expr::load(max_in, Expr::u32(0))),
                Node::let_bind("sum_val", Expr::f32(0.0)),
                Node::Block(attention_sum_pass(q, k, d, s, scale_expr)),
                Node::store(out, Expr::u32(0), Expr::var("sum_val")),
            ],
        )],
    )
}

/// Emit the weighted-value write pass for one contiguous query row `i`.
#[must_use]
pub fn attention_write_pass(
    q: &str,
    k: &str,
    v: &str,
    d: u32,
    s: u32,
    scale_expr: Expr,
    out: &str,
) -> Vec<Node> {
    let row_base = Expr::mul(Expr::var("i"), Expr::u32(d));
    attention_write_pass_bounded(
        q,
        k,
        v,
        d,
        Expr::u32(s),
        scale_expr,
        out,
        row_base.clone(),
        Expr::u32(0),
        Expr::u32(0),
        row_base,
    )
}

/// Emit a weighted-value write pass whose causal/cache key limit is an expression.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn attention_write_pass_bounded(
    q: &str,
    k: &str,
    v: &str,
    d: u32,
    key_limit: Expr,
    scale_expr: Expr,
    out: &str,
    query_base: Expr,
    key_base: Expr,
    value_base: Expr,
    output_base: Expr,
) -> Vec<Node> {
    attention_write_pass_bounded_typed(
        q,
        k,
        v,
        d,
        key_limit,
        scale_expr,
        out,
        query_base,
        key_base,
        value_base,
        output_base,
        DataType::F32,
    )
}

/// Emit a typed weighted-value write pass with F32 accumulation.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn attention_write_pass_bounded_typed(
    q: &str,
    k: &str,
    v: &str,
    d: u32,
    key_limit: Expr,
    scale_expr: Expr,
    out: &str,
    query_base: Expr,
    key_base: Expr,
    value_base: Expr,
    output_base: Expr,
    output_dtype: DataType,
) -> Vec<Node> {
    let parent = Ident::from(ATTENTION_WRITE_PASS_OP_ID);
    vec![Node::loop_for(
        "t",
        Expr::u32(0),
        Expr::u32(d),
        vec![
            Node::let_bind("accum", Expr::f32(0.0)),
            Node::loop_for(
                "j",
                Expr::u32(0),
                key_limit,
                vec![wrap_child_region(
                    DOT_PARTIAL_OP_ID,
                    parent,
                    vec![
                        Node::let_bind("dot_val", Expr::f32(0.0)),
                        dot_partial(
                            q,
                            k,
                            "dot_val",
                            query_base,
                            Expr::add(key_base, Expr::mul(Expr::var("j"), Expr::u32(d))),
                            d,
                        ),
                        Node::let_bind(
                            "score",
                            bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr)),
                        ),
                        Node::let_bind(
                            "exp_arg",
                            bounded_exp_arg(Expr::sub(Expr::var("score"), Expr::var("max_val"))),
                        ),
                        Node::let_bind(
                            "weight",
                            Expr::BinOp {
                                op: BinOp::Div,
                                left: Box::new(Expr::UnOp {
                                    op: UnOp::Exp,
                                    operand: Box::new(Expr::var("exp_arg")),
                                }),
                                right: Box::new(Expr::var("denom")),
                            },
                        ),
                        Node::let_bind(
                            "value",
                            finite_or(
                                Expr::cast(
                                    DataType::F32,
                                    Expr::load(
                                        v,
                                        Expr::add(
                                            Expr::add(
                                                value_base,
                                                Expr::mul(Expr::var("j"), Expr::u32(d)),
                                            ),
                                            Expr::var("t"),
                                        ),
                                    ),
                                ),
                                Expr::f32(0.0),
                            ),
                        ),
                        Node::assign(
                            "accum",
                            Expr::add(
                                Expr::var("accum"),
                                Expr::mul(Expr::var("weight"), Expr::var("value")),
                            ),
                        ),
                    ],
                )],
            ),
            Node::Store {
                buffer: out.into(),
                index: Expr::add(output_base, Expr::var("t")),
                value: Expr::cast(output_dtype.clone(), flush_tiny(Expr::var("accum"))),
            },
        ],
    )]
}

/// Buffer names and dimensions for a standalone weighted-value write pass.
pub struct AttentionWritePassProgramSpec<'a> {
    /// Query buffer.
    pub q: &'a str,
    /// Key buffer.
    pub k: &'a str,
    /// Value buffer.
    pub v: &'a str,
    /// Single-element max-score input buffer.
    pub max_in: &'a str,
    /// Single-element normalization-sum input buffer.
    pub sum_in: &'a str,
    /// Output buffer.
    pub out: &'a str,
    /// Sequence length.
    pub s: u32,
    /// Head dimension.
    pub d: u32,
}

/// Standalone weighted-value write pass for query row 0.
#[must_use]
pub fn attention_write_pass_program(spec: AttentionWritePassProgramSpec<'_>) -> Program {
    let AttentionWritePassProgramSpec {
        q,
        k,
        v,
        max_in,
        sum_in,
        out,
        s,
        d,
    } = spec;
    let scale_expr = Expr::f32(1.0f32 / (d as f32).sqrt());
    let elements = s.saturating_mul(d);
    if s <= 8 && d <= 16 {
        let max_val = Expr::load(max_in, Expr::u32(0));
        let denom = positive_denominator(Expr::load(sum_in, Expr::u32(0)));
        let mut stores = Vec::with_capacity(d as usize);
        for dim in 0..d {
            let mut accum = Expr::f32(0.0);
            for col in 0..s {
                let score = direct_score_expr(q, k, 0, col, d, scale_expr.clone());
                let weight = Expr::div(
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(bounded_exp_arg(Expr::sub(score, max_val.clone()))),
                    },
                    denom.clone(),
                );
                let value = finite_or(Expr::load(v, Expr::u32(col * d + dim)), Expr::f32(0.0));
                accum = Expr::add(accum, Expr::mul(weight, value));
            }
            stores.push(Node::store(out, Expr::u32(dim), flush_tiny(accum)));
        }
        return Program::wrapped(
            vec![
                BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
                BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(elements),
                BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(elements),
                BufferDecl::storage(max_in, 3, BufferAccess::ReadOnly, DataType::F32).with_count(1),
                BufferDecl::storage(sum_in, 4, BufferAccess::ReadOnly, DataType::F32).with_count(1),
                BufferDecl::storage(out, 5, BufferAccess::ReadWrite, DataType::F32).with_count(d),
            ],
            [1, 1, 1],
            vec![wrap_anonymous_region(ATTENTION_WRITE_PASS_OP_ID, stores)],
        );
    }
    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(d),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(max_in, 3, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage(sum_in, 4, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage(out, 5, BufferAccess::ReadWrite, DataType::F32).with_count(d),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            ATTENTION_WRITE_PASS_OP_ID,
            vec![
                Node::let_bind("i", Expr::u32(0)),
                Node::let_bind("max_val", Expr::load(max_in, Expr::u32(0))),
                Node::let_bind("sum_val", Expr::load(sum_in, Expr::u32(0))),
                Node::let_bind("denom", positive_denominator(Expr::var("sum_val"))),
                Node::Block(attention_write_pass(q, k, v, d, s, scale_expr, out)),
            ],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        ATTENTION_MAX_PASS_OP_ID,
        || attention_max_pass_program("q", "k", "out", 2, 2),
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32_bytes(&[0.0, 0.0]),
                to_f32_bytes(&[1.0, 0.0, 2.0, 0.0]),
                vec![0u8; 4],
            ]]
        }),
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![to_f32_bytes(&[0.0])]]
        }),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        ATTENTION_SUM_PASS_OP_ID,
        || attention_sum_pass_program("q", "k", "max", "out", 2, 2),
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32_bytes(&[0.0, 0.0]),
                to_f32_bytes(&[1.0, 0.0, 2.0, 0.0]),
                to_f32_bytes(&[0.0]),
                vec![0u8; 4],
            ]]
        }),
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![to_f32_bytes(&[2.0])]]
        }),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        ATTENTION_WRITE_PASS_OP_ID,
        || {
            attention_write_pass_program(AttentionWritePassProgramSpec {
                q: "q",
                k: "k",
                v: "v",
                max_in: "max",
                sum_in: "sum",
                out: "out",
                s: 2,
                d: 1,
            })
        },
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32_bytes(&[0.0]),
                to_f32_bytes(&[1.0, 1.0]),
                to_f32_bytes(&[20.0, 20.0]),
                to_f32_bytes(&[0.0]),
                to_f32_bytes(&[2.0]),
                vec![0u8; 4],
            ]]
        }),
        Some(|| {
            let to_f32_bytes = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![to_f32_bytes(&[20.0])]]
        }),
    )
}
