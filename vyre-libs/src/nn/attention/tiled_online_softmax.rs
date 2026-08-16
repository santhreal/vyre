//! The online-softmax attention core.
//!
//! One `(m, l, o_acc)` recurrence over one tiling serves every decoder in this
//! family. [`flash_attention`](super::flash_attention::flash_attention) and
//! [`flash_attention_2`](super::flash_attention_2::flash_attention_2) differ
//! only in the tile width the planner picks, and take the whole program from
//! [`online_softmax_attention`]. [`mla_decode`](super::mla::mla_decode) reads
//! its scores out of a compressed KV cache, so it supplies its own two
//! fragments to [`tiled_online_softmax_body`] and shares the recurrence around
//! them.
//!
//! Scalar is not a second algorithm. The scalar kernel is this core at
//! `tile_size = 1`, which is what the planner already records, so a
//! numerical-stability change cannot land in the tiled decoder and miss the
//! scalar one.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program, UnOp,
};

use super::planner::FlashAttentionWorkPlan;
use crate::nn::attention_stability::{bounded_exp_arg, bounded_score, positive_denominator};
use crate::nn::f32_stability::flush_tiny;

/// Registered owner of the online-softmax attention recurrence.
pub const OP_ID: &str = "vyre-libs::nn::attention::online_softmax";

pub(super) struct TiledOnlineSoftmaxSpec<'a> {
    pub(super) q: &'a str,
    pub(super) out: &'a str,
    pub(super) item_var: &'static str,
    pub(super) item_count: u32,
    pub(super) seq_len: u32,
    pub(super) head_dim: u32,
    pub(super) tile_size: u32,
    pub(super) tile_count: u32,
}

/// Flat index into one lane's `stride`-wide slice of workgroup scratch.
pub(super) fn scratch_index(stride: u32, local: Expr, offset: Expr) -> Expr {
    Expr::add(Expr::mul(local, Expr::u32(stride)), offset)
}

pub(super) fn tiled_online_softmax_body(
    spec: TiledOnlineSoftmaxSpec<'_>,
    compute_tile_scores: Vec<Node>,
    update_o_acc: Vec<Node>,
) -> Vec<Node> {
    let q_idx = |local: Expr, d: Expr| scratch_index(spec.head_dim, local, d);
    let score_idx = |local: Expr, j: Expr| scratch_index(spec.tile_size, local, j);
    let o_idx = |local: Expr, d: Expr| scratch_index(spec.head_dim, local, d);
    let load_q = Node::loop_for(
        "load_d",
        Expr::u32(0),
        Expr::u32(spec.head_dim),
        vec![Node::store(
            "q_scratch",
            q_idx(Expr::var("local"), Expr::var("load_d")),
            Expr::load(
                spec.q,
                Expr::add(
                    Expr::mul(Expr::var(spec.item_var), Expr::u32(spec.head_dim)),
                    Expr::var("load_d"),
                ),
            ),
        )],
    );
    let zero_o_acc = Node::loop_for(
        "zero_d",
        Expr::u32(0),
        Expr::u32(spec.head_dim),
        vec![Node::store(
            "o_acc",
            o_idx(Expr::var("local"), Expr::var("zero_d")),
            Expr::f32(0.0),
        )],
    );
    let find_tile_max = vec![
        Node::let_bind("tile_max", Expr::f32(f32::MIN)),
        Node::loop_for(
            "max_j",
            Expr::u32(0),
            Expr::var("tile_len"),
            vec![Node::assign(
                "tile_max",
                Expr::select(
                    Expr::is_nan(Expr::var("tile_max")),
                    Expr::var("tile_max"),
                    Expr::select(
                        Expr::gt(
                            Expr::load(
                                "score_tile",
                                score_idx(Expr::var("local"), Expr::var("max_j")),
                            ),
                            Expr::var("tile_max"),
                        ),
                        Expr::load(
                            "score_tile",
                            score_idx(Expr::var("local"), Expr::var("max_j")),
                        ),
                        Expr::var("tile_max"),
                    ),
                ),
            )],
        ),
    ];
    let mut tile_body = vec![
        Node::let_bind(
            "tile_start",
            Expr::mul(Expr::var("tile_idx"), Expr::u32(spec.tile_size)),
        ),
        Node::let_bind(
            "tile_end",
            Expr::min(
                Expr::add(Expr::var("tile_start"), Expr::u32(spec.tile_size)),
                Expr::u32(spec.seq_len),
            ),
        ),
        Node::let_bind(
            "tile_len",
            Expr::sub(Expr::var("tile_end"), Expr::var("tile_start")),
        ),
    ];
    tile_body.extend(compute_tile_scores);
    tile_body.extend(find_tile_max);
    tile_body.push(Node::let_bind(
        "m_new",
        Expr::select(
            Expr::gt(Expr::var("tile_max"), Expr::var("m")),
            Expr::var("tile_max"),
            Expr::var("m"),
        ),
    ));
    tile_body.push(Node::let_bind(
        "rescale",
        Expr::UnOp {
            op: UnOp::Exp,
            operand: Box::new(bounded_exp_arg(Expr::sub(
                Expr::var("m"),
                Expr::var("m_new"),
            ))),
        },
    ));
    tile_body.extend([
        Node::let_bind("tile_sum", Expr::f32(0.0)),
        Node::loop_for(
            "sum_j",
            Expr::u32(0),
            Expr::var("tile_len"),
            vec![Node::assign(
                "tile_sum",
                Expr::add(
                    Expr::var("tile_sum"),
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(bounded_exp_arg(Expr::sub(
                            Expr::load(
                                "score_tile",
                                score_idx(Expr::var("local"), Expr::var("sum_j")),
                            ),
                            Expr::var("m_new"),
                        ))),
                    },
                ),
            )],
        ),
        Node::assign(
            "l",
            Expr::add(
                Expr::mul(Expr::var("rescale"), Expr::var("l")),
                Expr::var("tile_sum"),
            ),
        ),
    ]);
    tile_body.extend(update_o_acc);
    tile_body.push(Node::assign("m", Expr::var("m_new")));

    let mut per_item = vec![
        load_q,
        Node::let_bind("m", Expr::f32(f32::MIN)),
        Node::let_bind("l", Expr::f32(0.0)),
        zero_o_acc,
        Node::loop_for(
            "tile_idx",
            Expr::u32(0),
            Expr::u32(spec.tile_count),
            tile_body,
        ),
        Node::let_bind("denom", positive_denominator(Expr::var("l"))),
    ];
    per_item.push(Node::loop_for(
        "final_d",
        Expr::u32(0),
        Expr::u32(spec.head_dim),
        vec![Node::store(
            spec.out,
            Expr::add(
                Expr::mul(Expr::var(spec.item_var), Expr::u32(spec.head_dim)),
                Expr::var("final_d"),
            ),
            flush_tiny(Expr::div(
                Expr::load("o_acc", o_idx(Expr::var("local"), Expr::var("final_d"))),
                Expr::var("denom"),
            )),
        )],
    ));
    vec![
        Node::let_bind(spec.item_var, Expr::InvocationId { axis: 0 }),
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var(spec.item_var), Expr::u32(spec.item_count)),
            per_item,
        ),
    ]
}

/// Scores of one KV tile against the staged query row.
///
/// The query is read from workgroup scratch, so a row's `head_dim` values
/// cross the memory bus once per invocation instead of once per key.
fn tile_scores(k: &str, head_dim: u32, tile_size: u32, scale: Expr) -> Vec<Node> {
    let q_idx = |local: Expr, d: Expr| scratch_index(head_dim, local, d);
    let score_idx = |local: Expr, j: Expr| scratch_index(tile_size, local, j);
    vec![Node::loop_for(
        "tile_j",
        Expr::u32(0),
        Expr::var("tile_len"),
        vec![
            Node::let_bind("dot_val", Expr::f32(0.0)),
            Node::loop_for(
                "score_d",
                Expr::u32(0),
                Expr::u32(head_dim),
                vec![Node::assign(
                    "dot_val",
                    Expr::add(
                        Expr::var("dot_val"),
                        Expr::mul(
                            Expr::load(
                                "q_scratch",
                                q_idx(Expr::var("local"), Expr::var("score_d")),
                            ),
                            Expr::load(
                                k,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("tile_start"), Expr::var("tile_j")),
                                        Expr::u32(head_dim),
                                    ),
                                    Expr::var("score_d"),
                                ),
                            ),
                        ),
                    ),
                )],
            ),
            Node::let_bind("raw_score", Expr::mul(Expr::var("dot_val"), scale)),
            Node::let_bind("score", bounded_score(Expr::var("raw_score"))),
            Node::store(
                "score_tile",
                score_idx(Expr::var("local"), Expr::var("tile_j")),
                Expr::var("score"),
            ),
        ],
    )]
}

/// Absorb one KV tile's values into the running accumulator.
fn absorb_tile_values(v: &str, head_dim: u32, tile_size: u32) -> Vec<Node> {
    let score_idx = |local: Expr, j: Expr| scratch_index(tile_size, local, j);
    let o_idx = |local: Expr, d: Expr| scratch_index(head_dim, local, d);
    vec![Node::loop_for(
        "out_d",
        Expr::u32(0),
        Expr::u32(head_dim),
        vec![
            Node::let_bind("weighted_v", Expr::f32(0.0)),
            Node::loop_for(
                "v_j",
                Expr::u32(0),
                Expr::var("tile_len"),
                vec![Node::assign(
                    "weighted_v",
                    Expr::add(
                        Expr::var("weighted_v"),
                        Expr::mul(
                            Expr::UnOp {
                                op: UnOp::Exp,
                                operand: Box::new(bounded_exp_arg(Expr::sub(
                                    Expr::load(
                                        "score_tile",
                                        score_idx(Expr::var("local"), Expr::var("v_j")),
                                    ),
                                    Expr::var("m_new"),
                                ))),
                            },
                            Expr::load(
                                v,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("tile_start"), Expr::var("v_j")),
                                        Expr::u32(head_dim),
                                    ),
                                    Expr::var("out_d"),
                                ),
                            ),
                        ),
                    ),
                )],
            ),
            Node::store(
                "o_acc",
                o_idx(Expr::var("local"), Expr::var("out_d")),
                Expr::add(
                    Expr::mul(
                        Expr::var("rescale"),
                        Expr::load("o_acc", o_idx(Expr::var("local"), Expr::var("out_d"))),
                    ),
                    Expr::var("weighted_v"),
                ),
            ),
        ],
    )]
}

/// The whole `[s, d]` online-softmax attention body for one work plan.
fn attention_body(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    plan: &FlashAttentionWorkPlan,
) -> Vec<Node> {
    let scale = Expr::f32(1.0f32 / (plan.head_dim as f32).sqrt());
    tiled_online_softmax_body(
        TiledOnlineSoftmaxSpec {
            q,
            out,
            item_var: "row",
            item_count: plan.seq_len,
            seq_len: plan.seq_len,
            head_dim: plan.head_dim,
            tile_size: plan.tile_size,
            tile_count: plan.tile_count,
        },
        tile_scores(k, plan.head_dim, plan.tile_size, scale),
        absorb_tile_values(v, plan.head_dim, plan.tile_size),
    )
}

/// Buffer table for a `[s, d]` online-softmax attention kernel.
fn attention_buffers(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    plan: &FlashAttentionWorkPlan,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32)
            .with_count(plan.logical_elements),
        BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
            .with_count(plan.logical_elements),
        BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32)
            .with_count(plan.logical_elements),
        BufferDecl::workgroup("q_scratch", plan.q_scratch_elements, DataType::F32),
        BufferDecl::workgroup("score_tile", plan.score_scratch_elements, DataType::F32),
        BufferDecl::workgroup("o_acc", plan.o_acc_scratch_elements, DataType::F32),
        BufferDecl::output(out, 3, DataType::F32).with_count(plan.logical_elements),
    ]
}

/// Online-softmax attention over `[s, d]` Q, K and V, as its own operation.
///
/// The tile width, lane count and scratch sizes all come from `plan`, so a
/// caller selects a schedule rather than restating the recurrence under it.
#[must_use]
pub fn online_softmax_attention(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    plan: &FlashAttentionWorkPlan,
) -> Program {
    Program::wrapped(
        attention_buffers(q, k, v, out, plan),
        [plan.workgroup_lanes, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            attention_body(q, k, v, out, plan),
        )],
    )
}

/// The same kernel, attributed to the operation that selected this plan.
///
/// The body is identical; the extra region records that `op_id` composed the
/// registered core instead of building a recurrence of its own, which is what
/// keeps the two entry points from drifting apart in the first place.
pub(super) fn compose_online_softmax_attention(
    op_id: &'static str,
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    plan: &FlashAttentionWorkPlan,
) -> Program {
    Program::wrapped(
        attention_buffers(q, k, v, out, plan),
        [plan.workgroup_lanes, 1, 1],
        vec![wrap_anonymous_region(
            op_id,
            vec![wrap_child_region(
                OP_ID,
                Ident::from(op_id),
                attention_body(q, k, v, out, plan),
            )],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            let plan = super::planner::plan_flash_attention_tiled(9, 1, 4)
                .expect("Fix: the registered online-softmax witness must plan");
            online_softmax_attention("q", "k", "v", "out", &plan)
        },
        // Zero Q and K make every key equally likely, so each row returns the
        // mean of V. The expectation is that arithmetic, written down, not a
        // second implementation of the kernel agreeing with itself.
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&[0.0_f32; 9]),
                vyre_primitives::wire::pack_f32_slice(&[0.0_f32; 9]),
                vyre_primitives::wire::pack_f32_slice(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ]]
        }),
        Some(|| {
            vec![vec![vyre_primitives::wire::pack_f32_slice(&[4.0_f32; 9])]]
        }),
    )
    .with_category("nn")
}
