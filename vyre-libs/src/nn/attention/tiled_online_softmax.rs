//! The tiled online-softmax skeleton shared by the attention decoders.
//!
//! [`flash_attention_2`](super::flash_attention_2::flash_attention_2) and
//! [`mla_decode`](super::mla::mla_decode) run the identical `(m, l, o_acc)`
//! recurrence over identical tiling; they differ only in how a tile's scores
//! are produced and how the accumulator absorbs the tile's values. Those two
//! op-specific fragments are the parameters of
//! [`tiled_online_softmax_body`]; everything around them lives here once, so a
//! numerical-stability change cannot land in one decoder and miss the other.

use vyre_foundation::ir::{Expr, Node, UnOp};

use crate::nn::attention_stability::{bounded_exp_arg, flush_tiny, positive_denominator};

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
