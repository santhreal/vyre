//! Shared IR-expression helpers for quantized program builders.

use super::I4_LANES_PER_WORD;

use vyre_foundation::ir::{DataType, Expr, Node};

pub(super) fn i4_matvec_scaled_body(
    weights_packed: &str,
    x: &str,
    row_scales: &str,
    out: &str,
    cols: u32,
    words_per_row: u32,
    row: Expr,
    x_base: Expr,
    out_index: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("i4_matvec_row", row),
        Node::let_bind("i4_matvec_x_base", x_base),
        Node::let_bind("i4_matvec_out_index", out_index),
        Node::let_bind("i4_matvec_acc", Expr::f32(0.0)),
        Node::loop_for(
            "i4_matvec_col",
            Expr::u32(0),
            Expr::u32(cols),
            vec![
                Node::let_bind(
                    "i4_matvec_word_index",
                    Expr::add(
                        Expr::mul(Expr::var("i4_matvec_row"), Expr::u32(words_per_row)),
                        Expr::div(Expr::var("i4_matvec_col"), Expr::u32(I4_LANES_PER_WORD)),
                    ),
                ),
                Node::let_bind(
                    "i4_matvec_shift",
                    Expr::mul(
                        Expr::rem(Expr::var("i4_matvec_col"), Expr::u32(I4_LANES_PER_WORD)),
                        Expr::u32(4),
                    ),
                ),
                Node::let_bind(
                    "i4_matvec_nibble",
                    Expr::bitand(
                        Expr::shr(
                            Expr::load(weights_packed, Expr::var("i4_matvec_word_index")),
                            Expr::var("i4_matvec_shift"),
                        ),
                        Expr::u32(0xF),
                    ),
                ),
                Node::let_bind(
                    "i4_matvec_weight",
                    signed_i4_nibble_f32_expr(Expr::var("i4_matvec_nibble")),
                ),
                Node::let_bind(
                    "i4_matvec_x_index",
                    Expr::add(Expr::var("i4_matvec_x_base"), Expr::var("i4_matvec_col")),
                ),
                Node::assign(
                    "i4_matvec_acc",
                    Expr::add(
                        Expr::var("i4_matvec_acc"),
                        Expr::mul(
                            Expr::var("i4_matvec_weight"),
                            Expr::load(x, Expr::var("i4_matvec_x_index")),
                        ),
                    ),
                ),
            ],
        ),
        Node::store(
            out,
            Expr::var("i4_matvec_out_index"),
            Expr::mul(
                Expr::var("i4_matvec_acc"),
                Expr::load(row_scales, Expr::var("i4_matvec_row")),
            ),
        ),
    ]
}

pub(super) fn i4_dot_accumulation_body(
    lhs_packed: &str,
    rhs_packed: &str,
    lane_count: u32,
    accumulator_zero: Expr,
    lane_value: fn(Expr) -> Expr,
    final_store: Node,
) -> Vec<Node> {
    vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![
            Node::let_bind("i4_dot_acc", accumulator_zero),
            Node::loop_for(
                "i4_dot_lane",
                Expr::u32(0),
                Expr::u32(lane_count),
                vec![
                    Node::let_bind(
                        "i4_dot_word_index",
                        Expr::div(Expr::var("i4_dot_lane"), Expr::u32(I4_LANES_PER_WORD)),
                    ),
                    Node::let_bind(
                        "i4_dot_shift",
                        Expr::mul(
                            Expr::rem(Expr::var("i4_dot_lane"), Expr::u32(I4_LANES_PER_WORD)),
                            Expr::u32(4),
                        ),
                    ),
                    Node::let_bind(
                        "i4_dot_lhs_nibble",
                        Expr::bitand(
                            Expr::shr(
                                Expr::load(lhs_packed, Expr::var("i4_dot_word_index")),
                                Expr::var("i4_dot_shift"),
                            ),
                            Expr::u32(0xF),
                        ),
                    ),
                    Node::let_bind(
                        "i4_dot_rhs_nibble",
                        Expr::bitand(
                            Expr::shr(
                                Expr::load(rhs_packed, Expr::var("i4_dot_word_index")),
                                Expr::var("i4_dot_shift"),
                            ),
                            Expr::u32(0xF),
                        ),
                    ),
                    Node::let_bind("i4_dot_lhs", lane_value(Expr::var("i4_dot_lhs_nibble"))),
                    Node::let_bind("i4_dot_rhs", lane_value(Expr::var("i4_dot_rhs_nibble"))),
                    Node::assign(
                        "i4_dot_acc",
                        Expr::add(
                            Expr::var("i4_dot_acc"),
                            Expr::mul(Expr::var("i4_dot_lhs"), Expr::var("i4_dot_rhs")),
                        ),
                    ),
                ],
            ),
            final_store,
        ],
    )]
}

pub(super) fn signed_i4_nibble_expr(nibble: Expr) -> Expr {
    Expr::select(
        Expr::eq(Expr::bitand(nibble.clone(), Expr::u32(0x8)), Expr::u32(0)),
        Expr::cast(DataType::I32, nibble.clone()),
        Expr::sub(Expr::cast(DataType::I32, nibble), Expr::i32(16)),
    )
}

pub(super) fn signed_i4_nibble_f32_expr(nibble: Expr) -> Expr {
    Expr::select(
        Expr::eq(Expr::bitand(nibble.clone(), Expr::u32(0x8)), Expr::u32(0)),
        Expr::cast(DataType::F32, nibble.clone()),
        Expr::sub(Expr::cast(DataType::F32, nibble), Expr::f32(16.0)),
    )
}

/// The packed-activation INT4 inner product over `cols`, accumulating into the
/// binding `{prefix}_acc`.
///
/// Reads the weight row named by `{prefix}_row` and the activation row named by
/// `{prefix}_batch`, both packed [`I4_LANES_PER_WORD`] lanes per word with
/// `words_per_row` words per row. `prefix` names every binding the loop opens
/// so a schedule that nests this inside a row scan keeps its own accumulator.
pub(super) fn i4_packed_dot_loop(
    prefix: &str,
    weights_packed: &str,
    activation_batches_packed: &str,
    cols: u32,
    words_per_row: u32,
) -> Node {
    let row = format!("{prefix}_row");
    let batch = format!("{prefix}_batch");
    let col = format!("{prefix}_col");
    let acc = format!("{prefix}_acc");
    let weight_word = format!("{prefix}_weight_word");
    let activation_word = format!("{prefix}_activation_word");
    let shift = format!("{prefix}_shift");
    let weight_nibble = format!("{prefix}_weight_nibble");
    let activation_nibble = format!("{prefix}_activation_nibble");
    let weight = format!("{prefix}_weight");
    let activation = format!("{prefix}_activation");
    Node::loop_for(
        col.clone(),
        Expr::u32(0),
        Expr::u32(cols),
        vec![
            Node::let_bind(
                weight_word.clone(),
                Expr::add(
                    Expr::mul(Expr::var(row), Expr::u32(words_per_row)),
                    Expr::div(Expr::var(col.clone()), Expr::u32(I4_LANES_PER_WORD)),
                ),
            ),
            Node::let_bind(
                activation_word.clone(),
                Expr::add(
                    Expr::mul(Expr::var(batch), Expr::u32(words_per_row)),
                    Expr::div(Expr::var(col.clone()), Expr::u32(I4_LANES_PER_WORD)),
                ),
            ),
            Node::let_bind(
                shift.clone(),
                Expr::mul(
                    Expr::rem(Expr::var(col), Expr::u32(I4_LANES_PER_WORD)),
                    Expr::u32(4),
                ),
            ),
            Node::let_bind(
                weight_nibble.clone(),
                Expr::bitand(
                    Expr::shr(
                        Expr::load(weights_packed, Expr::var(weight_word)),
                        Expr::var(shift.clone()),
                    ),
                    Expr::u32(0xF),
                ),
            ),
            Node::let_bind(
                activation_nibble.clone(),
                Expr::bitand(
                    Expr::shr(
                        Expr::load(activation_batches_packed, Expr::var(activation_word)),
                        Expr::var(shift),
                    ),
                    Expr::u32(0xF),
                ),
            ),
            Node::let_bind(
                weight.clone(),
                signed_i4_nibble_f32_expr(Expr::var(weight_nibble)),
            ),
            Node::let_bind(
                activation.clone(),
                signed_i4_nibble_f32_expr(Expr::var(activation_nibble)),
            ),
            Node::assign(
                acc.clone(),
                Expr::add(
                    Expr::var(acc),
                    Expr::mul(Expr::var(weight), Expr::var(activation)),
                ),
            ),
        ],
    )
}

/// The dequantized score for one `(row, batch)` pair: the accumulator scaled by
/// its row scale and then by its batch scale.
pub(super) fn i4_packed_scaled_score(prefix: &str, row_scales: &str, batch_scales: &str) -> Expr {
    Expr::mul(
        Expr::mul(
            Expr::var(format!("{prefix}_acc")),
            Expr::load(row_scales, Expr::var(format!("{prefix}_row"))),
        ),
        Expr::load(batch_scales, Expr::var(format!("{prefix}_batch"))),
    )
}
