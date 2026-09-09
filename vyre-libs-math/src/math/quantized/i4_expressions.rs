//! Shared IR-expression helpers for quantized program builders.

use super::i4_packed_contract;

use vyre_foundation::ir::{Expr, Node};
use vyre_foundation::numeric::FieldTarget;

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
                    "i4_matvec_nibble",
                    i4_packed_contract().load_row_field(
                        weights_packed,
                        Expr::mul(Expr::var("i4_matvec_row"), Expr::u32(words_per_row)),
                        Expr::var("i4_matvec_col"),
                    ),
                ),
                Node::let_bind(
                    "i4_matvec_weight",
                    i4_packed_contract()
                        .decode_field(Expr::var("i4_matvec_nibble"), FieldTarget::Float32),
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
    lane_target: FieldTarget,
    final_store: Node,
) -> Vec<Node> {
    vec![Node::if_then(
        Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
        vec![
            Node::let_bind("i4_dot_acc", accumulator_zero),
            Node::loop_for(
                "i4_dot_lane",
                Expr::u32(0),
                Expr::u32(lane_count),
                vec![
                    Node::let_bind(
                        "i4_dot_lhs_nibble",
                        i4_packed_contract().load_field(lhs_packed, Expr::var("i4_dot_lane")),
                    ),
                    Node::let_bind(
                        "i4_dot_rhs_nibble",
                        i4_packed_contract().load_field(rhs_packed, Expr::var("i4_dot_lane")),
                    ),
                    Node::let_bind(
                        "i4_dot_lhs",
                        i4_packed_contract()
                            .decode_field(Expr::var("i4_dot_lhs_nibble"), lane_target),
                    ),
                    Node::let_bind(
                        "i4_dot_rhs",
                        i4_packed_contract()
                            .decode_field(Expr::var("i4_dot_rhs_nibble"), lane_target),
                    ),
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

/// The packed-activation INT4 inner product over `cols`, accumulating into the
/// binding `{prefix}_acc`.
///
/// Reads the weight row named by `{prefix}_row` and the activation row named by
/// `{prefix}_batch`, both packed under the module's INT4 contract with
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
                weight_nibble.clone(),
                i4_packed_contract().load_row_field(
                    weights_packed,
                    Expr::mul(Expr::var(row), Expr::u32(words_per_row)),
                    Expr::var(col.clone()),
                ),
            ),
            Node::let_bind(
                activation_nibble.clone(),
                i4_packed_contract().load_row_field(
                    activation_batches_packed,
                    Expr::mul(Expr::var(batch), Expr::u32(words_per_row)),
                    Expr::var(col),
                ),
            ),
            Node::let_bind(
                weight.clone(),
                i4_packed_contract().decode_field(Expr::var(weight_nibble), FieldTarget::Float32),
            ),
            Node::let_bind(
                activation.clone(),
                i4_packed_contract()
                    .decode_field(Expr::var(activation_nibble), FieldTarget::Float32),
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
