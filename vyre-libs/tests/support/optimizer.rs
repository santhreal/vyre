//! Shared optimizer contract assertions.

use vyre_foundation::ir::Program;
use vyre_foundation::optimizer::optimize;

/// Assert the registered optimizer reaches a fixed point on `program`: a second
/// pass over its own output must change nothing.
pub(crate) fn assert_optimizer_is_idempotent(program: Program) {
    let optimized_once = optimize(program).expect("registered optimizer must converge");
    let optimized_twice =
        optimize(optimized_once.clone()).expect("registered optimizer must converge");
    assert_eq!(
        optimized_once,
        optimized_twice,
        "{}",
        first_debug_difference(&optimized_once, &optimized_twice)
    );
}

/// Where two debug renderings first diverge, with surrounding context from each
/// side so a mismatch names the node that moved.
fn first_debug_difference(left: &impl std::fmt::Debug, right: &impl std::fmt::Debug) -> String {
    let left = format!("{left:?}");
    let right = format!("{right:?}");
    let first_diff = left
        .bytes()
        .zip(right.bytes())
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| left.len().min(right.len()));
    let left_start = first_diff.saturating_sub(240);
    let right_start = first_diff.saturating_sub(240);
    let left_end = left.len().min(first_diff + 520);
    let right_end = right.len().min(first_diff + 520);
    format!(
        "first debug diff at byte {first_diff}\nleft: {}\nright: {}",
        &left[left_start..left_end],
        &right[right_start..right_end],
    )
}
