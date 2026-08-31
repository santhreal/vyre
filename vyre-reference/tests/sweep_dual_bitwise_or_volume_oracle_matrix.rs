//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]

mod dual_volume;

#[test]
fn sweep_dual_bitwise_or_volume_oracle_matrix() {
    dual_volume::assert_volume_oracle("primitive.bitwise.or", |left, right| left | right);
}
