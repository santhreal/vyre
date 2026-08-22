//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]

mod dual_volume;

#[test]
fn sweep_dual_compare_eq_volume_oracle_matrix() {
    dual_volume::assert_volume_oracle("primitive.compare.eq", |left, right| {
        u32::from(left == right)
    });
}
