//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]

use crate::dual_volume;

#[test]
fn sweep_dual_bitwise_shift_left_volume_oracle_matrix() {
    dual_volume::assert_volume_oracle("primitive.bitwise.shift_left", |left, right| {
        left << (right & 31)
    });
}
