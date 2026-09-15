use std::time::Duration;

pub(super) const QUIESCE_SPIN_POLLS: u64 = 64;
pub(super) const QUIESCE_MIN_PARK: Duration = Duration::from_micros(2);
pub(super) const QUIESCE_MAX_PARK: Duration = Duration::from_micros(50);
pub(super) const QUIESCE_BACKOFF_SHIFT_CAP: u64 = 5;

#[allow(clippy::unnecessary_min_or_max)]
pub(super) fn quiesce_backoff_duration(poll: u64) -> Duration {
    let parked_poll = poll.saturating_sub(QUIESCE_SPIN_POLLS);
    let shift = parked_poll.min(QUIESCE_BACKOFF_SHIFT_CAP) as u32;
    let multiplier = 1_u32 << shift;
    QUIESCE_MIN_PARK
        .checked_mul(multiplier)
        .unwrap_or(QUIESCE_MAX_PARK)
        .min(QUIESCE_MAX_PARK)
}

pub(super) fn quiesce_idle(poll: u64) {
    if poll < QUIESCE_SPIN_POLLS {
        std::hint::spin_loop();
    } else {
        std::thread::park_timeout(quiesce_backoff_duration(poll));
    }
}

pub(super) fn tenant_registry_retry_idle(retry: u64) {
    if retry < QUIESCE_SPIN_POLLS {
        std::hint::spin_loop();
    } else {
        std::thread::park_timeout(quiesce_backoff_duration(retry));
    }
}
