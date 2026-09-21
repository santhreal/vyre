use vyre_driver::BackendError;

use super::launch::reserve_target_capacity;

/// Requeue and aging counters produced by priority-aware schedulers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PriorityRequeueAccounting {
    /// Number of slots requeued due to contention or quota pressure.
    pub requeue_count: u64,
    /// Number of slots promoted because their priority age crossed policy.
    pub aged_promotions: u64,
    /// Largest age observed for any queued priority slot.
    pub max_priority_age: u32,
}

/// Counter headroom at or below which schedulers should drain telemetry.
pub const PRIORITY_COUNTER_DRAIN_HEADROOM: u64 = 1024;

/// Stable operator fix for priority counter drain recommendations.
pub const PRIORITY_COUNTER_DRAIN_FIX: &str =
    "drain scheduler telemetry before counters reach u64::MAX";

/// Reason a priority scheduler should drain telemetry into a launch request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityDrainReason {
    /// No priority telemetry is pending.
    None,
    /// Non-empty priority telemetry should be propagated to the policy.
    PendingTelemetry,
    /// The requeue counter is inside the configured drain headroom.
    RequeueCounterNearLimit,
    /// The aged-promotion counter is inside the configured drain headroom.
    AgedPromotionCounterNearLimit,
    /// The requeue counter is exhausted.
    RequeueCounterExhausted,
    /// The aged-promotion counter is exhausted.
    AgedPromotionCounterExhausted,
}

impl PriorityDrainReason {
    /// Stable label for tests, reports, and scheduler diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PendingTelemetry => "pending_telemetry",
            Self::RequeueCounterNearLimit => "requeue_counter_near_limit",
            Self::AgedPromotionCounterNearLimit => "aged_promotion_counter_near_limit",
            Self::RequeueCounterExhausted => "requeue_counter_exhausted",
            Self::AgedPromotionCounterExhausted => "aged_promotion_counter_exhausted",
        }
    }
}

/// Structured drain recommendation for priority scheduler counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityDrainRecommendation {
    /// True when the scheduler should drain telemetry before accepting more work.
    pub should_drain: bool,
    /// Concrete reason for the recommendation.
    pub reason: PriorityDrainReason,
    /// Requeue counter value included for propagation into launch telemetry.
    pub requeue_count: u64,
    /// Aged-promotion counter value included for propagation into launch telemetry.
    pub aged_promotions: u64,
    /// Largest priority age observed for any queued slot.
    pub max_priority_age: u32,
    /// Remaining requeue counter increments before exact overflow.
    pub requeue_counter_headroom: u64,
    /// Remaining aged-promotion counter increments before exact overflow.
    pub aged_promotion_counter_headroom: u64,
    /// Stable operator fix string to surface with drain diagnostics.
    pub fix: &'static str,
}

impl PriorityRequeueAccounting {
    /// Return a structured drain recommendation for scheduler telemetry.
    #[must_use]
    pub fn drain_recommendation(self) -> PriorityDrainRecommendation {
        let requeue_counter_headroom = u64::MAX.saturating_sub(self.requeue_count);
        let aged_promotion_counter_headroom = u64::MAX.saturating_sub(self.aged_promotions);
        let reason = if self.requeue_count == u64::MAX {
            PriorityDrainReason::RequeueCounterExhausted
        } else if self.aged_promotions == u64::MAX {
            PriorityDrainReason::AgedPromotionCounterExhausted
        } else if requeue_counter_headroom <= PRIORITY_COUNTER_DRAIN_HEADROOM {
            PriorityDrainReason::RequeueCounterNearLimit
        } else if aged_promotion_counter_headroom <= PRIORITY_COUNTER_DRAIN_HEADROOM {
            PriorityDrainReason::AgedPromotionCounterNearLimit
        } else if self.requeue_count != 0 || self.aged_promotions != 0 || self.max_priority_age != 0
        {
            PriorityDrainReason::PendingTelemetry
        } else {
            PriorityDrainReason::None
        };
        PriorityDrainRecommendation {
            should_drain: reason != PriorityDrainReason::None,
            reason,
            requeue_count: self.requeue_count,
            aged_promotions: self.aged_promotions,
            max_priority_age: self.max_priority_age,
            requeue_counter_headroom,
            aged_promotion_counter_headroom,
            fix: PRIORITY_COUNTER_DRAIN_FIX,
        }
    }

    /// Record one requeue event.
    pub fn record_requeue(&mut self, age_ticks: u32) {
        self.requeue_count = self.requeue_count.saturating_add(1);
        self.max_priority_age = self.max_priority_age.max(age_ticks);
    }

    /// Record one requeue event with exact overflow reporting.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the requeue counter would overflow.
    pub fn try_record_requeue(&mut self, age_ticks: u32) -> Result<(), BackendError> {
        self.requeue_count = self.requeue_count.checked_add(1).ok_or_else(|| {
            BackendError::new(
                "megakernel priority requeue_count overflowed u64. Fix: drain scheduler telemetry before counters reach u64::MAX.",
            )
        })?;
        self.max_priority_age = self.max_priority_age.max(age_ticks);
        Ok(())
    }

    /// Record one priority-aging promotion.
    pub fn record_aged_promotion(&mut self, age_ticks: u32) {
        self.aged_promotions = self.aged_promotions.saturating_add(1);
        self.max_priority_age = self.max_priority_age.max(age_ticks);
    }

    /// Record one priority-aging promotion with exact overflow reporting.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the aged-promotion counter would overflow.
    pub fn try_record_aged_promotion(&mut self, age_ticks: u32) -> Result<(), BackendError> {
        self.aged_promotions = self.aged_promotions.checked_add(1).ok_or_else(|| {
            BackendError::new(
                "megakernel aged_promotions overflowed u64. Fix: drain scheduler telemetry before counters reach u64::MAX.",
            )
        })?;
        self.max_priority_age = self.max_priority_age.max(age_ticks);
        Ok(())
    }
}

/// Diffuse priority signals across a set of priority-class siblings
/// via sheaf diffusion (P-RUNTIME-3). Higher-priority siblings pull
/// neighbors toward higher priority; lower-priority siblings drag
/// down. After a few diffusion steps, each item's priority reflects
/// both its own age and its neighborhood pressure  -  letting requeue
/// decisions be group-aware without hand-rolling a propagation pass.
///
/// `priority_stalks` is the per-item priority value (caller's choice
/// of scale; higher = more urgent). `restriction_diag` is the
/// per-item transmission coefficient (1.0 = freely shares priority,
/// 0.0 = isolated). `damping` controls the diffusion rate in [0, 1].
///
/// Returns the post-diffusion priority vector, same shape as input.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
) -> Result<Vec<f64>, BackendError> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    try_diffuse_priority_across_siblings_into(
        priority_stalks,
        restriction_diag,
        damping,
        iterations,
        &mut current,
        &mut next,
    )?;
    Ok(current)
}

/// Diffuse priority signals into caller-owned storage.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings_into(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, priority_stalks.len(), "priority diffusion output")?;
    out.extend_from_slice(priority_stalks);
    scratch.clear();
    if priority_stalks.len() != restriction_diag.len() {
        return Ok(());
    }
    for _ in 0..iterations {
        diffuse_step_into(out, restriction_diag, damping, scratch)?;
        std::mem::swap(out, scratch);
    }
    Ok(())
}

fn diffuse_step_into(
    stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    out: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, stalks.len(), "priority diffusion scratch")?;
    out.resize(stalks.len(), 0.0);
    for ((slot, &stalk), &restriction) in out
        .iter_mut()
        .zip(stalks.iter())
        .zip(restriction_diag.iter())
    {
        *slot = stalk - damping * restriction * stalk;
    }
    Ok(())
}
