//! Resident megakernel launch policy and queue-pressure decisions.

mod cache;
mod decision;
mod launch;
mod priority;

#[cfg(test)]
mod tests;

pub use decision::{
    ResidentExecutionMode, ResidentGraphBlasSwitchClass, ResidentLaunchCacheStats,
    ResidentLaunchRecommendation, ResidentLaunchRequest, ResidentPromotionEvidence,
    ResidentPromotionRoute, ResidentQueuePressure, ResidentQueueTopology, ResidentTopologyEvidence,
    HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION, TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
};
pub use launch::ResidentLaunchPolicy;
pub use priority::{
    try_diffuse_priority_across_siblings, try_diffuse_priority_across_siblings_into,
    PriorityDrainReason, PriorityDrainRecommendation, PriorityRequeueAccounting,
    PRIORITY_COUNTER_DRAIN_FIX, PRIORITY_COUNTER_DRAIN_HEADROOM,
};
