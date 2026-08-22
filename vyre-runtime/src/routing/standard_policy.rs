//! Standard routing policies for common compute workloads.

use super::{RoutingDecision, RoutingExplanation, RoutingPolicy};
use vyre_foundation::execution_plan::{ExecutionPlan, PolicyRoute, SchedulingPolicy};

/// Default megakernel-first release policy.
pub struct StandardPolicy;

impl RoutingPolicy for StandardPolicy {
    fn name(&self) -> &'static str {
        "standard-megakernel-first"
    }

    fn route(&self, plan: &ExecutionPlan) -> RoutingDecision {
        self.route_with_explanation(plan).decision
    }

    fn route_with_explanation(&self, plan: &ExecutionPlan) -> RoutingExplanation {
        // Every route the scheduling policy can suggest is served by the
        // persistent megakernel, so the decision does not branch; only the
        // evidence records which suggestion it started from. Asking the policy
        // rather than answering `PersistentMegakernel` outright keeps the
        // explanation honest about what was consulted.
        let suggested = SchedulingPolicy::standard().route(plan.fusion.node_count);
        RoutingExplanation {
            policy: self.name(),
            decision: RoutingDecision::PersistentMegakernel,
            reason: if suggested == PolicyRoute::PersistentMegakernel {
                "scheduling policy selected persistent megakernel directly"
            } else {
                "standard policy promotes the non-persistent suggestion to persistent megakernel for resident execution"
            },
        }
    }
}
