//! Standard routing policies for common compute workloads.

use super::{RoutingDecision, RoutingExplanation, RoutingPolicy};
use vyre_foundation::execution_plan::ExecutionPlan;

/// Default megakernel-first release policy.
pub struct StandardPolicy;

impl RoutingPolicy for StandardPolicy {
    fn name(&self) -> &'static str {
        "standard-megakernel-first"
    }

    fn route(&self, plan: &ExecutionPlan) -> RoutingDecision {
        self.route_with_explanation(plan).decision
    }

    fn route_with_explanation(&self, _plan: &ExecutionPlan) -> RoutingExplanation {
        // Every production compile emits a megakernel artifact, so there is one
        // route and nothing to consult. This used to ask a foundation policy
        // predicate that ignored its argument and always answered the same
        // route, and recorded the answer as evidence of a decision.
        RoutingExplanation {
            policy: self.name(),
            decision: RoutingDecision::PersistentMegakernel,
            reason: "every production compile emits a megakernel artifact",
        }
    }
}
