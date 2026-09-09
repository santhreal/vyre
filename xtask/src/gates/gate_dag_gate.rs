//! Enforces DAG schema consistency, acyclicity, and prerequisite closure across all registered gates.

use crate::gate::{Coverage, Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gate_dag::GateDag;
use crate::gate_metadata::GATE_METADATA;

/// Enforces that the registered gate DAG is acyclic, with valid prerequisites and input paths.
pub struct GateDagGate;

impl GateBehavior for GateDagGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::default();
        let identities: Vec<String> = GATE_METADATA.iter().map(|d| d.name.to_string()).collect();
        report.cover(Coverage::complete_identities(
            "registered gate DAG",
            identities,
        ));

        let dag = match GateDag::from_descriptors(GATE_METADATA) {
            Ok(d) => d,
            Err(err) => {
                report.find(Finding::new(
                    format!("gate DAG construction failed: {err}"),
                    "fix prerequisite declarations or remove dependency cycles in GATE_METADATA",
                ));
                return Ok(report);
            }
        };

        for failure in dag.validate(&ctx.root) {
            report.find(Finding::new(
                failure,
                "ensure all declared gate prerequisites exist and declared inputs are valid",
            ));
        }

        report.note(format!(
            "Validated DAG across {} registered gates ({} dependency edges)",
            dag.len(),
            dag.edge_count()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_dag_validates_live_metadata() {
        let root = crate::checkout::checkout_root();
        let ctx = GateCtx::new(root, Vec::new());
        let report = GateDagGate.run(&ctx).expect("GateDagGate must run");
        assert_eq!(
            report.count(),
            0,
            "gate-dag reported findings: {:?}",
            report.findings
        );
    }
}
