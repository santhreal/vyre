//! Semantic ownership, placement, and consolidation closure for registered operations.
//!
//! Composition is judged from the live program and registration metadata. Source
//! similarity remains useful for discovery, but it never establishes ownership.

use super::*;

/// Judge semantic ownership in both directions: every attributed child must
/// exist, and every operation with the same semantic body must have one owner.
pub(super) fn check_semantic_organization(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[11/11] Semantic ownership, placement, and consolidation".to_string());
    let mut findings = Vec::new();
    let known = ops
        .iter()
        .map(|op| op.id.as_str())
        .collect::<BTreeSet<_>>();

    for op in ops {
        check_source_placement(op, &mut findings);
        for node in op.program.entry() {
            check_attribution(op, node, &known, &mut findings);
        }
    }

    for (index, left) in ops.iter().enumerate() {
        for right in ops.iter().skip(index + 1) {
            check_pair(left, right, &mut findings);
        }
    }

    let count = findings.len();
    for finding in findings {
        report.find(finding);
    }
    if count == 0 {
        report.note("  semantic ownership is closed in both directions".to_string());
    }
    count
}

fn check_source_placement(op: &OpInfo, findings: &mut Vec<Finding>) {
    let Some((_owner, declared_domain)) = operation_owner(&op.id) else {
        findings.push(Finding::new(
            format!("operation `{}` has no crate and domain namespace", op.id),
            "name it `<owner-crate>::<domain>::<operation>` so its canonical owner is mechanically decidable",
        ));
        return;
    };

    if op.source_file.is_empty() || op.source_file == "<unattributed>" {
        findings.push(Finding::new(
            format!("operation `{}` has no registration source attribution", op.id),
            "construct the registration through the track-caller constructor so the registry records its owning source file",
        ));
        return;
    }

    let normalized = op.source_file.replace('\\', "/");
    let Some((owning_crate, rest)) = normalized
        .split_once("vyre-libs/src/")
        .map(|(_, rest)| ("vyre-libs", rest))
        .or_else(|| {
            normalized
                .split_once("vyre-primitives/src/")
                .map(|(_, rest)| ("vyre-primitives", rest))
        })
    else {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` registration is outside `vyre-libs/src/` or `vyre-primitives/src/`",
                op.id
            ),
            "move the registration into the owning crate's source tree",
        ));
        return;
    };

    if op.tier == Tier::T2 && owning_crate != "vyre-primitives" {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "Category C hardware operation `{}` must live in `vyre-primitives/src/hardware/`",
                op.id
            ),
            "move the Category C hardware operation to `vyre-primitives/src/hardware/`",
        ));
        return;
    }

    let segments: Vec<&str> = rest
        .split('/')
        .map(|segment| segment.trim_end_matches(".rs"))
        .collect();
    let matches_domain = segments.contains(&declared_domain)
        || (declared_domain == "matching" && segments.contains(&"scan"))
        || (declared_domain == "quant" && segments.contains(&"nn"))
        || (declared_domain == "optim" && segments.contains(&"nn"));

    if !matches_domain {
        let source_domain = segments.first().copied().unwrap_or_default();
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` declares domain `{declared_domain}` but its registration lives in domain `{source_domain}`",
                op.id
            ),
            format!(
                "move the semantic owner to `{owning_crate}/src/{declared_domain}/`, or rename the operation into `{owning_crate}::{source_domain}::*` when its effects and contract are domain-specific"
            ),
        ));
    }
}

fn operation_owner(id: &str) -> Option<(&str, &str)> {
    let mut segments = id.split("::");
    let owner = segments.next()?;
    let domain = segments.next()?;
    (!owner.is_empty() && !domain.is_empty()).then_some((owner, domain))
}

fn check_attribution(
    op: &OpInfo,
    node: &Node,
    known: &BTreeSet<&str>,
    findings: &mut Vec<Finding>,
) {
    if let Node::Region {
        source_region: Some(parent),
        generator,
        ..
    } = node
    {
        let child = generator.as_str();
        if !known.contains(child)
            && !vyre_foundation::composition::is_anonymous_generator(child)
        {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` attributes a composed region to unregistered child `{child}`",
                    op.id
                ),
                "register the child semantic owner and compose it by operation id, or mark the region anonymous when it owns no reusable operation",
            ));
        }
        if parent.as_str() != op.id && !known.contains(parent.as_str()) {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` carries unknown composition parent `{}`",
                    op.id,
                    parent.as_str()
                ),
                "preserve the registered parent operation id when transplanting the child region",
            ));
        }
    }
    for body in vyre_foundation::visit::child_bodies(node) {
        for child in body {
            check_attribution(op, child, known, findings);
        }
    }
}

fn check_pair(left: &OpInfo, right: &OpInfo, findings: &mut Vec<Finding>) {
    if left.children.contains(&right.id) || right.children.contains(&left.id) {
        return;
    }

    if left.semantic_fingerprint == right.semantic_fingerprint {
        findings.push(consolidation_finding(
            left,
            right,
            "have byte-identical canonical programs after erasing only the owner id",
        ));
    }
}

fn consolidation_finding(left: &OpInfo, right: &OpInfo, evidence: &str) -> Finding {
    let left_domain = operation_owner(&left.id).map(|(_, domain)| domain);
    let right_domain = operation_owner(&right.id).map(|(_, domain)| domain);
    let direction = if left_domain == right_domain {
        "keep one parameterized semantic owner in that domain and make every caller compose it"
    } else if left.tier == Tier::T3 && right.tier == Tier::T3 {
        "promote the shared semantic body to the lowest common substrate domain and make both domains compose it"
    } else {
        "keep the lowest-level canonical owner and make the higher-level operation compose it"
    };
    Finding::new(
        format!(
            "operations `{}` and `{}` {evidence}, but neither composes the other",
            left.id, right.id
        ),
        direction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BufferDecl, DataType};

    fn fixture(id: &'static str, value: u32) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(value))],
            ),
        )
    }
    fn fixture_distinct(id: &'static str) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![
                    BufferDecl::read("in", 0, DataType::U32).with_count(1),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::load("in", Expr::u32(0)))],
            ),
        )
    }

    /// WHY: exact semantic duplicates are the non-heuristic consolidation class.
    /// A same-domain copy and a cross-domain copy must both fail; differences in
    /// the literal body remain outside this exact-identity assertion.
    #[test]
    fn exact_semantic_duplicates_require_one_owner_in_every_domain_arrangement() {
        let same_domain_left = fixture("vyre-libs::math::left", 7);
        let same_domain_right = fixture("vyre-libs::math::right", 7);
        let cross_domain = fixture("vyre-libs::graph::right", 7);

        let mut findings = Vec::new();
        check_pair(&same_domain_left, &same_domain_right, &mut findings);
        check_pair(&same_domain_left, &cross_domain, &mut findings);
        assert_eq!(findings.len(), 2);
        assert!(findings[0].fix.contains("parameterized semantic owner"));
        assert!(findings[1].fix.contains("common substrate domain"));

        let distinct = fixture_distinct("vyre-libs::math::distinct");
        findings.clear();
        check_pair(&same_domain_left, &distinct, &mut findings);
        assert!(findings.is_empty());
    }

    #[test]
    fn source_domain_must_equal_registered_domain() {
        let mut op = fixture("vyre-libs::math::sum", 1);
        op.source_file = "vyre-libs/src/nn/sum.rs".to_string();
        let mut findings = Vec::new();
        check_source_placement(&op, &mut findings);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("declares domain `math`"));
    }
}
