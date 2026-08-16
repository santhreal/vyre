//! Check 3: a Tier 2.5 primitive has callers, or a reviewed exception.
//!
//! A primitive nobody calls is not a primitive, it is an operation that was
//! written speculatively. The admission registry records the exceptions with an
//! owner and a reason, and this check holds every family to one or the other.

use super::*;

pub(super) const PRIMITIVE_ADMISSION_PATH: &str = "docs/optimization/PRIMITIVE_ADMISSION.toml";

#[derive(Debug, serde::Deserialize)]
pub(super) struct PrimitiveAdmissionRegistry {
    schema_version: u32,
    minimum_independent_callers: usize,
    #[serde(default)]
    exception: Vec<PrimitiveAdmissionException>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct PrimitiveAdmissionException {
    family: String,
    owner: String,
    reason: String,
    review_boundary: String,
}

pub(super) const MIN_CALLERS_FOR_PRIMITIVE: usize = 2;

pub(super) fn is_synthetic_catalog_consumer(op_id: &str) -> bool {
    op_id.starts_with("vyre-libs::catalog::")
}

pub(super) fn primitive_caller_counts(ops: &[OpInfo]) -> HashMap<String, usize> {
    let mut caller_counts = HashMap::new();
    for op in ops
        .iter()
        .filter(|op| !is_synthetic_catalog_consumer(&op.id))
    {
        for child in &op.children {
            if tier_of(child) == Tier::T2_5 {
                *caller_counts.entry(child.clone()).or_insert(0) += 1;
            }
        }
    }
    caller_counts
}

pub(super) fn primitive_family(op_id: &str) -> Option<&str> {
    op_id
        .strip_prefix("vyre-primitives::")
        .and_then(|suffix| suffix.split("::").next())
}

pub(super) fn load_primitive_admission_registry() -> Result<PrimitiveAdmissionRegistry, String> {
    let root = workspace_root().ok_or_else(|| "workspace root is unavailable".to_string())?;
    let path = root.join(PRIMITIVE_ADMISSION_PATH);
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let registry: PrimitiveAdmissionRegistry =
        toml::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
    if registry.schema_version != 1 {
        return Err(format!(
            "{} must declare schema_version = 1",
            path.display()
        ));
    }
    if registry.minimum_independent_callers != MIN_CALLERS_FOR_PRIMITIVE {
        return Err(format!(
            "{} minimum_independent_callers={} disagrees with the audit floor {}",
            path.display(),
            registry.minimum_independent_callers,
            MIN_CALLERS_FOR_PRIMITIVE
        ));
    }
    Ok(registry)
}

pub(super) fn validate_primitive_admission(
    report: &mut Report,
    ops: &[OpInfo],
    caller_counts: &HashMap<String, usize>,
    registry: PrimitiveAdmissionRegistry,
) -> (usize, usize) {
    let mut flagged = 0usize;
    let mut exceptions = BTreeMap::new();
    for exception in registry.exception {
        if exception.family.trim().is_empty()
            || exception.owner.trim().is_empty()
            || exception.reason.trim().is_empty()
            || exception.review_boundary.trim().is_empty()
        {
            report.find(violation(format!("  ✗ primitive admission exception `{}` has an empty family, owner, reason, or review boundary",
                exception.family)));
            flagged += 1;
            continue;
        }
        if exceptions
            .insert(exception.family.clone(), exception)
            .is_some()
        {
            report.find(violation(
                "  ✗ duplicate primitive admission exception family".to_string(),
            ));
            flagged += 1;
        }
    }

    let mut under_adopted_families = BTreeSet::new();
    for op in ops {
        if op.tier != Tier::T2_5 {
            continue;
        }
        let callers = caller_counts.get(&op.id).copied().unwrap_or(0);
        if callers >= MIN_CALLERS_FOR_PRIMITIVE {
            continue;
        }
        let Some(family) = primitive_family(&op.id) else {
            report.find(violation(format!("  ✗ {} has no canonical primitive family. Fix: use `vyre-primitives::<family>::...`.",
                op.id)));
            flagged += 1;
            continue;
        };
        under_adopted_families.insert(family.to_string());
        if !exceptions.contains_key(family) {
            report.find(violation(format!("  ✗ {} has only {} caller(s) and family `{family}` has no owner-reviewed exception in {PRIMITIVE_ADMISSION_PATH}.",
                op.id, callers)));
            flagged += 1;
        }
    }

    for family in exceptions.keys() {
        if !under_adopted_families.contains(family) {
            report.find(violation(format!("  ✗ primitive admission exception `{family}` is stale because every family member meets the caller floor")));
            flagged += 1;
        }
    }
    (flagged, under_adopted_families.len())
}

/// Check 3: every Tier 2.5 primitive needs at least two independent callers
/// or an explicit owner-reviewed exception for its current family.
pub(super) fn check_3_primitive_coverage(report: &mut Report, ops: &[OpInfo]) -> usize {
    let mut flagged = 0usize;
    let mut exceptions_used = 0usize;
    report.note(format!("[3/10] Primitive coverage (Tier 2.5 primitives need ≥ {MIN_CALLERS_FOR_PRIMITIVE} callers)"));
    for op in ops
        .iter()
        .filter(|op| is_synthetic_catalog_consumer(&op.id))
    {
        report.find(violation(format!("  ✗ {} is a synthetic catalog consumer. Fix: exercise the primitive directly and record only product composition edges.",
            op.id)));
        flagged += 1;
    }

    let registry = match load_primitive_admission_registry() {
        Ok(registry) => registry,
        Err(error) => {
            report.find(violation(format!(
                "  ✗ primitive admission registry is invalid: {error}"
            )));
            return flagged + 1;
        }
    };
    let caller_counts = primitive_caller_counts(ops);
    let (admission_failures, reviewed_families) =
        validate_primitive_admission(report, ops, &caller_counts, registry);
    flagged += admission_failures;
    exceptions_used += reviewed_families;
    if flagged == 0 {
        report.note(format!("  ✓ no synthetic consumers; under-adopted primitives are covered by {exceptions_used} owner-reviewed family exception(s)"));
    }
    flagged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::lego_audit::test_ops::op;

    /// This test prevents generated consumer_a/consumer_b aliases from satisfying the two-caller primitive promotion rule.
    #[test]
    fn synthetic_catalog_consumers_do_not_count_as_primitive_callers() {
        let primitive_id = "vyre-primitives::math::shared_step";
        let ops = vec![
            op(primitive_id, Tier::T2_5, &[]),
            op("vyre-libs::math::real_consumer", Tier::T3, &[primitive_id]),
            op(
                "vyre-libs::catalog::math::shared_step::consumer_a",
                Tier::T3,
                &[primitive_id],
            ),
            op(
                "vyre-libs::catalog::math::shared_step::consumer_b",
                Tier::T3,
                &[primitive_id],
            ),
        ];

        assert_eq!(primitive_caller_counts(&ops).get(primitive_id), Some(&1));
    }

    /// This adversarial test reserves the complete catalog namespace so renamed synthetic aliases cannot bypass caller filtering.
    #[test]
    fn every_catalog_namespace_entry_is_synthetic() {
        assert!(is_synthetic_catalog_consumer(
            "vyre-libs::catalog::graph::frontier::production"
        ));
        assert!(!is_synthetic_catalog_consumer("vyre-libs::graph::frontier"));
    }

    /// This policy test requires low primitive adoption to match an explicit,
    /// owner-reviewed family exception instead of disappearing into prose.
    #[test]
    fn primitive_coverage_requires_registered_family_exception() {
        let ops = collect_ops(&mut Report::clean());
        assert!(ops
            .iter()
            .any(|op| primitive_family(&op.id) == Some("vfs")));
        assert_eq!(check_3_primitive_coverage(&mut Report::clean(), &ops), 0);
    }

    /// A newly under-adopted family fails closed until its owner records a
    /// concrete exception or real callers meet the promotion floor.
    #[test]
    fn unregistered_primitive_family_fails_admission() {
        let ops = vec![op(
            "vyre-primitives::unreviewed::new_primitive",
            Tier::T2_5,
            &[],
        )];
        let mut exceptions = load_primitive_admission_registry().expect("registry");
        exceptions
            .exception
            .retain(|exception| exception.family == "unreviewed");
        assert_eq!(
            validate_primitive_admission(
                &mut Report::clean(),
                &ops,
                &primitive_caller_counts(&ops),
                exceptions
            )
            .0,
            1
        );
    }

    /// This adversarial test ensures synthetic catalog wrappers remain hard failures even though low adoption is advisory.
    #[test]
    fn synthetic_primitive_consumers_remain_hard_failures() {
        let mut ops = collect_ops(&mut Report::clean());
        ops.push(op(
            "vyre-libs::catalog::math::new_primitive::consumer_a",
            Tier::T3,
            &["vyre-libs::math::new_primitive"],
        ));
        assert_eq!(check_3_primitive_coverage(&mut Report::clean(), &ops), 1);
    }

}
