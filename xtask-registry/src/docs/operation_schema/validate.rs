//! What every operation schema document has to satisfy, generated or supplied.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::OperationTier;

use super::schema::{OperationSchema, SCHEMA_VERSION};

pub(crate) fn validate_schema(
    schema: &OperationSchema,
    expected: Option<&OperationSchema>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if schema.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "operation schema version must be {SCHEMA_VERSION}, found {}",
            schema.schema_version
        ));
    }
    if schema.operation_count != schema.operations.len() {
        errors.push(format!(
            "operation_count {} does not match {} records",
            schema.operation_count,
            schema.operations.len()
        ));
    }
    let mut ids = BTreeSet::new();
    let mut tiers = BTreeMap::new();
    let mut categories = BTreeMap::new();
    let known_laws: BTreeSet<&str> = vyre_spec::law_catalog().iter().copied().collect();
    for op in &schema.operations {
        if op.id.trim().is_empty() || !ids.insert(op.id.as_str()) {
            errors.push(format!("operation id `{}` is empty or duplicated", op.id));
        }
    }
    let accepted_tiers: BTreeSet<&str> = OperationTier::ALL
        .iter()
        .map(|tier| tier.matrix_value())
        .filter(|spelling| *spelling != OperationTier::Unknown.matrix_value())
        .collect();
    for op in &schema.operations {
        if !accepted_tiers.contains(op.tier.as_str()) {
            let named = accepted_tiers.iter().copied().collect::<Vec<_>>().join(", ");
            errors.push(format!(
                "operation `{}` records tier `{}`; the accepted spellings are {named}",
                op.id, op.tier
            ));
        }
        if op.category.trim().is_empty() || op.category == "uncategorized" {
            errors.push(format!(
                "operation `{}` has invalid category `{}`",
                op.id, op.category
            ));
        }
        let signature_valid = match op.signature.kind.as_str() {
            "program_buffers" => {
                !op.signature.buffers.is_empty()
                    && op.signature.buffers.iter().all(|buffer| {
                        !buffer.name.trim().is_empty() && !buffer.access.trim().is_empty()
                    })
                    && op.signature.inputs.is_empty()
                    && op.signature.outputs.is_empty()
            }
            "dialect_parameters" => {
                op.signature.buffers.is_empty()
                    && op
                        .signature
                        .inputs
                        .iter()
                        .chain(op.signature.outputs.iter())
                        .all(|parameter| {
                            !parameter.name.trim().is_empty()
                                && !parameter.data_type.trim().is_empty()
                        })
            }
            _ => false,
        };
        if !signature_valid {
            errors.push(format!(
                "operation `{}` has an invalid operation signature",
                op.id
            ));
        }
        if op.features.is_empty() {
            errors.push(format!(
                "operation `{}` records no enabling feature; a registration that always links cannot be selected out",
                op.id
            ));
        }
        let reference_status = op
            .backend_support
            .get("reference")
            .map(|support| support.status.as_str());
        if reference_status == Some("supported") && !op.oracle.reference_eval {
            errors.push(format!(
                "operation `{}` does not declare its supported reference oracle",
                op.id
            ));
        }
        let mut sorted_target_facets = op.target_facets.clone();
        sorted_target_facets.sort();
        sorted_target_facets.dedup();
        if sorted_target_facets != op.target_facets
            || op
                .target_facets
                .iter()
                .any(|target| target.trim().is_empty())
        {
            errors.push(format!(
                "operation `{}` target facets must be non-empty identities in sorted unique order",
                op.id
            ));
        }
        for backend in ["reference", "cuda", "wgpu"] {
            match op.backend_support.get(backend) {
                Some(support) if !support.status.trim().is_empty() => {}
                _ => errors.push(format!(
                    "operation `{}` is missing valid backend `{backend}`",
                    op.id
                )),
            }
        }
        let mut sorted_laws = op.laws.clone();
        sorted_laws.sort();
        sorted_laws.dedup();
        if sorted_laws != op.laws
            || op
                .laws
                .iter()
                .any(|law| law.trim().is_empty() || !known_laws.contains(law.as_str()))
        {
            errors.push(format!(
                "operation `{}` laws must be known names in sorted unique order",
                op.id
            ));
        }
        let mut previous_depth = 0;
        for (index, step) in op.composition_chain.iter().enumerate() {
            if step.operation.trim().is_empty()
                || step.registered != ids.contains(step.operation.as_str())
                || (index == 0 && step.depth != 0)
                || (index > 0 && step.depth > previous_depth + 1)
            {
                errors.push(format!(
                    "operation `{}` has an inconsistent composition chain",
                    op.id
                ));
                break;
            }
            previous_depth = step.depth;
        }
        *tiers.entry(op.tier.clone()).or_insert(0) += 1;
        *categories.entry(op.category.clone()).or_insert(0) += 1;
    }
    if tiers != schema.tier_counts {
        errors.push("tier_counts do not match operation records".to_string());
    }
    if categories != schema.category_counts {
        errors.push("category_counts do not match operation records".to_string());
    }
    if let Some(expected) = expected {
        if schema != expected {
            errors.push("candidate operation schema differs from live registrations".to_string());
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
