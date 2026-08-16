//! What every operation schema document has to satisfy, generated or supplied.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::OperationTier;

use super::schema::{OperationSchema, SCHEMA_VERSION};

/// Most registrations the defining crate may link whatever features are
/// selected.
///
/// Measured at 9 over the 327 registrations in the checkout: the `vyre-libs`
/// modules that carry them are declared with no `cfg`, so nothing selects them
/// out. A registration above this cap is one nobody can compile away. The name
/// ends in `_CAP` so the ratchet gate reads it as a limit: it may be lowered
/// when a registration gains a gate or leaves the tree, and raising it needs a
/// measurement recorded here.
const UNCONDITIONAL_REGISTRATION_CAP: usize = 9;

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
            let named = accepted_tiers
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
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
    let unconditional: Vec<&str> = schema
        .operations
        .iter()
        .filter(|op| op.features.is_empty())
        .map(|op| op.id.as_str())
        .collect();
    if unconditional.len() > UNCONDITIONAL_REGISTRATION_CAP {
        errors.push(format!(
            "{} operation(s) record no enabling feature where at most {UNCONDITIONAL_REGISTRATION_CAP} may; a registration that always links cannot be selected out: {}",
            unconditional.len(),
            unconditional.join(", ")
        ));
    }
    if tiers != schema.tier_counts {
        errors.push("tier_counts do not match operation records".to_string());
    }
    if categories != schema.category_counts {
        errors.push("category_counts do not match operation records".to_string());
    }
    if let Some(expected) = expected {
        errors.extend(divergences(schema, expected));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Every way `schema` disagrees with the live registry, named field by field.
///
/// One sentence saying the document differs told a reader to regenerate and
/// nothing else, and a mutated feature route reported the same sentence as a
/// mutated tier. The route in particular is read from the checkout rather than
/// declared in the document, so naming it is the only way the message points at
/// the module declaration that decides it.
fn divergences(schema: &OperationSchema, expected: &OperationSchema) -> Vec<String> {
    let mut errors = Vec::new();
    if schema.schema_version != expected.schema_version {
        errors.push(format!(
            "the document records schema version {} where the registry generates {}",
            schema.schema_version, expected.schema_version
        ));
    }
    if schema.authority != expected.authority {
        errors.push(format!(
            "the document records authority `{}` where the registry generates `{}`",
            schema.authority, expected.authority
        ));
    }
    let live: BTreeMap<&str, &super::schema::OperationRecord> = expected
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let documented: BTreeSet<&str> = schema.operations.iter().map(|op| op.id.as_str()).collect();
    for id in live.keys() {
        if !documented.contains(id) {
            errors.push(format!(
                "operation `{id}` is registered and the document omits it"
            ));
        }
    }
    for op in &schema.operations {
        let Some(current) = live.get(op.id.as_str()) else {
            errors.push(format!(
                "the document records operation `{}`, which no registration mints",
                op.id
            ));
            continue;
        };
        if op.features != current.features {
            errors.push(format!(
                "operation `{}` records the feature route [{}] where the registry links it behind [{}]",
                op.id,
                op.features.join(", "),
                current.features.join(", ")
            ));
        }
        if op.tier != current.tier {
            errors.push(format!(
                "operation `{}` records tier `{}` where the registration declares `{}`",
                op.id, op.tier, current.tier
            ));
        }
        if op.category != current.category {
            errors.push(format!(
                "operation `{}` records category `{}` where the registry reads `{}`",
                op.id, op.category, current.category
            ));
        }
        for (field, differs) in [
            ("signature", op.signature != current.signature),
            ("oracle contract", op.oracle != current.oracle),
            (
                "backend support",
                op.backend_support != current.backend_support,
            ),
            ("target facets", op.target_facets != current.target_facets),
            ("laws", op.laws != current.laws),
            (
                "composition chain",
                op.composition_chain != current.composition_chain,
            ),
        ] {
            if differs {
                errors.push(format!(
                    "the {field} of operation `{}` is not the one the registry produces",
                    op.id
                ));
            }
        }
    }
    errors
}

/// `validate_schema` and `divergences` are crate-private, so no integration
/// test can hand them a document and a registry to compare. What the gate
/// reports over the live checkout is asserted in
/// `tests/registry_contracts/operation_schema.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::docs::operation_schema::schema::{
        OperationRecord, OperationSignature, OracleContract,
    };

    fn record(id: &str) -> OperationRecord {
        OperationRecord {
            id: id.to_string(),
            tier: "library".to_string(),
            category: "math".to_string(),
            signature: OperationSignature {
                kind: "dialect_parameters".to_string(),
                buffers: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: Vec::new(),
                bytes_extraction: false,
            },
            features: vec!["math-linalg".to_string()],
            oracle: OracleContract {
                reference_eval: false,
                flat_reference_facet: false,
                fixture_inputs: false,
                expected_output: false,
                tolerance_ulp: 0,
            },
            backend_support: BTreeMap::new(),
            target_facets: Vec::new(),
            laws: Vec::new(),
            composition_chain: Vec::new(),
        }
    }

    fn document(operations: Vec<OperationRecord>) -> OperationSchema {
        let mut tier_counts = BTreeMap::new();
        let mut category_counts = BTreeMap::new();
        for op in &operations {
            *tier_counts.entry(op.tier.clone()).or_insert(0) += 1;
            *category_counts.entry(op.category.clone()).or_insert(0) += 1;
        }
        OperationSchema {
            schema_version: SCHEMA_VERSION,
            authority: "live registry".to_string(),
            operation_count: operations.len(),
            tier_counts,
            category_counts,
            operations,
        }
    }

    /// The route is read from the checkout, so the message has to name it.
    #[test]
    fn a_changed_feature_route_is_named_as_a_route() {
        let live = document(vec![record("libs::math::matmul")]);
        let mut candidate = live.clone();
        candidate.operations[0].features = Vec::new();

        assert_eq!(
            divergences(&candidate, &live),
            vec![
                "operation `libs::math::matmul` records the feature route [] where the registry links it behind [math-linalg]"
                    .to_string()
            ]
        );
    }

    /// A document may not drop a registration, nor invent one.
    #[test]
    fn a_roster_difference_names_the_operation_on_either_side() {
        let live = document(vec![record("libs::math::matmul")]);
        let candidate = document(vec![record("libs::math::gemv")]);

        assert_eq!(
            divergences(&candidate, &live),
            vec![
                "operation `libs::math::matmul` is registered and the document omits it"
                    .to_string(),
                "the document records operation `libs::math::gemv`, which no registration mints"
                    .to_string(),
            ]
        );
    }

    /// Each field is reported as itself, not as one sentence about the document.
    #[test]
    fn each_differing_field_is_named() {
        let live = document(vec![record("libs::math::matmul")]);
        let mut candidate = live.clone();
        candidate.operations[0].tier = "intrinsic".to_string();
        candidate.operations[0].laws = vec!["associative".to_string()];

        assert_eq!(
            divergences(&candidate, &live),
            vec![
                "operation `libs::math::matmul` records tier `intrinsic` where the registration declares `library`".to_string(),
                "the laws of operation `libs::math::matmul` is not the one the registry produces"
                    .to_string(),
            ]
        );
    }

    /// A document that matches the registry reports nothing.
    #[test]
    fn an_identical_document_diverges_in_nothing() {
        let live = document(vec![record("libs::math::matmul")]);

        assert_eq!(divergences(&live.clone(), &live), Vec::<String>::new());
    }
}
