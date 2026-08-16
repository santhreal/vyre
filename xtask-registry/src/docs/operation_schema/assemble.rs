//! Joining the live registries into one schema document.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use vyre::ir::Program;
use vyre_foundation::algebraic_law_registry::AlgebraicLawRegistration;
use vyre_foundation::operation::classify_operation_id as classify_op_id;

use xtask::release::conformance_op_matrix::read_conformance_required_op_matrix;

use super::composition::{collect_composition, validate_composition};
use super::routing::{category_from_id, feature_route, read_manifest_features};
use super::schema::{
    BackendSupport, OperationRecord, OperationSchema, OracleContract, SCHEMA_VERSION,
};
use super::signature::{signature_from_declaration, signature_from_program};
use super::validate::validate_schema;

struct LiveEntry {
    id: &'static str,
    signature: Option<&'static vyre_foundation::dialect_lookup::Signature>,
    build: Option<fn() -> Program>,
    category: Option<&'static str>,
    has_inputs: bool,
    laws: &'static [&'static str],
    has_expected: bool,
    tolerance_ulp: u32,
}

impl LiveEntry {
    fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }
}

/// Build the canonical schema, or every reason the registry rejects one.
pub(crate) fn build() -> Result<OperationSchema, Vec<String>> {
    let root = workspace_root();
    let catalog = read_conformance_required_op_matrix(&root);
    let mut errors = catalog.errors;
    let manifest_features = read_manifest_features(&root, &mut errors);
    let mut backend_rows: BTreeMap<String, BTreeMap<String, BackendSupport>> = BTreeMap::new();
    for row in catalog.release_backend_specs {
        backend_rows.entry(row.op_id).or_default().insert(
            row.backend,
            BackendSupport {
                status: row.status,
                test_paths: row.test_paths,
            },
        );
    }
    let mut declared_laws: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for registration in inventory::iter::<AlgebraicLawRegistration>() {
        declared_laws
            .entry(registration.op_id)
            .or_default()
            .insert(registration.law.name().to_string());
    }

    let flat_reference_ids = vyre_reference::reference_facets()
        .map(|facet| facet.operation_id)
        .collect::<BTreeSet<_>>();
    let mut target_facets: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    match vyre_driver::registered_target_operation_facets() {
        Ok(facets) => {
            for facet in facets {
                target_facets
                    .entry(facet.operation_id)
                    .or_default()
                    .insert(facet.target_id.as_str().to_string());
            }
        }
        Err(error) => errors.push(format!("target facet registry startup failed: {error}")),
    }

    let live = vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| LiveEntry {
            id: entry.id,
            signature: entry.signature,
            build: entry.build,
            category: entry.category(),
            has_inputs: entry.test_inputs.is_some(),
            has_expected: entry.expected_output.is_some(),
            tolerance_ulp: entry.tolerance(),
            laws: entry.laws,
        })
        .collect::<Vec<_>>();
    let all_ids: BTreeSet<&str> = live.iter().map(|entry| entry.id).collect();
    let mut records = Vec::with_capacity(live.len());
    if all_ids.len() != live.len() {
        let mut seen = BTreeSet::new();
        for id in live.iter().map(|entry| entry.id) {
            if !seen.insert(id) {
                errors.push(format!("duplicate live operation id `{id}`"));
            }
        }
    }

    for entry in live {
        let category = entry
            .category
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| category_from_id(entry.id));
        let (signature, composition_chain, reference_eval) = if let Some(program) = entry.program()
        {
            let mut composition_chain = Vec::new();
            for node in program.entry() {
                collect_composition(node, 0, &all_ids, &mut composition_chain);
            }
            validate_composition(entry.id, &composition_chain, &mut errors);
            (signature_from_program(&program), composition_chain, true)
        } else if let Some(signature) = entry.signature {
            (signature_from_declaration(signature), Vec::new(), false)
        } else {
            errors.push(format!(
                "canonical operation `{}` has neither a neutral builder nor a declared signature",
                entry.id
            ));
            continue;
        };

        let mut laws = declared_laws.remove(entry.id).unwrap_or_default();
        laws.extend(entry.laws.iter().map(|law| (*law).to_string()));
        let laws = laws.into_iter().collect();

        let tier = classify_op_id(entry.id).matrix_value().to_string();
        if tier == "unknown" {
            errors.push(format!("operation `{}` has an unknown tier", entry.id));
        }
        let features = feature_route(entry.id, &category);
        if features.is_empty() {
            errors.push(format!(
                "operation `{}` has no enabling feature route",
                entry.id
            ));
        }
        let crate_name = entry.id.split("::").next().unwrap_or("");
        match manifest_features.get(crate_name) {
            Some(available) => {
                for feature in &features {
                    if !available.contains(feature) {
                        errors.push(format!(
                            "operation `{}` feature `{feature}` is not declared by `{crate_name}`",
                            entry.id
                        ));
                    }
                }
            }
            None => errors.push(format!(
                "operation `{}` has no owning manifest feature catalog",
                entry.id
            )),
        }
        let support = backend_rows.remove(entry.id).unwrap_or_default();
        for backend in ["reference", "cuda", "wgpu"] {
            if !support.contains_key(backend) {
                errors.push(format!(
                    "operation `{}` has no `{backend}` support row",
                    entry.id
                ));
            }
        }
        records.push(OperationRecord {
            id: entry.id.to_string(),
            tier,
            category,
            signature,
            features,
            oracle: OracleContract {
                flat_reference_facet: flat_reference_ids.contains(entry.id),
                reference_eval,
                fixture_inputs: entry.has_inputs,
                expected_output: entry.has_expected,
                tolerance_ulp: entry.tolerance_ulp,
            },
            backend_support: support,
            target_facets: target_facets
                .remove(entry.id)
                .unwrap_or_default()
                .into_iter()
                .collect(),
            laws,
            composition_chain,
        });
    }
    records.sort_by(|left, right| left.id.cmp(&right.id));
    for extra in backend_rows.keys() {
        if extra.starts_with("vyre-") {
            errors.push(format!(
                "backend matrix row `{extra}` has no live registration"
            ));
        }
    }
    for extra in declared_laws.keys() {
        if extra.starts_with("vyre-") {
            errors.push(format!(
                "algebraic-law declaration `{extra}` has no live registration"
            ));
        }
    }

    let mut tier_counts = BTreeMap::new();
    let mut category_counts = BTreeMap::new();
    for record in &records {
        *tier_counts.entry(record.tier.clone()).or_insert(0) += 1;
        *category_counts.entry(record.category.clone()).or_insert(0) += 1;
    }
    let schema = OperationSchema {
        schema_version: SCHEMA_VERSION,
        authority: "canonical OperationRegistration records joined with reference-owned ReferenceFacet and concrete-driver target facets, built Programs and signatures, Cargo manifests, algebraic-law inventories, and docs/optimization/OP_MATRIX.toml backend rows".to_string(),
        operation_count: records.len(),
        tier_counts,
        category_counts,
        operations: records,
    };
    if let Err(mut validation) = validate_schema(&schema, None) {
        errors.append(&mut validation);
    }
    if errors.is_empty() {
        Ok(schema)
    } else {
        Err(errors)
    }
}

fn workspace_root() -> PathBuf {
    xtask::checkout::checkout_root()
}
