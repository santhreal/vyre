//! Canonical generated operation schema built from live registrations.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vyre::ir::{Node, Program};
use vyre_foundation::algebraic_law_registry::AlgebraicLawRegistration;
use vyre_foundation::operation::classify_operation_id as classify_op_id;
use xtask::gate::{Gate, GateCtx, GateError, Report};

use xtask::release::conformance_op_matrix::read_conformance_required_op_matrix;

const DEFAULT_OUTPUT: &str = "docs/generated/OP_SCHEMA.json";
const MAX_SCHEMA_BYTES: u64 = 16_777_216;

/// Wire version of `docs/generated/OP_SCHEMA.json`.
///
/// `scripts/architecture_docs.py` reads the same file and pins the same
/// number. It cannot import this constant, so
/// `the_python_contract_pins_the_same_operation_schema_version` fails when the
/// two drift; that drift shipped once already, with the generator on 3 and the
/// script still demanding 2.
pub(crate) const SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationSchema {
    pub(crate) schema_version: u32,
    pub(crate) authority: String,
    pub(crate) operation_count: usize,
    pub(crate) tier_counts: BTreeMap<String, usize>,
    pub(crate) category_counts: BTreeMap<String, usize>,
    pub(crate) operations: Vec<OperationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationRecord {
    pub(crate) id: String,
    pub(crate) tier: String,
    pub(crate) category: String,
    pub(crate) signature: OperationSignature,
    pub(crate) features: Vec<String>,
    pub(crate) oracle: OracleContract,
    pub(crate) backend_support: BTreeMap<String, BackendSupport>,
    pub(crate) target_facets: Vec<String>,
    pub(crate) laws: Vec<String>,
    pub(crate) composition_chain: Vec<CompositionStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationSignature {
    pub(crate) kind: String,
    pub(crate) buffers: Vec<BufferSignature>,
    pub(crate) inputs: Vec<TypedParameter>,
    pub(crate) outputs: Vec<TypedParameter>,
    pub(crate) attributes: Vec<String>,
    pub(crate) bytes_extraction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TypedParameter {
    pub(crate) name: String,
    pub(crate) data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BufferSignature {
    pub(crate) binding: u32,
    pub(crate) name: String,
    pub(crate) access: String,
    pub(crate) memory: String,
    pub(crate) element: String,
    pub(crate) count: u32,
    pub(crate) pipeline_live_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OracleContract {
    pub(crate) reference_eval: bool,
    pub(crate) flat_reference_facet: bool,
    pub(crate) fixture_inputs: bool,
    pub(crate) expected_output: bool,
    pub(crate) tolerance_ulp: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BackendSupport {
    pub(crate) status: String,
    pub(crate) test_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CompositionStep {
    pub(crate) depth: usize,
    pub(crate) operation: String,
    pub(crate) registered: bool,
}

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
fn signature_from_program(program: &Program) -> OperationSignature {
    OperationSignature {
        kind: "program_buffers".to_string(),
        buffers: program
            .buffers()
            .iter()
            .map(|buffer| BufferSignature {
                binding: buffer.binding,
                name: buffer.name.to_string(),
                access: format!("{:?}", buffer.access),
                memory: format!("{:?}", buffer.kind),
                element: format!("{:?}", buffer.element),
                count: buffer.count,
                pipeline_live_out: buffer.pipeline_live_out,
            })
            .collect(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: Vec::new(),
        bytes_extraction: program
            .buffers()
            .iter()
            .any(|buffer| buffer.bytes_extraction),
    }
}

fn signature_from_declaration(
    signature: &vyre_foundation::dialect_lookup::Signature,
) -> OperationSignature {
    OperationSignature {
        kind: "dialect_parameters".to_string(),
        buffers: Vec::new(),
        inputs: signature
            .inputs
            .iter()
            .map(|parameter| TypedParameter {
                name: parameter.name.to_string(),
                data_type: parameter.ty.to_string(),
            })
            .collect(),
        outputs: signature
            .outputs
            .iter()
            .map(|parameter| TypedParameter {
                name: parameter.name.to_string(),
                data_type: parameter.ty.to_string(),
            })
            .collect(),
        attributes: signature
            .attrs
            .iter()
            .map(|attribute| format!("{attribute:?}"))
            .collect(),
        bytes_extraction: signature.bytes_extraction,
    }
}

/// Holds the canonical live operation contract schema to the registry.
pub struct OperationSchemaGate;

impl Gate for OperationSchemaGate {
    fn name(&self) -> &'static str {
        "operation-schema"
    }

    fn help(&self) -> &'static str {
        "Hold the canonical live operation contract schema to the registry; --write regenerates it, --validate PATH judges one document"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let schema = match build() {
            Ok(schema) => schema,
            Err(errors) => {
                return Ok(Report::from_messages(
                    errors,
                    "repair the registration the schema rejects, then run the gate again",
                ));
            }
        };
        if let Some(path) = ctx.flag("--validate") {
            let candidate = read_schema(Path::new(path)).map_err(|error| {
                GateError::new(error, "pass a readable schema document after --validate")
            })?;
            let mut report = match validate_schema(&candidate, Some(&schema)) {
                Ok(()) => Report::clean(),
                Err(errors) => Report::from_messages(
                    errors,
                    "repair the document, or regenerate it from the registry with --write",
                ),
            };
            report.note(format!(
                "{} live operation contract(s) in the validated document",
                candidate.operation_count
            ));
            return Ok(report);
        }
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates(DEFAULT_OUTPUT, &schema);
        let mut report = xtask::artifact_gate::settle_inspection(ctx, self.name(), inspection);
        report.note(format!(
            "{} live operation contract(s)",
            schema.operation_count
        ));
        Ok(report)
    }
}

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
    for op in &schema.operations {
        let expected_tier = classify_op_id(&op.id).matrix_value();
        if op.tier != expected_tier || op.tier == "unknown" {
            errors.push(format!(
                "operation `{}` tier `{}` does not match `{expected_tier}`",
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
        let expected_features = feature_route(&op.id, &op.category);
        if op.features != expected_features {
            errors.push(format!(
                "operation `{}` feature route {:?} does not match {:?}",
                op.id, op.features, expected_features
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

fn collect_composition(
    node: &Node,
    depth: usize,
    all_ids: &BTreeSet<&str>,
    out: &mut Vec<CompositionStep>,
) {
    match node {
        Node::Region {
            generator, body, ..
        } => {
            let operation = generator.as_str().to_string();
            out.push(CompositionStep {
                depth,
                registered: all_ids.contains(operation.as_str()),
                operation,
            });
            for child in body.iter() {
                collect_composition(child, depth + 1, all_ids, out);
            }
        }
        Node::Block(children) => {
            for child in children {
                collect_composition(child, depth, all_ids, out);
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                collect_composition(child, depth, all_ids, out);
            }
            for child in otherwise {
                collect_composition(child, depth, all_ids, out);
            }
        }
        Node::Loop { body, .. } => {
            for child in body {
                collect_composition(child, depth, all_ids, out);
            }
        }
        _ => {}
    }
}

fn validate_composition(id: &str, chain: &[CompositionStep], errors: &mut Vec<String>) {
    let mut previous_depth = 0;
    for (index, step) in chain.iter().enumerate() {
        if step.operation.trim().is_empty() {
            errors.push(format!("operation `{id}` has an empty composition step"));
        }
        if index == 0 && step.depth != 0 {
            errors.push(format!(
                "operation `{id}` composition chain starts at depth {} instead of 0",
                step.depth
            ));
        } else if index > 0 && step.depth > previous_depth + 1 {
            errors.push(format!(
                "operation `{id}` composition depth jumps from {previous_depth} to {}",
                step.depth
            ));
        }
        previous_depth = step.depth;
    }
}

fn category_from_id(id: &str) -> String {
    id.split("::")
        .nth(1)
        .or_else(|| id.split('.').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("uncategorized")
        .to_string()
}

fn feature_route(id: &str, category: &str) -> Vec<String> {
    if id.starts_with("vyre-primitives::") {
        let domain = id.split("::").nth(1).unwrap_or(category);
        let feature = if domain == "vfs" { "parsing" } else { domain };
        return vec![feature.to_string(), "inventory-registry".to_string()];
    }
    let feature = match category {
        "scan" | "matching" => "matching",
        "crypto" => "crypto",
        "math" | "optim" | "quant" => "math",
        "nn" => "nn",
        "parsing" => "parsing",
        "logical" => "logical",
        "security" => "security",
        "visual" => "visual",
        "hash" => "hash",
        "decode" => "decode",
        "rule" => "rule",
        "text" => "text",
        _ => "full",
    };
    vec![feature.to_string()]
}

fn read_manifest_features(
    root: &Path,
    errors: &mut Vec<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut catalog = BTreeMap::new();
    for crate_name in ["vyre-driver", "vyre-primitives", "vyre-libs"] {
        let path = root.join(crate_name).join("Cargo.toml");
        let text = match read_text_bounded(&path) {
            Ok(value) => value,
            Err(error) => {
                errors.push(format!(
                    "read {} for operation features: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let value = match toml::from_str::<toml::Value>(&text) {
            Ok(value) => value,
            Err(error) => {
                errors.push(format!(
                    "parse {} for operation features: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let features = value
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        catalog.insert(crate_name.to_string(), features);
    }
    catalog
}

fn workspace_root() -> PathBuf {
    xtask::checkout::checkout_root()
}

fn parse_args(args: &[String]) -> Result<(PathBuf, bool, Option<PathBuf>), String> {
    let mut output = workspace_root().join(DEFAULT_OUTPUT);
    let mut check = false;
    let mut validate = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--check" => check = true,
            "--output" => {
                index += 1;
                output = PathBuf::from(args.get(index).ok_or("Fix: --output needs a path")?);
            }
            "--validate" => {
                index += 1;
                validate = Some(PathBuf::from(
                    args.get(index).ok_or("Fix: --validate needs a path")?,
                ));
            }
            other => return Err(format!("Fix: unknown operation-schema argument `{other}`")),
        }
        index += 1;
    }
    if check && validate.is_some() {
        return Err("Fix: --check and --validate are mutually exclusive".to_string());
    }
    Ok((output, check, validate))
}

fn read_schema(path: &Path) -> Result<OperationSchema, String> {
    let text =
        read_text_bounded(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_SCHEMA_BYTES, "operation schema")
}

#[cfg(test)]
mod tests {
    use super::SCHEMA_VERSION;

    /// The op-schema wire version is spelled in two languages: this crate
    /// generates the file, and `scripts/architecture_docs.py` re-checks it.
    /// They drifted once, generator on 3 against a script still demanding 2,
    /// and nothing went red. This fails the moment they disagree again.
    #[test]
    fn the_python_contract_pins_the_same_operation_schema_version() {
        let script = std::fs::read_to_string(
            xtask::checkout::checkout_root().join("scripts/architecture_docs.py"),
        )
        .expect("Fix: scripts/architecture_docs.py must be readable");

        let expected = format!("OPERATION_SCHEMA_VERSION = {SCHEMA_VERSION}");
        assert!(
            script.contains(&expected),
            "scripts/architecture_docs.py must declare `{expected}`; \
             bump it in the same change as SCHEMA_VERSION"
        );
        assert!(
            script.contains("!= OPERATION_SCHEMA_VERSION"),
            "scripts/architecture_docs.py must compare against \
             OPERATION_SCHEMA_VERSION, not a second literal"
        );
    }
}
