//! The compilation request: search bounds, external facts, and the validation
//! that turns an unvalidated request into one the compiler will accept.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::{GraphValueId, ProgramGraph, ShapeDim, ValueLifetime};
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

use crate::device_facts::{validate_device_support, DeviceFacts};
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::grid_sync;
use crate::identity::Digest;

/// Explicit bounds for one whole-program schedule search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchBudget {
    /// Maximum legal candidates examined by the open cost model.
    pub max_candidates: u32,
    /// Maximum abstract CPU work units consumed by analysis and search.
    pub max_cpu_work: u64,
    /// Maximum target compilations used for finalist evaluation.
    pub max_target_compilations: u32,
    /// Maximum on-device measurements used for finalist evaluation.
    pub max_measurements: u32,
    /// Maximum elapsed search time in nanoseconds.
    pub max_elapsed_ns: u64,
}

impl SearchBudget {
    /// Construct an explicit bounded search budget.
    #[must_use]
    pub const fn new(
        max_candidates: u32,
        max_cpu_work: u64,
        max_target_compilations: u32,
        max_measurements: u32,
        max_elapsed_ns: u64,
    ) -> Self {
        Self {
            max_candidates,
            max_cpu_work,
            max_target_compilations,
            max_measurements,
            max_elapsed_ns,
        }
    }
}
/// Exact bounded work performed while selecting a whole-program plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchWork {
    /// Legal candidates scored by the open cost model.
    pub candidates_explored: u32,
    /// Abstract deterministic CPU work units consumed.
    pub cpu_work: u64,
    /// Target compilations performed for finalist evaluation.
    pub target_compilations: u32,
    /// On-device measurements performed for finalist evaluation.
    pub measurements: u32,
    /// Deterministic elapsed-budget units charged by the search.
    pub elapsed_ns: u64,
}

/// Stable external semantic facts not encoded by graph topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFacts {
    /// Digest of validated semantic configuration outside the graph.
    pub configuration_digest: Digest,
    /// Exact value for every symbolic graph dimension.
    pub symbolic_bindings: BTreeMap<String, u64>,
    /// Verified content identity for every constant graph value.
    pub constant_identities: BTreeMap<GraphValueId, Digest>,
    /// Launches the caller will submit against this artifact.
    ///
    /// Persistent execution pays a one-time setup cost and saves one launch
    /// overhead per submission, so the count the caller expects decides whether
    /// that trade is profitable. One submission never amortizes it.
    pub expected_launch_batch: u32,
}

impl ExternalFacts {
    /// Construct external facts with no constant identities and one launch.
    #[must_use]
    pub fn new(configuration_digest: Digest, symbolic_bindings: BTreeMap<String, u64>) -> Self {
        Self {
            configuration_digest,
            symbolic_bindings,
            constant_identities: BTreeMap::new(),
            expected_launch_batch: 1,
        }
    }

    /// Record how many launches the caller will submit against the artifact.
    #[must_use]
    pub fn with_expected_launch_batch(mut self, expected_launch_batch: u32) -> Self {
        self.expected_launch_batch = expected_launch_batch;
        self
    }
}

/// Unvalidated whole-program compilation request.
pub struct CompileRequest {
    pub(crate) graph: ProgramGraph,
    pub(crate) facts: ExternalFacts,
    pub(crate) device: DeviceFacts,
    pub(crate) search_budget: SearchBudget,
    pub(crate) max_artifact_bytes: u64,
}

impl CompileRequest {
    /// Construct a request. Call [`Self::validate`] before compilation.
    #[must_use]
    pub const fn new(
        graph: ProgramGraph,
        facts: ExternalFacts,
        device: DeviceFacts,
        search_budget: SearchBudget,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            graph,
            facts,
            device,
            search_budget,
            max_artifact_bytes,
        }
    }

    /// Validate topology, programs, device facts, external facts, and bounds.
    pub fn validate(self) -> Result<ValidatedCompileRequest, CompileError> {
        if self.max_artifact_bytes == 0 {
            return Err(failure(
                CompilerFailureKind::ArtifactLimit,
                "request.max_artifact_bytes",
                "artifact byte limit must be greater than zero",
                "supply a positive bounded artifact byte limit",
            ));
        }
        if self.search_budget.max_candidates == 0
            || self.search_budget.max_cpu_work == 0
            || self.search_budget.max_elapsed_ns == 0
        {
            return Err(failure(
                CompilerFailureKind::InvalidSearchBudget,
                "request.search_budget",
                "candidate, CPU-work, and elapsed-work bounds must be positive",
                "supply explicit positive bounds for every mandatory search dimension",
            ));
        }
        if self.facts.expected_launch_batch == 0 {
            return Err(failure(
                CompilerFailureKind::InvalidDeviceFacts,
                "request.facts.expected_launch_batch",
                "expected launch batch is zero, so the artifact would never run",
                "supply the number of launches the caller will submit, at least one",
            ));
        }
        self.graph.analyze().map_err(|error| {
            failure(
                CompilerFailureKind::InvalidProgram,
                "request.graph",
                error.to_string(),
                "supply a structurally valid acyclic ProgramGraph",
            )
        })?;
        // The cut runs before every consumer of the graph: device admission,
        // binding validation, IR validation, and schedule search all see a graph
        // with no whole-grid fence left in a node body. A graph without a fence is
        // returned untouched, so the artifact digest is unchanged. Validating
        // after the cut validates exactly the programs compilation consumes, and
        // against the live device rather than a compiler-wide capability floor.
        let graph = grid_sync::split_graph(self.graph)?;
        validate_node_programs(&graph, self.device.capabilities)?;
        validate_device_support(&graph, self.device)?;
        validate_bindings(&graph, &self.facts.symbolic_bindings)?;
        validate_constant_identities(&graph, &self.facts.constant_identities)?;
        Ok(ValidatedCompileRequest {
            graph,
            facts: self.facts,
            device: self.device,
            search_budget: self.search_budget,
            max_artifact_bytes: self.max_artifact_bytes,
        })
    }
}

/// Validate every node program against the live capability snapshot.
///
/// The first error wins and is relocated onto the graph node that carried it, so
/// a diagnostic names the node rather than an anonymous program.
fn validate_node_programs(
    graph: &ProgramGraph,
    capabilities: BackendCapabilities,
) -> Result<(), CompileError> {
    for node in graph.nodes() {
        let report = validate_with_options(
            &node.program,
            ValidationOptions::universal().with_backend_capabilities(capabilities),
        );
        if let Some(issue) = report.errors.into_iter().next() {
            let path = format!("request.graph.nodes[{}].program", node.id.0);
            let mut diagnostic = issue.diagnostic();
            if let Some(location) = diagnostic.location.as_mut() {
                location.path = Some(path);
                location.graph_node = Some(node.id.0);
            }
            return Err(CompileError { diagnostic });
        }
    }
    Ok(())
}

/// A graph and complete immutable facts that passed request validation.
pub struct ValidatedCompileRequest {
    pub(crate) graph: ProgramGraph,
    pub(crate) facts: ExternalFacts,
    pub(crate) device: DeviceFacts,
    pub(crate) search_budget: SearchBudget,
    pub(crate) max_artifact_bytes: u64,
}

impl ValidatedCompileRequest {
    /// Borrow the validated source graph.
    #[must_use]
    pub const fn graph(&self) -> &ProgramGraph {
        &self.graph
    }

    /// Borrow validated external semantic facts.
    #[must_use]
    pub const fn facts(&self) -> &ExternalFacts {
        &self.facts
    }

    /// Return the explicit bounded-search policy.
    #[must_use]
    pub const fn search_budget(&self) -> SearchBudget {
        self.search_budget
    }

    /// Return the maximum accepted artifact byte length.
    #[must_use]
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }

    /// Return the live device facts the plan was selected against.
    #[must_use]
    pub const fn device(&self) -> DeviceFacts {
        self.device
    }
}

fn validate_bindings(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
) -> Result<(), CompileError> {
    let symbols: BTreeSet<&str> = graph
        .values()
        .iter()
        .flat_map(|value| &value.contract.shape)
        .filter_map(|dim| match dim {
            ShapeDim::Known(_) => None,
            ShapeDim::Symbol(symbol) => Some(symbol.as_str()),
        })
        .collect();
    if let Some(symbol) = symbols
        .iter()
        .find(|symbol| !bindings.contains_key(**symbol))
    {
        return Err(failure(
            CompilerFailureKind::MissingSymbol,
            format!("request.facts.symbolic_bindings.{symbol}"),
            "graph symbol has no exact extent",
            "bind every symbolic graph dimension before compilation",
        ));
    }
    if let Some(symbol) = bindings
        .keys()
        .find(|symbol| !symbols.contains(symbol.as_str()))
    {
        return Err(failure(
            CompilerFailureKind::UnknownSymbol,
            format!("request.facts.symbolic_bindings.{symbol}"),
            "binding does not occur in the graph",
            "remove stale bindings or use the graph's exact symbol name",
        ));
    }
    Ok(())
}

fn validate_constant_identities(
    graph: &ProgramGraph,
    identities: &BTreeMap<GraphValueId, Digest>,
) -> Result<(), CompileError> {
    let constants = graph
        .values()
        .iter()
        .filter(|value| value.contract.lifetime == ValueLifetime::Constant)
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if let Some(id) = constants.iter().find(|id| !identities.contains_key(*id)) {
        return Err(failure(
            CompilerFailureKind::MissingConstantIdentity,
            format!("request.facts.constant_identities.{}", id.0),
            "constant graph value has no verified content identity",
            "supply one digest keyed by the constant GraphValueId",
        ));
    }
    if let Some(id) = identities.keys().find(|id| !constants.contains(id)) {
        return Err(failure(
            CompilerFailureKind::UnknownConstantIdentity,
            format!("request.facts.constant_identities.{}", id.0),
            "constant identity names a non-constant or missing graph value",
            "remove stale identities and key constant content by GraphValueId",
        ));
    }
    Ok(())
}
