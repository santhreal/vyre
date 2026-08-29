//! The compilation request: search bounds, external facts, and the validation
//! that turns an unvalidated request into one the compiler will accept.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::{GraphValueId, ProgramGraph, ShapeDim, ValueLifetime};
use vyre_foundation::numeric::NumericContract;
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

use crate::device_facts::{validate_device_support, DeviceFacts};
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::grammar::{DerivationStep, ScheduleProduction};
use crate::grid_sync;
use crate::identity::Digest;
use crate::measure::MeasurementRecord;
use crate::mesh::MeshFacts;
use crate::objective::{CompileObjective, ObjectiveMetric};

/// Schedule family the caller requires the selected plan to exercise.
///
/// Selection is otherwise the objective's to make. A requirement states that
/// the caller wants the same semantic graph executed a particular way, which is
/// what conformance needs in order to check one operation's declared contract
/// under every legal schedule instead of only under whichever schedule the
/// objective happened to rank first.
///
/// A requirement no legal candidate satisfies is a refusal, never a fallback: a
/// caller told a schedule was unreachable can raise the budget or change the
/// device facts, while a caller silently served a different schedule has
/// measured something else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredSchedule {
    /// The unspecialized baseline: a plan that applies no grammar production.
    Baseline,
    /// A plan whose derivation applies `production` at least once.
    Production(ScheduleProduction),
}

impl RequiredSchedule {
    /// Whether a candidate derived by `derivation` satisfies this requirement.
    #[must_use]
    pub fn admits(self, derivation: &[DerivationStep]) -> bool {
        match self {
            Self::Baseline => derivation.is_empty(),
            Self::Production(required) => derivation.iter().any(|step| step.production == required),
        }
    }

    /// Stable machine-readable identity of the required family.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Baseline => "MKP000_BASELINE",
            Self::Production(production) => production.code(),
        }
    }
}

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
    pub(crate) objective: CompileObjective,
    pub(crate) search_budget: SearchBudget,
    representative_inputs: BTreeMap<GraphValueId, Vec<u8>>,
    recorded_measurement: Option<MeasurementRecord>,
    mesh: Option<MeshFacts>,
    numeric: Option<NumericContract>,
    required_schedule: Option<RequiredSchedule>,
}

impl CompileRequest {
    /// Construct a request. Call [`Self::validate`] before compilation.
    ///
    /// `objective` states what the compilation optimizes and every hard bound it
    /// refuses to exceed, including the artifact byte ceiling. It is a
    /// constructor argument rather than a default because a plan selected
    /// without a stated objective cannot say what it was selected for.
    #[must_use]
    pub const fn new(
        graph: ProgramGraph,
        facts: ExternalFacts,
        device: DeviceFacts,
        search_budget: SearchBudget,
        objective: CompileObjective,
    ) -> Self {
        Self {
            graph,
            facts,
            objective,
            device,
            search_budget,
            representative_inputs: BTreeMap::new(),
            recorded_measurement: None,
            mesh: None,
            numeric: None,
            required_schedule: None,
        }
    }

    /// State the numeric budget the caller admits for the graph's outputs.
    ///
    /// A schedule that reorders a rounding accumulation computes a different
    /// number. Without a stated budget the search refuses every such schedule,
    /// because nothing says how far the result may move. With one it admits the
    /// reorderings whose composed error stays inside the budget, which is what
    /// makes a tree reduction and a spatial split available over a floating
    /// accumulation.
    #[must_use]
    pub const fn with_numeric_budget(mut self, numeric: NumericContract) -> Self {
        self.numeric = Some(numeric);
        self
    }

    /// Require the selected plan to exercise one schedule family.
    ///
    /// Every candidate the grammar derives is still derived and every legality
    /// decision is still made, so the requirement narrows what may be selected
    /// and never what may be considered. A requirement no legal candidate
    /// satisfies fails the compile with the family it could not reach.
    #[must_use]
    pub const fn requiring_schedule(mut self, required: RequiredSchedule) -> Self {
        self.required_schedule = Some(required);
        self
    }

    /// Supply the device mesh this compilation may place work on.
    ///
    /// A caller that states none is compiled for the one device its facts
    /// describe, which is a mesh of one device rather than a separate path.
    #[must_use]
    pub fn with_mesh(mut self, mesh: MeshFacts) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Supply representative workload input bytes for finalist measurement.
    #[must_use]
    pub fn with_representative_inputs(
        mut self,
        representative_inputs: BTreeMap<GraphValueId, Vec<u8>>,
    ) -> Self {
        self.representative_inputs = representative_inputs;
        self
    }

    /// Supply the measurement evidence an already authenticated artifact
    /// carries, so this compilation cannot replace its winner on measurement
    /// noise.
    ///
    /// A caller recompiling the same graph on the same device passes the record
    /// the previous artifact recorded. The measured path then keeps that winner
    /// unless a challenger clears the protocol's equivalence band, which is what
    /// makes a re-run reproduce the artifact it re-ran.
    #[must_use]
    pub fn with_recorded_measurement(mut self, recorded_measurement: MeasurementRecord) -> Self {
        self.recorded_measurement = Some(recorded_measurement);
        self
    }

    /// Validate topology, programs, device facts, external facts, and bounds.
    pub fn validate(self) -> Result<ValidatedCompileRequest, CompileError> {
        if self
            .objective
            .bounds()
            .limit(ObjectiveMetric::ArtifactBytes)
            .is_none_or(|limit| limit == 0)
        {
            return Err(failure(
                CompilerFailureKind::ArtifactLimit,
                "request.objective.bounds.artifact_bytes",
                "artifact byte limit must be stated and greater than zero",
                "bound ObjectiveMetric::ArtifactBytes at the largest artifact the caller will retain",
            ));
        }
        self.objective.validate(self.device)?;
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
        // A device without cooperative launch needs every hoistable grid fence
        // lowered to launch boundaries before capability admission. A cooperative
        // device keeps the original fenced node: cutting it would manufacture a
        // retained host input for backend-allocated in-place pipeline storage,
        // and direct cooperative dispatch already provides the required fence.
        //
        // An uncuttable fence survives the non-cooperative cut and is refused by
        // `validate_device_support`; a cuttable fence becomes ordered segments.
        let graph = if self.device.supports_cooperative_launch() {
            self.graph
        } else {
            grid_sync::split_graph(self.graph)?
        };
        validate_node_programs(&graph, self.device.capabilities)?;
        validate_device_support(&graph, self.device)?;
        validate_bindings(&graph, &self.facts.symbolic_bindings)?;
        validate_constant_identities(&graph, &self.facts.constant_identities)?;
        validate_representative_inputs(
            &graph,
            &self.representative_inputs,
            &self.facts.symbolic_bindings,
        )?;
        if let Some(recorded) = &self.recorded_measurement {
            recorded.validate()?;
        }
        let mesh = match self.mesh {
            Some(mesh) => {
                mesh.authenticate()?;
                mesh
            }
            None => MeshFacts::single_device(0)?,
        };
        Ok(ValidatedCompileRequest {
            graph,
            facts: self.facts,
            representative_inputs: self.representative_inputs,
            recorded_measurement: self.recorded_measurement,
            device: self.device,
            objective: self.objective,
            search_budget: self.search_budget,
            mesh,
            numeric: self.numeric,
            required_schedule: self.required_schedule,
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
    pub(crate) objective: CompileObjective,
    pub(crate) search_budget: SearchBudget,
    representative_inputs: BTreeMap<GraphValueId, Vec<u8>>,
    recorded_measurement: Option<MeasurementRecord>,
    mesh: MeshFacts,
    pub(crate) numeric: Option<NumericContract>,
    pub(crate) required_schedule: Option<RequiredSchedule>,
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

    /// The schedule family the caller required, when one was stated.
    #[must_use]
    pub const fn required_schedule(&self) -> Option<RequiredSchedule> {
        self.required_schedule
    }

    /// Borrow the validated representative workload used for finalist measurement.
    #[must_use]
    pub const fn representative_inputs(&self) -> &BTreeMap<GraphValueId, Vec<u8>> {
        &self.representative_inputs
    }

    /// Borrow the measurement evidence an already authenticated artifact
    /// recorded for this graph and device, when the caller supplied one.
    #[must_use]
    pub const fn recorded_measurement(&self) -> Option<&MeasurementRecord> {
        self.recorded_measurement.as_ref()
    }

    /// The numeric budget the caller stated, when it stated one.
    #[must_use]
    pub const fn numeric_budget(&self) -> Option<NumericContract> {
        self.numeric
    }

    /// Return the explicit bounded-search policy.
    #[must_use]
    pub const fn search_budget(&self) -> SearchBudget {
        self.search_budget
    }

    /// Borrow the stated compile objective.
    #[must_use]
    pub const fn objective(&self) -> &CompileObjective {
        &self.objective
    }

    /// Return the maximum accepted artifact byte length.
    ///
    /// This is the artifact-byte bound the objective states. Validation refuses
    /// a request that states none, so one ceiling is stated in one place and
    /// read here.
    #[must_use]
    pub fn max_artifact_bytes(&self) -> u64 {
        self.objective
            .bounds()
            .limit(ObjectiveMetric::ArtifactBytes)
            .unwrap_or(u64::MAX)
    }

    /// Return the live device facts the plan was selected against.
    #[must_use]
    pub const fn device(&self) -> DeviceFacts {
        self.device
    }

    /// Borrow the authenticated device mesh the plan is placed on.
    #[must_use]
    pub const fn mesh(&self) -> &MeshFacts {
        &self.mesh
    }

    /// The same validated request under a different stated objective.
    ///
    /// Portfolio selection compiles one part of a workload partition at a time,
    /// and a part is this graph, these facts, and this device under an objective
    /// narrowed to that part. The objective stays the request's own field rather
    /// than a parameter threaded past it, so every compile reads the objective
    /// from one place and records that one in the artifact.
    pub(crate) fn restated(&self, objective: CompileObjective) -> Self {
        Self {
            graph: self.graph.clone(),
            facts: self.facts.clone(),
            device: self.device,
            objective,
            search_budget: self.search_budget,
            representative_inputs: self.representative_inputs.clone(),
            recorded_measurement: self.recorded_measurement.clone(),
            mesh: self.mesh.clone(),
            numeric: self.numeric,
        }
    }

    /// The same validated request under narrowed external facts.
    ///
    /// A specialization variant is this graph and this device compiled for a
    /// pinned dimension or a stated submission arrangement. Only the facts move,
    /// so only the fact validations run again: the graph already passed
    /// structural, capability and device admission, and re-running the grid-fence
    /// cut over an already cut graph would be a second answer to a question that
    /// was settled.
    pub(crate) fn restated_facts(&self, facts: ExternalFacts) -> Result<Self, CompileError> {
        if facts.expected_launch_batch == 0 {
            return Err(failure(
                CompilerFailureKind::InvalidDeviceFacts,
                "request.facts.expected_launch_batch",
                "expected launch batch is zero, so the artifact would never run",
                "supply the number of launches the caller will submit, at least one",
            ));
        }
        validate_bindings(&self.graph, &facts.symbolic_bindings)?;
        validate_constant_identities(&self.graph, &facts.constant_identities)?;
        validate_representative_inputs(
            &self.graph,
            &self.representative_inputs,
            &facts.symbolic_bindings,
        )?;
        Ok(Self {
            graph: self.graph.clone(),
            facts,
            device: self.device,
            objective: self.objective,
            search_budget: self.search_budget,
            representative_inputs: self.representative_inputs.clone(),
            recorded_measurement: self.recorded_measurement.clone(),
            mesh: self.mesh.clone(),
            numeric: self.numeric,
        })
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

fn validate_representative_inputs(
    graph: &ProgramGraph,
    representative_inputs: &BTreeMap<GraphValueId, Vec<u8>>,
    bindings: &BTreeMap<String, u64>,
) -> Result<(), CompileError> {
    let graph_inputs: BTreeMap<GraphValueId, &vyre_foundation::ir::ProgramGraphValue> = graph
        .values()
        .iter()
        .filter(|value| value.producer.is_none())
        .map(|value| (value.id, value))
        .collect();

    for (id, bytes) in representative_inputs {
        let Some(value) = graph_inputs.get(id) else {
            return Err(failure(
                CompilerFailureKind::UnknownRepresentativeInput,
                format!("request.representative_inputs.{}", id.0),
                "representative input names an unknown or graph-produced value",
                "remove stale representative inputs and key inputs by external GraphValueId",
            ));
        };

        let expected_byte_count = crate::resource_records::value_byte_count(value, bindings)?;
        let actual_byte_count = bytes.len() as u64;
        if actual_byte_count != expected_byte_count {
            return Err(failure(
                CompilerFailureKind::RepresentativeInputLengthMismatch,
                format!("request.representative_inputs.{}", id.0),
                format!(
                    "representative input for value `{}` has {} bytes, but graph declaration requires {expected_byte_count} bytes",
                    value.name,
                    bytes.len()
                ),
                "supply representative input bytes matching the static shape and element type",
            ));
        }
    }
    Ok(())
}
