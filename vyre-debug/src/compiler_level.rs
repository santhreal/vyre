//! Structural inspection and rendering across all five compiler levels.
//!
//! Level 1: Semantic IR (Program, ProgramGraph)
//! Level 2: Optimizer & IR transforms (Facts, Simplifications, Proofs)
//! Level 3: Lowering (KernelDescriptor)
//! Level 4: Megakernel & Plan (Artifact, SelectedPlan, SearchCertificate)
//! Level 5: Target Emission (TargetPayload, TargetModuleBundle, WGSL/PTX/binary)
//! Driver/Runtime: Execution state and session diagnostics.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use vyre::compiler::{Artifact, TargetPayload};
use vyre::ir::{Program, ProgramGraph};
use vyre_lower::KernelDescriptor;

/// View of compiler state at any of the five architectural compiler levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompilerLevelView {
    /// Level 1: Semantic IR frontend program or graph.
    SemanticIr {
        /// Name of the primary entry node.
        name: String,
        /// Number of nodes in the graph.
        node_count: usize,
        /// Total operation count across all nodes.
        op_count: usize,
        /// Declared buffer names and datatypes.
        buffers: Vec<(String, String)>,
    },
    /// Level 2: Optimizer IR with transform facts and proof citations.
    Optimizer {
        /// Number of transform passes applied.
        pass_count: usize,
        /// Proved algebraic and layout invariants.
        proof_citations: Vec<String>,
        /// Inferred external and symbolic facts.
        inferred_facts: BTreeMap<String, String>,
    },
    /// Level 3: Lowering physical kernel descriptor.
    Lowering {
        /// Entry point name.
        entry_point: String,
        /// Total lowered operations.
        op_count: usize,
        /// Bound physical registers and buffers.
        bindings: Vec<u32>,
        /// Workgroup size declared by the descriptor.
        workgroup_size: [u32; 3],
    },
    /// Level 4: Megakernel whole-program plan and artifact container.
    MegakernelPlan {
        /// Neutral artifact content digest.
        digest: String,
        /// Number of scheduled fusion groups.
        fusion_groups: usize,
        /// Barrier count.
        barrier_count: usize,
        /// Resource count.
        resource_count: usize,
        /// Evaluated candidate count during search.
        candidates_evaluated: usize,
        /// Pruned candidate count during search.
        candidates_pruned: usize,
    },
    /// Level 5: Target emission payload.
    TargetEmission {
        /// Target payload format identity.
        format: String,
        /// Target format version.
        format_version: u16,
        /// Emitted bytecode / shader size in bytes.
        byte_size: usize,
        /// Target entry point names.
        entry_points: Vec<String>,
    },
    /// Driver and runtime state.
    DriverRuntime {
        /// Active session status.
        status: String,
        /// Admitted resident resources.
        resident_resources: usize,
        /// Submissions completed.
        completed_submissions: usize,
    },
}

impl CompilerLevelView {
    /// Create Level 1 view from a Program.
    #[must_use]
    pub fn from_program(name: impl Into<String>, program: &Program) -> Self {
        let buffers = program
            .buffers
            .iter()
            .map(|b| (b.name.to_string(), format!("{:?}", b.dtype)))
            .collect();
        Self::SemanticIr {
            name: name.into(),
            node_count: 1,
            op_count: program.entry.len(),
            buffers,
        }
    }

    /// Create Level 1 view from a ProgramGraph.
    #[must_use]
    pub fn from_program_graph(graph: &ProgramGraph) -> Self {
        let mut total_ops = 0;
        let mut buffers = Vec::new();
        for node in graph.nodes() {
            total_ops += node.program.entry.len();
            for b in &*node.program.buffers {
                buffers.push((format!("{}:{}", node.name, b.name), format!("{:?}", b.dtype)));
            }
        }
        Self::SemanticIr {
            name: "graph".to_string(),
            node_count: graph.nodes().len(),
            op_count: total_ops,
            buffers,
        }
    }

    /// Create Level 3 view from a KernelDescriptor.
    #[must_use]
    pub fn from_descriptor(entry_point: impl Into<String>, desc: &KernelDescriptor) -> Self {
        let bindings = desc.bindings.slots.iter().map(|b| b.slot).collect();
        Self::Lowering {
            entry_point: entry_point.into(),
            op_count: desc.body.ops.len(),
            bindings,
            workgroup_size: desc.dispatch.workgroup_size,
        }
    }

    /// Create Level 4 view from an Artifact.
    #[must_use]
    pub fn from_artifact(artifact: &Artifact) -> Self {
        let plan = artifact.selected_plan();
        let evaluated = plan.certificate.derived.iter().map(|d| d.derived as usize).sum();
        let pruned = plan.certificate.pruned.iter().map(|p| p.count as usize).sum();
        Self::MegakernelPlan {
            digest: format!("{:02x?}", artifact.digest().as_bytes()),
            fusion_groups: plan.fusion.len(),
            barrier_count: plan.barriers.len(),
            resource_count: artifact.resources().len(),
            candidates_evaluated: evaluated,
            candidates_pruned: pruned,
        }
    }

    /// Create Level 5 view from a TargetPayload.
    #[must_use]
    pub fn from_target_payload(payload: &TargetPayload) -> Self {
        let entry_points = payload.entries().iter().map(|e| e.name.clone()).collect();
        Self::TargetEmission {
            format: payload.format().identity().to_string(),
            format_version: payload.format().version(),
            byte_size: payload.bytes().len(),
            entry_points,
        }
    }

    /// Stable compiler level number (1 through 5, or 6 for runtime).
    #[must_use]
    pub const fn level_number(&self) -> u8 {
        match self {
            Self::SemanticIr { .. } => 1,
            Self::Optimizer { .. } => 2,
            Self::Lowering { .. } => 3,
            Self::MegakernelPlan { .. } => 4,
            Self::TargetEmission { .. } => 5,
            Self::DriverRuntime { .. } => 6,
        }
    }

    /// Stable label for this level.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::SemanticIr { .. } => "level_1_semantic_ir",
            Self::Optimizer { .. } => "level_2_optimizer",
            Self::Lowering { .. } => "level_3_lowering",
            Self::MegakernelPlan { .. } => "level_4_megakernel_plan",
            Self::TargetEmission { .. } => "level_5_target_emission",
            Self::DriverRuntime { .. } => "driver_runtime",
        }
    }

    /// Render human-readable summary.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::SemanticIr { name, node_count, op_count, buffers } => {
                format!(
                    "Level 1 (Semantic IR): name={name}, nodes={node_count}, ops={op_count}, buffers={buffers:?}"
                )
            }
            Self::Optimizer { pass_count, proof_citations, inferred_facts } => {
                format!(
                    "Level 2 (Optimizer): passes={pass_count}, proofs={proof_citations:?}, facts={inferred_facts:?}"
                )
            }
            Self::Lowering { entry_point, op_count, bindings, workgroup_size } => {
                format!(
                    "Level 3 (Lowering): entry={entry_point}, ops={op_count}, bindings={bindings:?}, workgroup_size={workgroup_size:?}"
                )
            }
            Self::MegakernelPlan { digest, fusion_groups, barrier_count, resource_count, candidates_evaluated, candidates_pruned } => {
                format!(
                    "Level 4 (Megakernel Plan): digest={digest}, fusion_groups={fusion_groups}, barriers={barrier_count}, resources={resource_count}, evaluated={candidates_evaluated}, pruned={candidates_pruned}"
                )
            }
            Self::TargetEmission { format, format_version, byte_size, entry_points } => {
                format!(
                    "Level 5 (Target Emission): format={format} v{format_version}, size={byte_size} bytes, entries={entry_points:?}"
                )
            }
            Self::DriverRuntime { status, resident_resources, completed_submissions } => {
                format!(
                    "Driver/Runtime: status={status}, resident_resources={resident_resources}, completed={completed_submissions}"
                )
            }
        }
    }
}

/// Structural difference between two compiler level views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerLevelDiff {
    /// Level being diffed.
    pub level_number: u8,
    /// Whether the two views have matching representations.
    pub is_identical: bool,
    /// Detailed field-by-field differences.
    pub deltas: Vec<String>,
}

/// Compare two compiler level views structurally.
#[must_use]
pub fn diff_compiler_levels(
    before: &CompilerLevelView,
    after: &CompilerLevelView,
) -> CompilerLevelDiff {
    let mut deltas = Vec::new();
    if before.level_number() != after.level_number() {
        deltas.push(format!(
            "level mismatch: before={} ({}), after={} ({})",
            before.level_number(),
            before.label(),
            after.level_number(),
            after.label()
        ));
        return CompilerLevelDiff {
            level_number: before.level_number(),
            is_identical: false,
            deltas,
        };
    }
    match (before, after) {
        (
            CompilerLevelView::SemanticIr { name: n1, node_count: nc1, op_count: oc1, buffers: b1 },
            CompilerLevelView::SemanticIr { name: n2, node_count: nc2, op_count: oc2, buffers: b2 },
        ) => {
            if n1 != n2 { deltas.push(format!("name: `{n1}` -> `{n2}`")); }
            if nc1 != nc2 { deltas.push(format!("node_count: {nc1} -> {nc2}")); }
            if oc1 != oc2 { deltas.push(format!("op_count: {oc1} -> {oc2}")); }
            if b1 != b2 { deltas.push(format!("buffers: {b1:?} -> {b2:?}")); }
        }
        (
            CompilerLevelView::Optimizer { pass_count: p1, proof_citations: pr1, inferred_facts: f1 },
            CompilerLevelView::Optimizer { pass_count: p2, proof_citations: pr2, inferred_facts: f2 },
        ) => {
            if p1 != p2 { deltas.push(format!("pass_count: {p1} -> {p2}")); }
            if pr1 != pr2 { deltas.push(format!("proofs: {pr1:?} -> {pr2:?}")); }
            if f1 != f2 { deltas.push(format!("facts: {f1:?} -> {f2:?}")); }
        }
        (
            CompilerLevelView::Lowering { entry_point: e1, op_count: o1, bindings: b1, workgroup_size: w1 },
            CompilerLevelView::Lowering { entry_point: e2, op_count: o2, bindings: b2, workgroup_size: w2 },
        ) => {
            if e1 != e2 { deltas.push(format!("entry_point: `{e1}` -> `{e2}`")); }
            if o1 != o2 { deltas.push(format!("op_count: {o1} -> {o2}")); }
            if b1 != b2 { deltas.push(format!("bindings: {b1:?} -> {b2:?}")); }
            if w1 != w2 { deltas.push(format!("workgroup_size: {w1:?} -> {w2:?}")); }
        }
        (
            CompilerLevelView::MegakernelPlan { digest: d1, fusion_groups: f1, barrier_count: b1, resource_count: r1, candidates_evaluated: ce1, candidates_pruned: cp1 },
            CompilerLevelView::MegakernelPlan { digest: d2, fusion_groups: f2, barrier_count: b2, resource_count: r2, candidates_evaluated: ce2, candidates_pruned: cp2 },
        ) => {
            if d1 != d2 { deltas.push(format!("digest: {d1} -> {d2}")); }
            if f1 != f2 { deltas.push(format!("fusion_groups: {f1} -> {f2}")); }
            if b1 != b2 { deltas.push(format!("barriers: {b1} -> {b2}")); }
            if r1 != r2 { deltas.push(format!("resources: {r1} -> {r2}")); }
            if ce1 != ce2 { deltas.push(format!("candidates_evaluated: {ce1} -> {ce2}")); }
            if cp1 != cp2 { deltas.push(format!("candidates_pruned: {cp1} -> {cp2}")); }
        }
        (
            CompilerLevelView::TargetEmission { format: f1, format_version: v1, byte_size: s1, entry_points: e1 },
            CompilerLevelView::TargetEmission { format: f2, format_version: v2, byte_size: s2, entry_points: e2 },
        ) => {
            if f1 != f2 { deltas.push(format!("format: `{f1}` -> `{f2}`")); }
            if v1 != v2 { deltas.push(format!("version: {v1} -> {v2}")); }
            if s1 != s2 { deltas.push(format!("byte_size: {s1} -> {s2}")); }
            if e1 != e2 { deltas.push(format!("entry_points: {e1:?} -> {e2:?}")); }
        }
        (
            CompilerLevelView::DriverRuntime { status: s1, resident_resources: r1, completed_submissions: c1 },
            CompilerLevelView::DriverRuntime { status: s2, resident_resources: r2, completed_submissions: c2 },
        ) => {
            if s1 != s2 { deltas.push(format!("status: `{s1}` -> `{s2}`")); }
            if r1 != r2 { deltas.push(format!("resident_resources: {r1} -> {r2}")); }
            if c1 != c2 { deltas.push(format!("completed: {c1} -> {c2}")); }
        }
        _ => {}
    }
    CompilerLevelDiff {
        level_number: before.level_number(),
        is_identical: deltas.is_empty(),
        deltas,
    }
}
