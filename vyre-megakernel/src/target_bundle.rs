//! Target module bundle and image representations.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use vyre_foundation::{
    fp_parity::approximable_operations,
    ir::{Expr, Program},
    numeric::{ScalarFormat, NUMERIC_CONTRACT_VERSION},
    schedule::SchedulePhase,
    visit::walk_exprs,
};
use vyre_lower::{KernelDescriptor, MemoryClass};

use super::TargetCompileError;
use crate::candidate::ExecutionTopology;
use crate::{ArtifactAbi, ArtifactNodeId, FusionGroupId, TargetResourceBinding};

/// One compiler-selected group decoded into verified semantic modules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedModule {
    /// Stable selected group identity.
    pub group: FusionGroupId,
    /// Dependency stage selected by the whole-program planner.
    pub stage: u32,
    /// Typed graph node identities in deterministic emission order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Canonical Programs corresponding one-for-one with `nodes`.
    pub programs: Vec<Program>,
}

/// One compiler-selected group after canonical semantic optimization and
/// verified representation lowering.
#[derive(Clone, Debug)]
pub struct SelectedLowering {
    /// Exact neutral artifact identity.
    pub artifact: crate::Digest,
    /// Stable selected group identity.
    pub group: FusionGroupId,
    /// Dependency stage selected by the whole-program planner.
    pub stage: u32,
    /// Typed graph node identities in deterministic emission order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Exact backend-neutral selected schedule phase lowered into this kernel.
    pub schedule_phase: SchedulePhase,
    /// Verified backend-neutral physical kernel consumed by concrete emitters.
    pub(crate) physical: vyre_lower::PhysicalKernel,
    /// Canonical ABI slice for this selected group.
    pub abi: ArtifactAbi,
    /// Canonical descriptor-to-artifact resource association.
    pub canonical_bindings: Vec<TargetResourceBinding>,
    /// Authoritative logical invocation span before target grid projection.
    pub logical_element_count: u32,
    /// Selected frontier-density traversal topology.
    pub frontier_topology: crate::candidate::FrontierTopology,
    pub(crate) program: Program,
}

impl SelectedLowering {
    /// Borrow the verified physical descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &KernelDescriptor {
        self.physical.descriptor()
    }
}

/// Canonical target-module bundle schema carried inside one target payload.
pub const TARGET_MODULE_BUNDLE_SCHEMA_VERSION: u16 = 5;

/// Which executable arm one selected module is submitted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetArmAssignment {
    /// Selected fusion group this assignment binds.
    pub group: FusionGroupId,
    /// Dependency stage the group executes in.
    pub stage: u32,
    /// Queue or spatial partition index carrying the module.
    pub arm: u32,
}

/// What one lowered module does to the numbers it computes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleNumericRecord {
    /// Numeric contract shape this record is stated under.
    pub version: u32,
    /// Scalar formats the module holds values in, in ascending order.
    pub formats: Vec<ScalarFormat>,
    /// Scalar formats the module converts values to, in ascending order.
    pub conversions: Vec<ScalarFormat>,
    /// Operations the module computes through the approximation window, named
    /// by their neutral IR variant, in ascending order.
    pub approximations: Vec<String>,
    /// Elements one invocation combines, when the phase selected more than one.
    pub chunk: Option<u32>,
}

impl ModuleNumericRecord {
    /// The numeric choices lowering made for one emitted module.
    #[must_use]
    pub fn of(program: &Program, phase: &SchedulePhase) -> Self {
        let mut formats = BTreeSet::new();
        let mut conversions = BTreeSet::new();
        for buffer in program.buffers() {
            if let Some(format) = ScalarFormat::of(&buffer.element()) {
                formats.insert(format);
            }
        }
        walk_exprs(program, |expr| {
            if let Expr::Cast { target, .. } = expr {
                if let Some(format) = ScalarFormat::of(target) {
                    conversions.insert(format);
                    formats.insert(format);
                }
            }
        });
        Self {
            version: NUMERIC_CONTRACT_VERSION,
            formats: formats.into_iter().collect(),
            conversions: conversions.into_iter().collect(),
            approximations: approximable_operations(program),
            chunk: (phase.vector_width > 1).then_some(phase.vector_width),
        }
    }
}

/// One generated target module corresponding to one selected fusion group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleImage {
    /// Stable selected fusion group.
    pub group: FusionGroupId,
    /// Dependency stage of this module.
    pub stage: u32,
    /// Exact selected node identities in deterministic order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Canonical optimized Program wire consumed without semantic re-lowering.
    pub program: Vec<u8>,
    /// Verified physical descriptor consumed by materializers without re-lowering.
    pub descriptor: KernelDescriptor,
    /// Target entry-point name.
    pub entry_point: String,
    /// Numeric choices lowering made for this module.
    pub numeric: ModuleNumericRecord,
    /// Immutable target-native module bytes.
    pub bytes: Vec<u8>,
}

impl TargetModuleImage {
    /// Resolve a Program buffer name to the exact target `(group, slot)`.
    #[must_use]
    pub fn binding_slot(&self, name: &str) -> Option<(u32, u32)> {
        let mut found = None;
        for slot in &self.descriptor.bindings.slots {
            if slot.name != name {
                continue;
            }
            let group = match slot.memory_class {
                MemoryClass::Shared | MemoryClass::Scratch => continue,
                MemoryClass::Uniform => 1,
                MemoryClass::Global | MemoryClass::Constant => 0,
            };
            if found.replace((group, slot.slot)).is_some() {
                return None;
            }
        }
        found
    }
}

/// Canonical ordered target modules for one neutral artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleBundle {
    /// Bundle schema.
    pub schema_version: u16,
    /// Execution topology the compiler selected for this artifact.
    pub topology: ExecutionTopology,
    /// Arm carrying each module, ordered with `modules`.
    pub arms: Vec<TargetArmAssignment>,
    /// Modules ordered by dependency stage and fusion-group identity.
    pub modules: Vec<TargetModuleImage>,
}

pub(crate) fn assign_arms(
    topology: ExecutionTopology,
    modules: &[TargetModuleImage],
) -> Vec<TargetArmAssignment> {
    let width = topology.arm_width();
    let mut arms = Vec::with_capacity(modules.len());
    let mut stage = None;
    let mut within = 0u32;
    for module in modules {
        if stage != Some(module.stage) {
            stage = Some(module.stage);
            within = 0;
        }
        arms.push(TargetArmAssignment {
            group: module.group,
            stage: module.stage,
            arm: within % width,
        });
        within = within.saturating_add(1);
    }
    arms
}

impl TargetModuleBundle {
    /// Construct and canonically order target modules on the sequential baseline.
    #[must_use]
    pub fn new(modules: Vec<TargetModuleImage>) -> Self {
        Self::with_topology(ExecutionTopology::Sequential, modules)
    }

    /// Construct and canonically order target modules under a selected topology.
    #[must_use]
    pub fn with_topology(topology: ExecutionTopology, mut modules: Vec<TargetModuleImage>) -> Self {
        modules.sort_by_key(|module| (module.stage, module.group));
        let arms = assign_arms(topology, &modules);
        Self {
            schema_version: TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
            topology,
            arms,
            modules,
        }
    }

    /// Encode canonical target-module bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TargetCompileError> {
        let body = serde_json::to_vec(self)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))?;
        let digest = blake3::hash(&body);
        let mut bytes = Vec::with_capacity(32 + body.len());
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }

    /// Decode and validate canonical target-module bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TargetCompileError> {
        let (expected, body) = bytes.split_at_checked(32).ok_or_else(|| {
            TargetCompileError::ModuleBundle("target module bundle is truncated".to_string())
        })?;
        let actual = blake3::hash(body);
        if actual.as_bytes() != expected {
            return Err(TargetCompileError::ModuleBundle(
                "target module bundle digest mismatch".to_string(),
            ));
        }
        let bundle: Self = serde_json::from_slice(body)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))?;
        if bundle.schema_version != TARGET_MODULE_BUNDLE_SCHEMA_VERSION {
            return Err(TargetCompileError::ModuleBundle(format!(
                "schema {} is unsupported; expected {}",
                bundle.schema_version, TARGET_MODULE_BUNDLE_SCHEMA_VERSION
            )));
        }
        for module in &bundle.modules {
            if module.nodes.is_empty() {
                return Err(TargetCompileError::ModuleBundle(format!(
                    "fusion group {} has no selected nodes",
                    module.group.0
                )));
            }
            Program::from_wire(&module.program).map_err(|error| {
                TargetCompileError::ModuleBundle(format!(
                    "fusion group {} selected Program is malformed: {error}",
                    module.group.0
                ))
            })?;
            vyre_lower::verify_descriptor(&module.descriptor).map_err(|error| {
                TargetCompileError::ModuleBundle(format!(
                    "fusion group {} descriptor is invalid: {error:?}",
                    module.group.0
                ))
            })?;
        }
        if bundle.modules.windows(2).any(|modules| {
            (modules[0].stage, modules[0].group) >= (modules[1].stage, modules[1].group)
        }) {
            return Err(TargetCompileError::ModuleBundle(
                "module bundle is not in canonical stage/group order".to_string(),
            ));
        }
        if bundle.arms != assign_arms(bundle.topology, &bundle.modules) {
            return Err(TargetCompileError::ModuleBundle(
                "module bundle arms are not the selected topology's assignment".to_string(),
            ));
        }
        let canonical = bundle.to_bytes()?;
        if canonical != bytes {
            return Err(TargetCompileError::ModuleBundle(
                "module bundle is not in canonical stage/group order".to_string(),
            ));
        }
        Ok(bundle)
    }
}
