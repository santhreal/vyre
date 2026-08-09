use serde::{Deserialize, Serialize};
use thiserror::Error;
use vyre_foundation::{execution_plan::fusion::merge_programs_shared, ir::Program};

use crate::{
    Artifact, ArtifactAbi, ArtifactNodeId, CompileError, FusionGroupId, FusionRecord,
    TargetPayload, TargetPayloadFormat,
};

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

/// Canonical target-module bundle schema carried inside one target payload.
pub const TARGET_MODULE_BUNDLE_SCHEMA_VERSION: u16 = 1;

/// One generated target module corresponding to one selected fusion group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleImage {
    /// Stable selected fusion group.
    pub group: FusionGroupId,
    /// Dependency stage of this module.
    pub stage: u32,
    /// Target entry-point name.
    pub entry_point: String,
    /// Immutable target-native module bytes.
    pub bytes: Vec<u8>,
}

/// Canonical ordered target modules for one neutral artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleBundle {
    /// Bundle schema.
    pub schema_version: u16,
    /// Modules ordered by dependency stage and fusion-group identity.
    pub modules: Vec<TargetModuleImage>,
}

impl TargetModuleBundle {
    /// Construct and canonically order target modules.
    #[must_use]
    pub fn new(mut modules: Vec<TargetModuleImage>) -> Self {
        modules.sort_by_key(|module| (module.stage, module.group));
        Self {
            schema_version: TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
            modules,
        }
    }

    /// Encode canonical target-module bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TargetCompileError> {
        serde_json::to_vec(self)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))
    }

    /// Decode and validate canonical target-module bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TargetCompileError> {
        let bundle: Self = serde_json::from_slice(bytes)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))?;
        if bundle.schema_version != TARGET_MODULE_BUNDLE_SCHEMA_VERSION {
            return Err(TargetCompileError::ModuleBundle(format!(
                "schema {} is unsupported; expected {}",
                bundle.schema_version, TARGET_MODULE_BUNDLE_SCHEMA_VERSION
            )));
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

/// Failure produced by a registered target compiler facet.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TargetCompileError {
    /// The neutral artifact could not be decoded into selected modules.
    #[error("target compiler rejected the neutral artifact: {0}")]
    InvalidArtifact(String),
    /// The target cannot represent one selected module or ABI contract.
    #[error("target capability rejected the selected plan: {0}")]
    Unsupported(String),
    /// Verified target lowering or emission failed.
    #[error("target emission failed: {0}")]
    Emission(String),
    /// Canonical target-module bundle encoding or decoding failed.
    #[error("target module bundle failed: {0}")]
    ModuleBundle(String),
    /// The emitted payload violated the canonical payload contract.
    #[error("target payload construction failed: {0}")]
    Payload(#[from] CompileError),
}

/// Pure compiler facet from a selected neutral artifact to immutable target bytes.
pub trait TargetCompiler: Send + Sync {
    /// Exact target payload format produced by this facet.
    fn format(&self) -> &TargetPayloadFormat;

    /// Compile every selected module and project the canonical artifact ABI.
    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError>;
}

/// Decode compiler-selected modules from one authenticated neutral artifact.
///
/// Target compilers use this function instead of reconstructing graph order or
/// reading raw frontend Programs from callers.
pub fn selected_modules(artifact: &Artifact) -> Result<Vec<SelectedModule>, TargetCompileError> {
    artifact
        .fusion()
        .iter()
        .map(|group| decode_group(artifact, group))
        .collect()
}

/// Form one generated semantic Program for a compiler-selected fusion group.
///
/// Programs in a graph composition use shared buffer names for connected
/// values. Shared fusion preserves those dataflow names, alpha-renames local
/// collisions, inserts required intra-kernel barriers, and rejects unsafe
/// geometry or aliasing.
pub fn fuse_selected_module(module: &SelectedModule) -> Result<Program, TargetCompileError> {
    merge_programs_shared(&module.programs).map_err(|error| {
        TargetCompileError::Unsupported(format!(
            "fusion group {} cannot form one target module: {error}",
            module.group.0
        ))
    })
}

/// Canonical ABI supplied unchanged to every target compiler.
#[must_use]
pub const fn artifact_abi(artifact: &Artifact) -> &ArtifactAbi {
    artifact.abi()
}

fn decode_group(
    artifact: &Artifact,
    group: &FusionRecord,
) -> Result<SelectedModule, TargetCompileError> {
    let mut nodes = group.members.clone();
    nodes.sort();
    let programs = nodes
        .iter()
        .map(|node| {
            let record = artifact
                .nodes()
                .iter()
                .find(|record| record.id == *node)
                .ok_or_else(|| {
                    TargetCompileError::InvalidArtifact(format!(
                        "fusion group {} references missing node {}",
                        group.id.0, node.0
                    ))
                })?;
            Program::from_wire(&record.program).map_err(|error| {
                TargetCompileError::InvalidArtifact(format!(
                    "node {} canonical Program failed to decode: {error}",
                    node.0
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SelectedModule {
        group: group.id,
        stage: group.stage,
        nodes,
        programs,
    })
}
