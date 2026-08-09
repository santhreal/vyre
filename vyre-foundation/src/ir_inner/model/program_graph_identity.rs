//! Canonical content identity for connected Program compositions.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::program_graph::{ProgramGraph, ProgramGraphError, ShapeDim, ValueLifetime};
use super::program_graph_analysis::ProgramGraphAnalysisError;

/// Current domain-separated composition identity format.
pub const PROGRAM_GRAPH_IDENTITY_VERSION: u16 = 2;

/// Stable caller-supplied facts outside graph topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramGraphIdentityContext {
    /// Version of the consuming compiled-artifact schema.
    pub artifact_schema_version: u32,
    /// Digest of complete validated semantic configuration outside the graph.
    pub configuration_digest: [u8; 32],
    /// Exact concrete value for every symbolic graph dimension.
    pub symbolic_bindings: BTreeMap<String, u64>,
    /// Exact verified content digest for every constant graph value.
    pub constant_identities: BTreeMap<String, [u8; 32]>,
}

/// Versioned identity suitable for compiled-artifact and residency keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramGraphIdentity {
    /// Identity format version used to derive `digest`.
    pub format_version: u16,
    /// Domain-separated BLAKE3 digest.
    pub digest: [u8; 32],
}

/// Refusal to derive identity from incomplete or inconsistent provenance.
#[derive(Debug, Error)]
pub enum ProgramGraphIdentityError {
    /// Whole-graph structure is invalid.
    #[error("program graph identity rejected invalid topology: {source}")]
    InvalidGraph {
        /// Structural counterexample.
        #[source]
        source: ProgramGraphAnalysisError,
    },
    /// Graph could not produce canonical VGR0 bytes.
    #[error("program graph identity could not encode canonical topology: {source}")]
    NonCanonicalGraph {
        /// Wire-format error.
        #[source]
        source: ProgramGraphError,
    },
    /// A graph shape symbol has no concrete binding.
    #[error("program graph identity is missing symbolic binding `{symbol}`")]
    MissingSymbol {
        /// Missing symbol.
        symbol: String,
    },
    /// Context includes a binding unused by graph contracts.
    #[error("program graph identity has unexpected symbolic binding `{symbol}`")]
    UnexpectedSymbol {
        /// Unexpected symbol.
        symbol: String,
    },
    /// A constant graph value has no verified content identity.
    #[error("program graph identity is missing constant identity `{name}`")]
    MissingConstantIdentity {
        /// Missing constant value name.
        name: String,
    },
    /// Context includes an identity for a non-constant graph value.
    #[error("program graph identity has unexpected constant identity `{name}`")]
    UnexpectedConstantIdentity {
        /// Unexpected constant value name.
        name: String,
    },
    /// A length does not fit the stable u64 framing.
    #[error("program graph identity input length exceeds u64")]
    LengthOverflow,
}

impl ProgramGraph {
    /// Derive canonical composition identity from graph and external provenance.
    ///
    /// Mutable retained contents are deliberately absent. Retained schemas and
    /// transitions remain part of the graph bytes, so state changes do not churn
    /// compiled-artifact identity while a contract change does.
    pub fn identity(
        &self,
        context: &ProgramGraphIdentityContext,
    ) -> Result<ProgramGraphIdentity, ProgramGraphIdentityError> {
        self.analyze()
            .map_err(|source| ProgramGraphIdentityError::InvalidGraph { source })?;
        validate_context(self, context)?;
        let graph_wire = self
            .to_wire()
            .map_err(|source| ProgramGraphIdentityError::NonCanonicalGraph { source })?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-program-graph-identity\0");
        hasher.update(&PROGRAM_GRAPH_IDENTITY_VERSION.to_le_bytes());
        hasher.update(&context.artifact_schema_version.to_le_bytes());
        update_bytes(&mut hasher, &graph_wire)?;
        hasher.update(&context.configuration_digest);
        update_count(&mut hasher, context.symbolic_bindings.len())?;
        for (symbol, value) in &context.symbolic_bindings {
            update_bytes(&mut hasher, symbol.as_bytes())?;
            hasher.update(&value.to_le_bytes());
        }
        update_count(&mut hasher, context.constant_identities.len())?;
        for (name, identity) in &context.constant_identities {
            update_bytes(&mut hasher, name.as_bytes())?;
            hasher.update(identity);
        }
        Ok(ProgramGraphIdentity {
            format_version: PROGRAM_GRAPH_IDENTITY_VERSION,
            digest: *hasher.finalize().as_bytes(),
        })
    }
}

fn validate_context(
    graph: &ProgramGraph,
    context: &ProgramGraphIdentityContext,
) -> Result<(), ProgramGraphIdentityError> {
    let symbols = graph
        .values()
        .iter()
        .flat_map(|value| &value.contract.shape)
        .filter_map(|dimension| match dimension {
            ShapeDim::Symbol(symbol) => Some(symbol.as_str()),
            ShapeDim::Known(_) => None,
        })
        .collect::<BTreeSet<_>>();
    for symbol in &symbols {
        if !context.symbolic_bindings.contains_key(*symbol) {
            return Err(ProgramGraphIdentityError::MissingSymbol {
                symbol: (*symbol).to_owned(),
            });
        }
    }
    for symbol in context.symbolic_bindings.keys() {
        if !symbols.contains(symbol.as_str()) {
            return Err(ProgramGraphIdentityError::UnexpectedSymbol {
                symbol: symbol.clone(),
            });
        }
    }

    let constants = graph
        .values()
        .iter()
        .filter(|value| value.contract.lifetime == ValueLifetime::Constant)
        .map(|value| value.name.as_str())
        .collect::<BTreeSet<_>>();
    for name in &constants {
        if !context.constant_identities.contains_key(*name) {
            return Err(ProgramGraphIdentityError::MissingConstantIdentity {
                name: (*name).to_owned(),
            });
        }
    }
    for name in context.constant_identities.keys() {
        if !constants.contains(name.as_str()) {
            return Err(ProgramGraphIdentityError::UnexpectedConstantIdentity {
                name: name.clone(),
            });
        }
    }
    Ok(())
}

fn update_count(
    hasher: &mut blake3::Hasher,
    count: usize,
) -> Result<(), ProgramGraphIdentityError> {
    let count = u64::try_from(count).map_err(|_| ProgramGraphIdentityError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    Ok(())
}

fn update_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), ProgramGraphIdentityError> {
    update_count(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}
