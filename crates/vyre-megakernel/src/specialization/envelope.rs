//! One authenticated container for a whole guarded artifact set.
//!
//! Admitting variants one at a time cannot state what the set is. A consumer
//! that receives three envelopes and a guard table read from somewhere else can
//! be handed two of the three, or a guard table from another compile, and
//! nothing in the bytes contradicts it. So the set is one product: the contract,
//! every guard, every variant, the remainder, and the target identity they were
//! all compiled for, framed and digested together.
//!
//! Decoding re-runs the exclusivity and coverage proofs and re-checks that every
//! variant was produced by one compiler for one graph under one objective. An
//! edited guard, a variant swapped in from another compile, and a set missing
//! the remainder its own contract declares are each rejected before anything
//! runs.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::guard::VariantGuard;
use super::{RemainderKind, SpecializationContract};
use crate::error::{failure, CompilerFailureKind};
use crate::frame;
use crate::identity::Digest;
use crate::schema::Artifact;
use crate::{ArtifactEnvelope, CompileError};

/// Current schema for the envelope that carries one guarded artifact set.
pub const PORTFOLIO_ENVELOPE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariantBody {
    guard: VariantGuard,
    envelope: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortfolioBody {
    schema_version: u16,
    contract: SpecializationContract,
    remainder_kind: RemainderKind,
    target_identity: Digest,
    variants: Vec<VariantBody>,
    remainder: Option<Vec<u8>>,
}

/// A complete guarded artifact set, authenticated as one product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioEnvelope {
    contract: SpecializationContract,
    remainder_kind: RemainderKind,
    target_identity: Digest,
    variants: Vec<(VariantGuard, ArtifactEnvelope)>,
    remainder: Option<ArtifactEnvelope>,
}

impl PortfolioEnvelope {
    /// Start a set under one contract, for one target identity.
    #[must_use]
    pub const fn new(
        contract: SpecializationContract,
        remainder_kind: RemainderKind,
        target_identity: Digest,
    ) -> Self {
        Self {
            contract,
            remainder_kind,
            target_identity,
            variants: Vec::new(),
            remainder: None,
        }
    }

    /// The contract the set was compiled under.
    #[must_use]
    pub const fn contract(&self) -> &SpecializationContract {
        &self.contract
    }

    /// What serves facts no guard admits.
    #[must_use]
    pub const fn remainder_kind(&self) -> RemainderKind {
        self.remainder_kind
    }

    /// The target identity every variant was compiled for.
    #[must_use]
    pub const fn target_identity(&self) -> Digest {
        self.target_identity
    }

    /// Guarded variants in canonical guard order.
    #[must_use]
    pub fn variants(&self) -> &[(VariantGuard, ArtifactEnvelope)] {
        &self.variants
    }

    /// The generic remainder, when the contract declares one.
    #[must_use]
    pub const fn remainder(&self) -> Option<&ArtifactEnvelope> {
        self.remainder.as_ref()
    }

    /// Variant indices in the order selection evaluates them.
    ///
    /// Precedence decides first and the canonical guard order breaks the tie, so
    /// every consumer of one set evaluates it in one order and two consumers
    /// cannot serve one request from two variants.
    #[must_use]
    pub fn evaluation_order(&self) -> Vec<usize> {
        let guards = self
            .variants
            .iter()
            .map(|(guard, _)| guard.clone())
            .collect::<Vec<_>>();
        super::precedence_order(&guards)
    }

    /// Attach one variant under its guard.
    ///
    /// # Errors
    ///
    /// Returns an error when the guard is not valid under the contract, or when
    /// a variant is already attached under it.
    pub fn attach_variant(
        &mut self,
        guard: VariantGuard,
        envelope: ArtifactEnvelope,
    ) -> Result<(), CompileError> {
        self.contract.validate_guard(&guard)?;
        if self.variants.iter().any(|(held, _)| *held == guard) {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.variants",
                "two variants are attached under one guard",
                "attach at most one variant for each guard",
            ));
        }
        self.variants.push((guard, envelope));
        self.variants.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(())
    }

    /// Attach the generic remainder.
    ///
    /// # Errors
    ///
    /// Returns an error when the contract declares the remainder unsupported, or
    /// when a remainder is already attached.
    pub fn attach_remainder(&mut self, envelope: ArtifactEnvelope) -> Result<(), CompileError> {
        if self.remainder_kind == RemainderKind::Unsupported {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.remainder",
                "a generic remainder is attached to a set that declares none",
                "declare a generic remainder in the set, or drop the artifact",
            ));
        }
        if self.remainder.is_some() {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.remainder",
                "two remainders are attached",
                "attach one remainder",
            ));
        }
        self.remainder = Some(envelope);
        Ok(())
    }

    /// Prove the set is complete, unambiguous, and produced by one compile.
    ///
    /// # Errors
    ///
    /// Returns an error when the guard set does not prove, when a declared
    /// generic remainder is absent, when the set holds no artifact, or when two
    /// artifacts disagree on the graph, objective, or compiler that produced
    /// them.
    pub fn seal(&self) -> Result<(), CompileError> {
        let guards = self
            .variants
            .iter()
            .map(|(guard, _)| guard.clone())
            .collect::<Vec<_>>();
        self.contract.prove(&guards, self.remainder_kind)?;
        if self.remainder_kind == RemainderKind::Generic && self.remainder.is_none() {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.remainder",
                "the set declares a generic remainder and carries none",
                "attach the generic remainder, or declare the remainder unsupported",
            ));
        }
        let artifacts = self.artifacts();
        if artifacts.is_empty() {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.variants",
                "the set carries no artifact",
                "attach at least one variant or a generic remainder",
            ));
        }
        let first = artifacts[0].provenance();
        for artifact in &artifacts[1..] {
            let provenance = artifact.provenance();
            if provenance.semantic_graph != first.semantic_graph {
                return Err(disagreement("semantic_graph"));
            }
            if provenance.objective != first.objective {
                return Err(disagreement("objective"));
            }
            if provenance.compiler_version != first.compiler_version {
                return Err(disagreement("compiler_version"));
            }
        }
        let schemas: BTreeSet<u16> = artifacts
            .iter()
            .map(|artifact| artifact.schema_version())
            .collect();
        if schemas.len() > 1 {
            return Err(disagreement("schema_version"));
        }
        Ok(())
    }

    /// Reject a consumer whose authenticated target is not the one the set was
    /// compiled for.
    ///
    /// # Errors
    ///
    /// Returns an error when the identities differ.
    pub fn require_target_identity(&self, identity: Digest) -> Result<(), CompileError> {
        if identity == self.target_identity {
            return Ok(());
        }
        Err(failure(
            CompilerFailureKind::TargetIdentityMismatch,
            "portfolio.target_identity",
            "the authenticated target is not the target this set was compiled for",
            "compile the set for this target, or admit a set compiled for it",
        ))
    }

    /// Encode the complete authenticated set.
    ///
    /// # Errors
    ///
    /// Returns an error when the set does not seal, or when an attached envelope
    /// cannot be encoded.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CompileError> {
        self.seal()?;
        let body = PortfolioBody {
            schema_version: PORTFOLIO_ENVELOPE_SCHEMA_VERSION,
            contract: self.contract.clone(),
            remainder_kind: self.remainder_kind,
            target_identity: self.target_identity,
            variants: self
                .variants
                .iter()
                .map(|(guard, envelope)| {
                    Ok(VariantBody {
                        guard: guard.clone(),
                        envelope: envelope.to_bytes()?,
                    })
                })
                .collect::<Result<_, CompileError>>()?,
            remainder: self
                .remainder
                .as_ref()
                .map(ArtifactEnvelope::to_bytes)
                .transpose()?,
        };
        let body = serde_json::to_vec(&body).map_err(serialization_failure)?;
        Ok(frame::PORTFOLIO
            .encode(PORTFOLIO_ENVELOPE_SCHEMA_VERSION, &body)?
            .bytes)
    }

    /// Decode, authenticate, and re-prove a complete set.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame does not authenticate, when the body is
    /// not canonical, when an attached envelope does not decode, or when the set
    /// does not seal.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let decoded = frame::PORTFOLIO.decode(bytes)?;
        let body: PortfolioBody = serde_json::from_slice(decoded.body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.body",
                error.to_string(),
                "supply canonical portfolio bytes emitted by this crate",
            )
        })?;
        if body.schema_version != decoded.version {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "portfolio.body.schema_version",
                "portfolio body schema disagrees with its framing schema",
                "repackage the set instead of rewriting its framing",
            ));
        }
        if body.contract.schema_version() != super::SPECIALIZATION_SCHEMA_VERSION {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "portfolio.contract.schema_version",
                format!(
                    "the set states specialization schema {} and this build reads {}",
                    body.contract.schema_version(),
                    super::SPECIALIZATION_SCHEMA_VERSION
                ),
                "recompile the set under the current specialization schema",
            ));
        }
        let canonical = serde_json::to_vec(&body).map_err(serialization_failure)?;
        if canonical.as_slice() != decoded.body {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "portfolio.body",
                "portfolio body is not canonical JSON",
                "use bytes emitted by PortfolioEnvelope::to_bytes",
            ));
        }
        let mut envelope = Self::new(body.contract, body.remainder_kind, body.target_identity);
        for variant in body.variants {
            envelope.attach_variant(
                variant.guard,
                ArtifactEnvelope::from_bytes(&variant.envelope)?,
            )?;
        }
        if let Some(remainder) = body.remainder {
            envelope.attach_remainder(ArtifactEnvelope::from_bytes(&remainder)?)?;
        }
        envelope.seal()?;
        Ok(envelope)
    }

    /// Every neutral artifact the set carries.
    fn artifacts(&self) -> Vec<&Artifact> {
        self.variants
            .iter()
            .map(|(_, envelope)| envelope.neutral())
            .chain(self.remainder.iter().map(ArtifactEnvelope::neutral))
            .collect()
    }
}

fn disagreement(field: &str) -> CompileError {
    failure(
        CompilerFailureKind::PortfolioProvenanceMismatch,
        format!("portfolio.provenance.{field}"),
        format!("the set holds artifacts that disagree on `{field}`"),
        "package one compile of one graph, or recompile the whole set together",
    )
}

fn serialization_failure(error: serde_json::Error) -> CompileError {
    failure(
        CompilerFailureKind::MalformedArtifact,
        "portfolio.body",
        error.to_string(),
        "report the compiler defect: canonical portfolio bodies must serialize",
    )
}
