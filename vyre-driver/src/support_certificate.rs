//! Production-path support certificate for backend operations.
//!
//! # Production-path Join
//!
//! Backend support is NOT derived from registration or tier alone. Support is a
//! certificate joined from seven required production-path facts:
//!
//! 1. **Validation**: The program and its schedule pass structural and typed IR validation.
//! 2. **Emission**: Lowering and target dialect emission succeed without error.
//! 3. **Native Toolchain Compilation**: The backend native toolchain validates and compiles the emitted text into a native module.
//! 4. **Materialization**: Device module and pipeline creation succeed on the target device/runtime.
//! 5. **Hostile Binding Verification**: Adversarial binding tests (OOB, invalid buffers, mismatched permissions) fail closed without device corruption.
//! 6. **Device Execution**: The kernel executes on an authenticated device with verified launch geometry.
//! 7. **Independent Oracle Agreement**: The device execution output matches the independent semantic oracle (`vyre-reference`) within the operation's declared numerical tolerance.
//!
//! An operation missing ANY one of these facts is **unsupported** on that backend.

use std::collections::{BTreeMap, HashSet};
use std::sync::{LazyLock, RwLock};

use serde::{Deserialize, Serialize};
use vyre_foundation::failure_domain::{
    govern_rwlock_read, govern_rwlock_write_restartable, RecoveryClass,
};
use vyre_foundation::ir::OpId;

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "driver support certificate registry";

/// The state every poison report in this module names.
const CERTIFICATE_TABLE: &str = "the backend support certificate table";

/// Schema version for production path support certificates.
pub const SUPPORT_CERTIFICATE_SCHEMA: u32 = 1;

/// Stage of the production path required for backend support certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProductionPathStage {
    /// Step 1: Program and IR structural/typed validation.
    Validation,
    /// Step 2: Target-specific physical IR lowering and dialect emission.
    Emission,
    /// Step 3: Native toolchain compilation of emitted text into a native module.
    NativeCompilation,
    /// Step 4: Device module creation and pipeline materialization.
    Materialization,
    /// Step 5: Hostile/adversarial binding validation.
    HostileBindings,
    /// Step 6: Authenticated hardware device execution.
    DeviceExecution,
    /// Step 7: Independent reference oracle output parity within declared tolerance.
    OracleAgreement,
}

impl ProductionPathStage {
    /// All required production path stages in order.
    pub const ALL: [ProductionPathStage; 7] = [
        ProductionPathStage::Validation,
        ProductionPathStage::Emission,
        ProductionPathStage::NativeCompilation,
        ProductionPathStage::Materialization,
        ProductionPathStage::HostileBindings,
        ProductionPathStage::DeviceExecution,
        ProductionPathStage::OracleAgreement,
    ];

    /// Name of the stage as a static str.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Emission => "emission",
            Self::NativeCompilation => "native_compilation",
            Self::Materialization => "materialization",
            Self::HostileBindings => "hostile_bindings",
            Self::DeviceExecution => "device_execution",
            Self::OracleAgreement => "oracle_agreement",
        }
    }
}

/// Status of an individual production-path verification fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactStatus {
    /// The stage passed with verified proof.
    Proven {
        /// Deterministic proof signature or hash digest.
        proof_digest: String,
        /// Detail notes or metrics.
        details: Option<String>,
    },
    /// The stage was attempted and failed.
    Failed {
        /// Failure reason.
        reason: String,
    },
    /// The stage has not been proven on an authenticated device/path.
    Unproven,
}

impl FactStatus {
    /// Return true when this fact is proven.
    #[must_use]
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven { .. })
    }
}

/// A single production-path verification fact for an operation on a backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductionPathFact {
    /// Production path stage.
    pub stage: ProductionPathStage,
    /// Verification status.
    pub status: FactStatus,
}

/// A complete, versioned support certificate for an operation on a backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportCertificate {
    /// Certificate schema version.
    pub schema: u32,
    /// Target backend identifier.
    pub backend_id: String,
    /// Target identifier.
    pub target_id: String,
    /// Operation identifier.
    pub op_id: OpId,
    /// Facts recorded for each stage.
    pub facts: BTreeMap<ProductionPathStage, ProductionPathFact>,
}

/// Support status evaluated from a support certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportStatus {
    /// Fully supported: all 7 production path facts are proven.
    Supported {
        /// Certificate summary / digest.
        certificate_digest: String,
    },
    /// Unsupported: one or more production path stages are missing or failed.
    Unsupported {
        /// First missing or failed stage.
        missing_stage: ProductionPathStage,
        /// Detailed reason.
        reason: String,
    },
}

impl SupportStatus {
    /// Return true when fully supported.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

impl SupportCertificate {
    /// Create a new empty certificate with all stages unproven.
    #[must_use]
    pub fn new(backend_id: impl Into<String>, target_id: impl Into<String>, op_id: OpId) -> Self {
        let backend_id = backend_id.into();
        let target_id = target_id.into();
        let mut facts = BTreeMap::new();
        for stage in ProductionPathStage::ALL {
            facts.insert(
                stage,
                ProductionPathFact {
                    stage,
                    status: FactStatus::Unproven,
                },
            );
        }
        Self {
            schema: SUPPORT_CERTIFICATE_SCHEMA,
            backend_id,
            target_id,
            op_id,
            facts,
        }
    }

    /// Record a proven stage.
    #[must_use]
    pub fn with_proven_stage(
        mut self,
        stage: ProductionPathStage,
        proof_digest: impl Into<String>,
        details: Option<String>,
    ) -> Self {
        self.facts.insert(
            stage,
            ProductionPathFact {
                stage,
                status: FactStatus::Proven {
                    proof_digest: proof_digest.into(),
                    details,
                },
            },
        );
        self
    }

    /// Record a failed stage.
    #[must_use]
    pub fn with_failed_stage(
        mut self,
        stage: ProductionPathStage,
        reason: impl Into<String>,
    ) -> Self {
        self.facts.insert(
            stage,
            ProductionPathFact {
                stage,
                status: FactStatus::Failed {
                    reason: reason.into(),
                },
            },
        );
        self
    }

    /// Evaluate whether the certificate represents complete support.
    ///
    /// Evaluates the join across all 7 stages. If ANY stage is unproven or failed,
    /// returns `SupportStatus::Unsupported` naming the first unproven/failed stage.
    #[must_use]
    pub fn evaluate(&self) -> SupportStatus {
        for stage in ProductionPathStage::ALL {
            match self.facts.get(&stage) {
                Some(fact) => match &fact.status {
                    FactStatus::Proven { .. } => continue,
                    FactStatus::Failed { reason } => {
                        return SupportStatus::Unsupported {
                            missing_stage: stage,
                            reason: format!("Stage `{}` failed: {reason}", stage.name()),
                        };
                    }
                    FactStatus::Unproven => {
                        return SupportStatus::Unsupported {
                            missing_stage: stage,
                            reason: format!(
                                "Stage `{}` is unproven on production path",
                                stage.name()
                            ),
                        };
                    }
                },
                None => {
                    return SupportStatus::Unsupported {
                        missing_stage: stage,
                        reason: format!(
                            "Stage `{}` fact is missing from certificate",
                            stage.name()
                        ),
                    };
                }
            }
        }
        let digest = self.compute_digest();
        SupportStatus::Supported {
            certificate_digest: digest,
        }
    }

    /// Compute deterministic blake3 digest of the certificate.
    #[must_use]
    pub fn compute_digest(&self) -> String {
        let serialized = serde_json::to_string(self).unwrap_or_default();
        blake3::hash(serialized.as_bytes()).to_hex().to_string()
    }
}

/// Global registry of backend support certificates and joined support sets.
#[derive(Default)]
pub struct SupportCertificateRegistry {
    certificates: RwLock<BTreeMap<(String, OpId), SupportCertificate>>,
}

impl SupportCertificateRegistry {
    /// Return the global singleton registry.
    #[must_use]
    pub fn global() -> &'static Self {
        static REGISTRY: LazyLock<SupportCertificateRegistry> =
            LazyLock::new(SupportCertificateRegistry::default);
        &REGISTRY
    }

    /// Register or update a support certificate.
    ///
    /// This is the documented recovery for a poisoned table: a certificate is
    /// proof that a backend lowers an op, a half-written table is not proof, so
    /// registration discards what a panic left and starts from the certificate
    /// it was handed. Every discarded entry is re-registered by the backend
    /// that declared it.
    pub fn register_certificate(&self, cert: SupportCertificate) {
        let mut certs = govern_rwlock_write_restartable(
            &self.certificates,
            OWNER,
            CERTIFICATE_TABLE,
            BTreeMap::clear,
        );
        certs.insert((cert.backend_id.clone(), cert.op_id.clone()), cert);
    }

    /// Evaluate support status for an operation on a backend.
    ///
    /// A poisoned table reports unsupported and names the poison, because a
    /// table a panic left half written proves nothing and reporting it as an
    /// absent certificate would name a condition this call never observed.
    #[must_use]
    pub fn evaluate_support(&self, backend_id: &str, op_id: &OpId) -> SupportStatus {
        let certs = match govern_rwlock_read(
            &self.certificates,
            OWNER,
            CERTIFICATE_TABLE,
            RecoveryClass::RestartableFromCanonicalInput,
        ) {
            Ok(certs) => certs,
            Err(error) => {
                return SupportStatus::Unsupported {
                    missing_stage: ProductionPathStage::Validation,
                    reason: error.to_string(),
                }
            }
        };
        if let Some(cert) = certs.get(&(backend_id.to_string(), op_id.clone())) {
            return cert.evaluate();
        }
        SupportStatus::Unsupported {
            missing_stage: ProductionPathStage::Validation,
            reason: format!(
                "No support certificate registered for backend `{backend_id}` and op `{op_id}`"
            ),
        }
    }

    /// Get all supported operation IDs for a backend (joined from certificates).
    ///
    /// A poisoned table yields the empty set: no op is proven supported by a
    /// table a panic left half written.
    #[must_use]
    pub fn supported_ops_for_backend(&self, backend_id: &str) -> HashSet<OpId> {
        let mut supported = HashSet::new();
        let Ok(certs) = govern_rwlock_read(
            &self.certificates,
            OWNER,
            CERTIFICATE_TABLE,
            RecoveryClass::RestartableFromCanonicalInput,
        ) else {
            return supported;
        };
        for ((b_id, op_id), cert) in certs.iter() {
            if b_id == backend_id && cert.evaluate().is_supported() {
                supported.insert(op_id.clone());
            }
        }
        supported
    }
}
