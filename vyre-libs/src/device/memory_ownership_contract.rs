//! Host/device memory ownership contract validation.

use std::collections::BTreeSet;

/// Allowed owner for a buffer at a system boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryOwner {
    /// Caller owns the host-visible input/output allocation.
    HostCaller,
    /// The backend owns a resident device allocation.
    DeviceResident,
    /// Runtime owns pinned staging for transfers only.
    PinnedStaging,
    /// Caller owns output slots reused across dispatches.
    BorrowedOutputSlot,
    /// CPU/reference memory exists only inside parity tests.
    ParityOnly,
}

/// One buffer ownership declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryOwnershipRecord<'a> {
    /// Stable buffer or resource name.
    pub resource: &'a str,
    /// Owning subsystem.
    pub subsystem: &'a str,
    /// Declared memory owner.
    pub owner: MemoryOwner,
    /// Whether this record is for production code.
    pub production: bool,
}

/// Memory ownership proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryOwnershipProof {
    /// Number of ownership records validated.
    pub record_count: usize,
    /// Number of device-resident records.
    pub device_resident_count: usize,
    /// Number of borrowed output-slot records.
    pub borrowed_output_slot_count: usize,
}

/// Memory ownership validation errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryOwnershipError {
    /// No ownership records were supplied.
    EmptyRecords,
    /// Required metadata is empty.
    EmptyMetadata {
        /// Resource name.
        resource: String,
        /// Field name.
        field: &'static str,
    },
    /// Resource is declared more than once.
    DuplicateResource {
        /// Resource name.
        resource: String,
    },
    /// Parity-only memory is production-visible.
    ParityOnlyInProduction {
        /// Resource name.
        resource: String,
    },
    /// Release contract lacks device-resident ownership evidence.
    MissingDeviceResident,
    /// Release contract lacks borrowed output-slot evidence.
    MissingBorrowedOutputSlots,
}

impl std::fmt::Display for MemoryOwnershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRecords => write!(
                f,
                "memory ownership contract has no records. Fix: declare host caller, device resident, pinned staging, borrowed output, and parity-only boundaries."
            ),
            Self::EmptyMetadata { resource, field } => write!(
                f,
                "memory ownership record `{resource}` has empty {field}. Fix: every resource needs a subsystem and owner."
            ),
            Self::DuplicateResource { resource } => write!(
                f,
                "memory ownership resource `{resource}` is declared more than once. Fix: choose one owner and route other users through that contract."
            ),
            Self::ParityOnlyInProduction { resource } => write!(
                f,
                "memory ownership resource `{resource}` is parity-only but production-visible. Fix: move CPU/reference memory behind a test-only boundary."
            ),
            Self::MissingDeviceResident => write!(
                f,
                "memory ownership contract has no device-resident resources. Fix: release paths must declare resident device ownership explicitly."
            ),
            Self::MissingBorrowedOutputSlots => write!(
                f,
                "memory ownership contract has no borrowed output slots. Fix: repeated dispatch outputs must use caller-owned reusable slots."
            ),
        }
    }
}

impl std::error::Error for MemoryOwnershipError {}

/// Validate host/device memory ownership records.
pub fn validate_memory_ownership_contract(
    records: &[MemoryOwnershipRecord<'_>],
) -> Result<MemoryOwnershipProof, MemoryOwnershipError> {
    if records.is_empty() {
        return Err(MemoryOwnershipError::EmptyRecords);
    }

    let mut resources = BTreeSet::new();
    let mut device_resident_count = 0_usize;
    let mut borrowed_output_slot_count = 0_usize;

    for record in records {
        for (field, value) in [
            ("resource", record.resource),
            ("subsystem", record.subsystem),
        ] {
            if value.trim().is_empty() {
                return Err(MemoryOwnershipError::EmptyMetadata {
                    resource: record.resource.to_owned(),
                    field,
                });
            }
        }
        if !resources.insert(record.resource) {
            return Err(MemoryOwnershipError::DuplicateResource {
                resource: record.resource.to_owned(),
            });
        }
        if record.production && record.owner == MemoryOwner::ParityOnly {
            return Err(MemoryOwnershipError::ParityOnlyInProduction {
                resource: record.resource.to_owned(),
            });
        }
        match record.owner {
            MemoryOwner::DeviceResident => device_resident_count += 1,
            MemoryOwner::BorrowedOutputSlot => borrowed_output_slot_count += 1,
            MemoryOwner::HostCaller | MemoryOwner::PinnedStaging | MemoryOwner::ParityOnly => {}
        }
    }

    if device_resident_count == 0 {
        return Err(MemoryOwnershipError::MissingDeviceResident);
    }
    if borrowed_output_slot_count == 0 {
        return Err(MemoryOwnershipError::MissingBorrowedOutputSlots);
    }

    Ok(MemoryOwnershipProof {
        record_count: records.len(),
        device_resident_count,
        borrowed_output_slot_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_ownership_accepts_release_boundaries() {
        let proof = validate_memory_ownership_contract(&[
            record("frontend-input", "vyrec", MemoryOwner::HostCaller, true),
            record("resident-csr", "backend", MemoryOwner::DeviceResident, true),
            record("upload-stage", "backend", MemoryOwner::PinnedStaging, true),
            record(
                "analysis-output",
                "backend",
                MemoryOwner::BorrowedOutputSlot,
                true,
            ),
            record(
                "reference-oracle",
                "backend-parity-tests",
                MemoryOwner::ParityOnly,
                false,
            ),
        ])
        .expect("Fix: valid memory ownership contract should pass");

        assert_eq!(proof.record_count, 5);
        assert_eq!(proof.device_resident_count, 1);
        assert_eq!(proof.borrowed_output_slot_count, 1);
    }

    #[test]
    fn memory_ownership_rejects_parity_memory_in_production() {
        assert_eq!(
            validate_memory_ownership_contract(&[
                record("resident-csr", "backend", MemoryOwner::DeviceResident, true),
                record(
                    "analysis-output",
                    "backend",
                    MemoryOwner::BorrowedOutputSlot,
                    true,
                ),
                record("cpu-oracle", "backend", MemoryOwner::ParityOnly, true),
            ])
            .expect_err("production parity-only memory should fail"),
            MemoryOwnershipError::ParityOnlyInProduction {
                resource: "cpu-oracle".to_owned(),
            }
        );
    }

    #[test]
    fn memory_ownership_requires_residency_and_borrowed_outputs() {
        assert_eq!(
            validate_memory_ownership_contract(&[record(
                "analysis-output",
                "backend",
                MemoryOwner::BorrowedOutputSlot,
                true,
            )])
            .expect_err("missing device resident record should fail"),
            MemoryOwnershipError::MissingDeviceResident
        );
        assert_eq!(
            validate_memory_ownership_contract(&[record(
                "resident-csr",
                "backend",
                MemoryOwner::DeviceResident,
                true,
            )])
            .expect_err("missing borrowed output slot should fail"),
            MemoryOwnershipError::MissingBorrowedOutputSlots
        );
    }

    fn record<'a>(
        resource: &'a str,
        subsystem: &'a str,
        owner: MemoryOwner,
        production: bool,
    ) -> MemoryOwnershipRecord<'a> {
        MemoryOwnershipRecord {
            resource,
            subsystem,
            owner,
            production,
        }
    }
}
