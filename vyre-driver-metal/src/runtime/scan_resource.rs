/// Schema version for Metal resident scan resource table evidence.
pub const METAL_RESIDENT_SCAN_RESOURCE_TABLE_SCHEMA_VERSION: u32 = 1;

const METAL_SCAN_RESOURCE_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const METAL_SCAN_RESOURCE_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Lifetime class for a resident scan table bound through Metal resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalResidentScanResourceLifetime {
    /// The entry is valid only for the current command buffer.
    CommandBuffer,
    /// The entry is valid for a compiled pipeline instance.
    Pipeline,
    /// The entry is valid for a resident scan session until explicitly invalidated.
    ResidentSession,
}

impl MetalResidentScanResourceLifetime {
    /// Stable evidence label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CommandBuffer => "command_buffer",
            Self::Pipeline => "pipeline",
            Self::ResidentSession => "resident_session",
        }
    }

    pub(super) const fn tag(self) -> u64 {
        match self {
            Self::CommandBuffer => 1,
            Self::Pipeline => 2,
            Self::ResidentSession => 3,
        }
    }
}

/// One Metal argument-buffer entry for a resident scan table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetalResidentScanResourceEntry {
    /// Resident pattern table id from the shared scan metadata.
    pub table_id: u64,
    /// Resident verifier id from the shared scan metadata.
    pub verifier_id: u64,
    /// Resident output slab id consumed by the scan batch.
    pub output_slab_id: u64,
    /// Metal argument-buffer entry index used for this resident table.
    pub argument_buffer_entry: u32,
    /// Lifetime class expected for this entry.
    pub lifetime: MetalResidentScanResourceLifetime,
    /// Digest of the shared scan table metadata.
    pub shared_scan_metadata_digest: u64,
    /// Digest of the Metal resource metadata derived from the shared metadata.
    pub metal_resource_metadata_digest: u64,
}

impl MetalResidentScanResourceEntry {
    /// Construct one resident scan resource entry.
    #[must_use]
    pub const fn new(
        table_id: u64,
        verifier_id: u64,
        output_slab_id: u64,
        argument_buffer_entry: u32,
        lifetime: MetalResidentScanResourceLifetime,
        shared_scan_metadata_digest: u64,
        metal_resource_metadata_digest: u64,
    ) -> Self {
        Self {
            table_id,
            verifier_id,
            output_slab_id,
            argument_buffer_entry,
            lifetime,
            shared_scan_metadata_digest,
            metal_resource_metadata_digest,
        }
    }
}

/// Evidence for a validated Metal resident scan resource table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalResidentScanResourceTableEvidence {
    /// Evidence schema version.
    pub schema_version: u32,
    /// Number of resident scan resource entries.
    pub entry_count: u32,
    /// Lifetime required for every entry in the table.
    pub expected_lifetime: MetalResidentScanResourceLifetime,
    /// Number of unique pattern table ids.
    pub table_id_count: u32,
    /// Number of unique verifier ids.
    pub verifier_id_count: u32,
    /// Number of unique output slab ids.
    pub output_slab_id_count: u32,
    /// Number of unique Metal argument-buffer entries.
    pub argument_buffer_entry_count: u32,
    /// True when every Metal resource metadata digest matches shared scan metadata.
    pub shared_metadata_parity: bool,
    /// Deterministic digest of the table entries.
    pub resource_table_digest: u64,
}

impl MetalResidentScanResourceTableEvidence {
    /// Return true when the evidence is complete enough for resident scan claims.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.schema_version == METAL_RESIDENT_SCAN_RESOURCE_TABLE_SCHEMA_VERSION
            && self.entry_count != 0
            && self.argument_buffer_entry_count == self.entry_count
            && self.shared_metadata_parity
            && self.resource_table_digest != 0
    }
}

/// Metal resident scan resource table validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalResidentScanResourceError {
    /// The table contains no entries.
    EmptyTable,
    /// The table has too many entries for the evidence ABI.
    EntryCountOverflow {
        /// Entry count that could not fit u32.
        entry_count: usize,
    },
    /// A required id field is zero.
    ZeroId {
        /// Entry index that failed validation.
        entry_index: usize,
        /// Id field name.
        field: &'static str,
    },
    /// Shared scan metadata digest is absent.
    ZeroSharedMetadataDigest {
        /// Entry index that failed validation.
        entry_index: usize,
    },
    /// Metal resource metadata diverged from shared scan metadata.
    MetadataDigestMismatch {
        /// Entry index that failed validation.
        entry_index: usize,
        /// Digest of shared scan metadata.
        shared_scan_metadata_digest: u64,
        /// Digest of Metal resource metadata.
        metal_resource_metadata_digest: u64,
    },
    /// The entry lifetime does not match the expected table lifetime.
    LifetimeMismatch {
        /// Entry index that failed validation.
        entry_index: usize,
        /// Expected lifetime label.
        expected: MetalResidentScanResourceLifetime,
        /// Actual lifetime label.
        actual: MetalResidentScanResourceLifetime,
    },
    /// A Metal argument-buffer entry is reused by two scan table entries.
    DuplicateArgumentBufferEntry {
        /// Duplicated Metal argument-buffer entry.
        argument_buffer_entry: u32,
    },
}

impl std::fmt::Display for MetalResidentScanResourceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => formatter.write_str(
                "Metal resident scan resource table is empty. Fix: bind at least one resident scan table before dispatch.",
            ),
            Self::EntryCountOverflow { entry_count } => write!(
                formatter,
                "Metal resident scan resource table has {entry_count} entries, which exceeds the u32 evidence ABI. Fix: shard resident scan tables before dispatch."
            ),
            Self::ZeroId { entry_index, field } => write!(
                formatter,
                "Metal resident scan resource table entry {entry_index} has zero {field}. Fix: allocate table, verifier, and output slab ids before building Metal resources."
            ),
            Self::ZeroSharedMetadataDigest { entry_index } => write!(
                formatter,
                "Metal resident scan resource table entry {entry_index} has zero shared metadata digest. Fix: derive Metal resource metadata from the shared scan table metadata."
            ),
            Self::MetadataDigestMismatch {
                entry_index,
                shared_scan_metadata_digest,
                metal_resource_metadata_digest,
            } => write!(
                formatter,
                "Metal resident scan resource table entry {entry_index} metadata digest mismatch shared={shared_scan_metadata_digest:#x} metal={metal_resource_metadata_digest:#x}. Fix: rebuild the Metal argument-buffer entry from shared scan metadata."
            ),
            Self::LifetimeMismatch {
                entry_index,
                expected,
                actual,
            } => write!(
                formatter,
                "Metal resident scan resource table entry {entry_index} has lifetime {} but expected {}. Fix: rebuild the argument-buffer table at one consistent resource lifetime.",
                actual.as_str(),
                expected.as_str()
            ),
            Self::DuplicateArgumentBufferEntry {
                argument_buffer_entry,
            } => write!(
                formatter,
                "Metal resident scan resource table reuses argument-buffer entry {argument_buffer_entry}. Fix: assign one unique Metal argument-buffer entry per resident scan table."
            ),
        }
    }
}

impl std::error::Error for MetalResidentScanResourceError {}

/// Validate resident scan table metadata before binding it through Metal resources.
///
/// # Errors
///
/// Returns [`MetalResidentScanResourceError`] when ids are missing, metadata
/// digests drift, lifetimes are inconsistent, or argument-buffer entries are
/// reused.
pub fn metal_resident_scan_resource_table(
    entries: &[MetalResidentScanResourceEntry],
    expected_lifetime: MetalResidentScanResourceLifetime,
) -> Result<MetalResidentScanResourceTableEvidence, MetalResidentScanResourceError> {
    if entries.is_empty() {
        return Err(MetalResidentScanResourceError::EmptyTable);
    }
    let entry_count = u32::try_from(entries.len()).map_err(|_| {
        MetalResidentScanResourceError::EntryCountOverflow {
            entry_count: entries.len(),
        }
    })?;

    let mut table_ids = std::collections::BTreeSet::new();
    let mut verifier_ids = std::collections::BTreeSet::new();
    let mut output_slab_ids = std::collections::BTreeSet::new();
    let mut argument_buffer_entries = std::collections::BTreeSet::new();
    let mut shared_metadata_parity = true;
    let mut digest = METAL_SCAN_RESOURCE_FNV_OFFSET;

    for (entry_index, entry) in entries.iter().copied().enumerate() {
        validate_metal_resident_scan_resource_entry(entry_index, entry, expected_lifetime)?;
        table_ids.insert(entry.table_id);
        verifier_ids.insert(entry.verifier_id);
        output_slab_ids.insert(entry.output_slab_id);
        if !argument_buffer_entries.insert(entry.argument_buffer_entry) {
            return Err(
                MetalResidentScanResourceError::DuplicateArgumentBufferEntry {
                    argument_buffer_entry: entry.argument_buffer_entry,
                },
            );
        }
        shared_metadata_parity &=
            entry.shared_scan_metadata_digest == entry.metal_resource_metadata_digest;
        digest = mix_metal_scan_resource_entry_digest(digest, entry);
    }

    let evidence = MetalResidentScanResourceTableEvidence {
        schema_version: METAL_RESIDENT_SCAN_RESOURCE_TABLE_SCHEMA_VERSION,
        entry_count,
        expected_lifetime,
        table_id_count: u32::try_from(table_ids.len()).unwrap_or(u32::MAX),
        verifier_id_count: u32::try_from(verifier_ids.len()).unwrap_or(u32::MAX),
        output_slab_id_count: u32::try_from(output_slab_ids.len()).unwrap_or(u32::MAX),
        argument_buffer_entry_count: u32::try_from(argument_buffer_entries.len())
            .unwrap_or(u32::MAX),
        shared_metadata_parity,
        resource_table_digest: digest,
    };
    if !evidence.is_complete() {
        return Err(MetalResidentScanResourceError::MetadataDigestMismatch {
            entry_index: 0,
            shared_scan_metadata_digest: entries[0].shared_scan_metadata_digest,
            metal_resource_metadata_digest: entries[0].metal_resource_metadata_digest,
        });
    }
    Ok(evidence)
}

fn validate_metal_resident_scan_resource_entry(
    entry_index: usize,
    entry: MetalResidentScanResourceEntry,
    expected_lifetime: MetalResidentScanResourceLifetime,
) -> Result<(), MetalResidentScanResourceError> {
    if entry.table_id == 0 {
        return Err(MetalResidentScanResourceError::ZeroId {
            entry_index,
            field: "table_id",
        });
    }
    if entry.verifier_id == 0 {
        return Err(MetalResidentScanResourceError::ZeroId {
            entry_index,
            field: "verifier_id",
        });
    }
    if entry.output_slab_id == 0 {
        return Err(MetalResidentScanResourceError::ZeroId {
            entry_index,
            field: "output_slab_id",
        });
    }
    if entry.shared_scan_metadata_digest == 0 {
        return Err(MetalResidentScanResourceError::ZeroSharedMetadataDigest { entry_index });
    }
    if entry.shared_scan_metadata_digest != entry.metal_resource_metadata_digest {
        return Err(MetalResidentScanResourceError::MetadataDigestMismatch {
            entry_index,
            shared_scan_metadata_digest: entry.shared_scan_metadata_digest,
            metal_resource_metadata_digest: entry.metal_resource_metadata_digest,
        });
    }
    if entry.lifetime != expected_lifetime {
        return Err(MetalResidentScanResourceError::LifetimeMismatch {
            entry_index,
            expected: expected_lifetime,
            actual: entry.lifetime,
        });
    }
    Ok(())
}

fn mix_metal_scan_resource_entry_digest(
    mut digest: u64,
    entry: MetalResidentScanResourceEntry,
) -> u64 {
    digest = mix_metal_scan_resource_digest(digest, entry.table_id);
    digest = mix_metal_scan_resource_digest(digest, entry.verifier_id);
    digest = mix_metal_scan_resource_digest(digest, entry.output_slab_id);
    digest = mix_metal_scan_resource_digest(digest, u64::from(entry.argument_buffer_entry));
    digest = mix_metal_scan_resource_digest(digest, entry.lifetime.tag());
    digest = mix_metal_scan_resource_digest(digest, entry.shared_scan_metadata_digest);
    mix_metal_scan_resource_digest(digest, entry.metal_resource_metadata_digest)
}

fn mix_metal_scan_resource_digest(mut digest: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(METAL_SCAN_RESOURCE_FNV_PRIME);
    }
    digest
}
