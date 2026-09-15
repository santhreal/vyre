//! Generic resident workspace region layout for megakernel adapters.
//!
//! Domain adapters own their region identifiers and capacity policy. Runtime
//! owns the checked contiguous-layout arithmetic because every resident
//! megakernel workspace has the same ABI shape: ordered u32-word regions with
//! fixed record widths and explicit capacities.

/// One contiguous u32-word region inside a resident megakernel workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentWorkspaceRegion<R> {
    /// Domain-owned region id encoded in the workspace manifest.
    pub id: R,
    /// Offset from workspace word zero.
    pub offset_words: u32,
    /// Total words reserved for the region.
    pub words: u32,
    /// Words in one logical record for this region.
    pub record_words: u32,
    /// Logical record capacity for this region.
    pub capacity_records: u32,
}

/// Declarative region specification for bulk resident-workspace layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentWorkspaceRegionSpec<R> {
    /// Region whose total word count is already known.
    Fixed {
        /// Domain-owned region id encoded in the workspace manifest.
        id: R,
        /// Total words reserved for the region.
        words: u32,
        /// Words in one logical record for this region.
        record_words: u32,
        /// Logical record capacity for this region.
        capacity_records: u32,
    },
    /// Region sized as `record_words * capacity_records`.
    Record {
        /// Domain-owned region id encoded in the workspace manifest.
        id: R,
        /// Words in one logical record for this region.
        record_words: u32,
        /// Logical record capacity for this region.
        capacity_records: u32,
    },
}

impl<R> ResidentWorkspaceRegionSpec<R> {
    /// Build a fixed-size region specification.
    #[must_use]
    pub const fn fixed(id: R, words: u32, record_words: u32, capacity_records: u32) -> Self {
        Self::Fixed {
            id,
            words,
            record_words,
            capacity_records,
        }
    }

    /// Build a record-backed region specification.
    #[must_use]
    pub const fn record(id: R, record_words: u32, capacity_records: u32) -> Self {
        Self::Record {
            id,
            record_words,
            capacity_records,
        }
    }
}

/// Error returned by bulk resident-workspace layout planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentWorkspaceLayoutError<R> {
    /// `record_words * capacity_records` overflowed for this record-backed region.
    RecordWordsOverflow {
        /// Region whose record arena overflowed.
        region: R,
    },
    /// Contiguous region offset arithmetic overflowed for this region.
    OffsetOverflow {
        /// Region whose starting offset could not fit the accumulated layout.
        region: R,
    },
}

impl<R: Copy> ResidentWorkspaceRegion<R> {
    /// Exclusive end offset for this region.
    #[must_use]
    pub const fn end_words(self) -> Option<u32> {
        self.offset_words.checked_add(self.words)
    }
}

/// Return `record_words * capacity_records` for a record-backed region.
#[must_use]
pub const fn workspace_record_words(record_words: u32, capacity_records: u32) -> Option<u32> {
    record_words.checked_mul(capacity_records)
}

/// Build the first region in a resident workspace.
#[must_use]
pub const fn first_workspace_region<R>(
    id: R,
    words: u32,
    record_words: u32,
    capacity_records: u32,
) -> ResidentWorkspaceRegion<R> {
    ResidentWorkspaceRegion {
        id,
        offset_words: 0,
        words,
        record_words,
        capacity_records,
    }
}

/// Build the next contiguous region after `previous`.
#[must_use]
pub fn next_workspace_region<R: Copy>(
    previous: ResidentWorkspaceRegion<R>,
    id: R,
    words: u32,
    record_words: u32,
    capacity_records: u32,
) -> Option<ResidentWorkspaceRegion<R>> {
    Some(ResidentWorkspaceRegion {
        id,
        offset_words: previous.end_words()?,
        words,
        record_words,
        capacity_records,
    })
}

/// Build the next contiguous record-backed region after `previous`.
#[must_use]
pub fn next_record_workspace_region<R: Copy>(
    previous: ResidentWorkspaceRegion<R>,
    id: R,
    record_words: u32,
    capacity_records: u32,
) -> Option<ResidentWorkspaceRegion<R>> {
    next_workspace_region(
        previous,
        id,
        workspace_record_words(record_words, capacity_records)?,
        record_words,
        capacity_records,
    )
}

/// Build a contiguous resident-workspace layout from declarative specs.
///
/// This is the generic seam domain adapters should use when they own many
/// regions. It centralizes checked record multiplication and checked offset
/// accumulation in `vyre-runtime`, while each adapter keeps its own region ids
/// and capacity policy.
pub fn build_workspace_regions<R: Copy>(
    specs: &[ResidentWorkspaceRegionSpec<R>],
) -> Result<Vec<ResidentWorkspaceRegion<R>>, ResidentWorkspaceLayoutError<R>> {
    let mut regions = Vec::with_capacity(specs.len());
    let mut next_offset_words = 0_u32;

    for spec in specs {
        let (id, words, record_words, capacity_records) = match *spec {
            ResidentWorkspaceRegionSpec::Fixed {
                id,
                words,
                record_words,
                capacity_records,
            } => (id, words, record_words, capacity_records),
            ResidentWorkspaceRegionSpec::Record {
                id,
                record_words,
                capacity_records,
            } => {
                let words = workspace_record_words(record_words, capacity_records)
                    .ok_or(ResidentWorkspaceLayoutError::RecordWordsOverflow { region: id })?;
                (id, words, record_words, capacity_records)
            }
        };
        let end_words = next_offset_words
            .checked_add(words)
            .ok_or(ResidentWorkspaceLayoutError::OffsetOverflow { region: id })?;
        regions.push(ResidentWorkspaceRegion {
            id,
            offset_words: next_offset_words,
            words,
            record_words,
            capacity_records,
        });
        next_offset_words = end_words;
    }

    Ok(regions)
}
