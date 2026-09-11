//! CUDA-facing resident D2H readback interval fusion adapter.
//!
//! The interval coalescing policy is backend-neutral. This module preserves the
//! CUDA domain names used by resident IO/dispatch while delegating the actual
//! fusion algorithm to `vyre-driver`.

use vyre_driver::resident_transfer_fusion::{
    fuse_resident_transfer_intervals, FusedResidentTransfers, ResidentTransferInterval,
    ResidentTransferView,
};
use vyre_driver::{BackendError, ResidentHandle};

use super::resident::ResidentBufferView;

/// One validated device-to-host readback request.
pub(crate) type ResidentReadbackCopy = ResidentTransferInterval;

/// How an original requested output is sliced out of a fused transfer.
pub(crate) type ResidentReadbackView = ResidentTransferView;

/// Fused transfer plan plus original-output views.
pub(crate) type FusedResidentReadbacks = FusedResidentTransfers;

/// Fuse overlapping or adjacent readback intervals, scoped by resident handle.
pub(crate) fn fuse_resident_readback_copies(
    requested: &[ResidentReadbackCopy],
) -> Result<FusedResidentReadbacks, BackendError> {
    fuse_resident_transfer_intervals(requested)
}

/// Validate one requested readback against its resident view and produce the
/// copy the fusion plan consumes.
///
/// Four resident readback paths (compact, batched, ranged download, fused
/// sequence) each proved the same two facts about the same descriptor:
/// `byte_offset..byte_offset + byte_len` lies inside the resident allocation,
/// and `view.ptr + byte_offset` stays inside `CUdeviceptr` arithmetic. `role`
/// was the only difference between them and it only ever reached the error
/// text. A bounds check spelled four times is a bounds check that gets
/// corrected in one place and left wrong in three, so the check lives here and
/// the caller supplies the phrase that names its path.
///
/// An empty range yields a null source pointer without pointer arithmetic:
/// `view.ptr + byte_offset` is not required to be a valid address when no
/// bytes are copied, and a zero-length allocation has no valid address to
/// offset from.
pub(crate) fn resident_readback_copy(
    role: &str,
    handle: ResidentHandle,
    view: ResidentBufferView,
    byte_offset: usize,
    byte_len: usize,
) -> Result<ResidentReadbackCopy, BackendError> {
    vyre_driver::accounting::checked_usize_byte_range_end_lazy(
        byte_offset,
        byte_len,
        view.byte_len,
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {role} for handle {handle} overflows usize at offset {byte_offset} len {byte_len}."
            ),
        }
        },
        |end| {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {role} for handle {handle} requested bytes [{byte_offset}..{end}) but buffer has {} bytes.",
                view.byte_len
            ),
        }
        },
    )?;
    let src = if byte_len == 0 {
        0
    } else {
        vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
            view.ptr,
            byte_offset,
            || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {role} device offset {byte_offset} does not fit CUdeviceptr arithmetic for handle {handle}."
                ),
            }
            },
            || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {role} pointer arithmetic overflowed for handle {handle} at offset {byte_offset}."
                ),
            }
            },
        )?
    };
    Ok(ResidentReadbackCopy {
        handle_id: handle.id(),
        src,
        byte_len,
    })
}

pub(crate) fn validate_fused_resident_readbacks(
    fused: &FusedResidentReadbacks,
    requested_output_slots: usize,
    context: &'static str,
) -> Result<(), BackendError> {
    if fused.views.len() != requested_output_slots {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {context} fused readback view count {} does not match {} requested output slot(s). Keep resident readback fusion cardinality-preserving before materializing outputs.",
                fused.views.len(),
                requested_output_slots
            ),
        });
    }
    if fused.non_empty_copy_count != fused.copies.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {context} fused readback counted {} non-empty copy operation(s) but staged {} copy slot(s). Keep resident readback telemetry and staging in the same fusion plan.",
                fused.non_empty_copy_count,
                fused.copies.len()
            ),
        });
    }
    let mut staged_copy_bytes = 0u64;
    for copy in fused.copies.iter().copied() {
        let copy_bytes = u64::try_from(copy.byte_len).map_err(|_| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} fused readback staged copy for handle {} has {} byte(s), which does not fit telemetry accounting.",
                    copy.handle_id, copy.byte_len
                ),
            }
        })?;
        staged_copy_bytes = staged_copy_bytes
            .checked_add(copy_bytes)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} fused readback staged byte accounting overflowed while validating handle {}.",
                    copy.handle_id
                ),
            })?;
    }
    if fused.bytes != staged_copy_bytes {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {context} fused readback reports {} telemetry bytes but stages {} bytes. Keep resident readback telemetry equal to the fused copy plan.",
                fused.bytes, staged_copy_bytes
            ),
        });
    }
    for (view_index, view) in fused.views.iter().copied().enumerate() {
        if view.byte_len == 0 {
            continue;
        }
        let Some(copy) = fused.copies.get(view.copy_slot) else {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} fused readback view {view_index} references missing copy_slot {} for {} byte(s). Rebuild the resident readback fusion plan before materializing outputs.",
                    view.copy_slot,
                    view.byte_len
                ),
            });
        };
        let view_end =
            view.byte_offset
                .checked_add(view.byte_len)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA {context} fused readback view {view_index} overflows usize at offset {} len {}.",
                        view.byte_offset, view.byte_len
                    ),
                })?;
        if view_end > copy.byte_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} fused readback view {view_index} requested bytes [{}..{}) which exceeds the {} byte fused copy. Rebuild the resident readback fusion plan before materializing outputs.",
                    view.byte_offset, view_end, copy.byte_len
                ),
            });
        }
    }
    Ok(())
}

// Inline: covers `FusedResidentReadbacks`, `ResidentReadbackCopy`, `ResidentReadbackView`,
// `fuse_resident_readback_copies` and 1 more item this module keeps private, which no integration
// test can name.
#[cfg(test)]
mod tests {
    use smallvec::smallvec;
    use vyre_driver::resident_transfer_fixtures::{
        assert_fused_transfers_preserve_requests, generated_transfer_requests,
    };

    use super::{
        fuse_resident_readback_copies, validate_fused_resident_readbacks, FusedResidentReadbacks,
        ResidentReadbackCopy, ResidentReadbackView,
    };

    /// Grades this crate's adapter against the neutral fusion model, including
    /// the validator this crate adds on top of it. A copy of the corpus or of
    /// the materialization property here would let the adapter drift from the
    /// algorithm it delegates to and still pass.
    #[test]
    fn generated_fusion_preserves_every_requested_output_and_accounts_union_bytes() {
        for seed in 0..8192_u64 {
            let requested = generated_transfer_requests(seed);
            let fused = fuse_resident_readback_copies(&requested)
                .expect("Fix: generated resident readback requests must fuse without overflow");
            validate_fused_resident_readbacks(&fused, requested.len(), "generated readback")
                .expect(
                    "Fix: generated resident readback fusion must produce a materializable plan",
                );

            assert_fused_transfers_preserve_requests(seed, &requested, &fused);
        }
    }

    #[test]
    fn fused_readback_validation_rejects_non_materializable_views() {
        let bad_cardinality = FusedResidentReadbacks {
            copies: smallvec![ResidentReadbackCopy {
                handle_id: 1,
                src: 16,
                byte_len: 4,
            }],
            views: smallvec![],
            non_empty_copy_count: 1,
            bytes: 4,
        };
        let cardinality =
            validate_fused_resident_readbacks(&bad_cardinality, 1, "test").unwrap_err();
        assert!(
            cardinality.to_string().contains("view count"),
            "Fix: CUDA fused readback validation must reject plans that would silently skip output slots: {cardinality}"
        );

        let bad_slot = FusedResidentReadbacks {
            copies: smallvec![ResidentReadbackCopy {
                handle_id: 1,
                src: 16,
                byte_len: 4,
            }],
            views: smallvec![ResidentReadbackView {
                copy_slot: 1,
                byte_offset: 0,
                byte_len: 1,
            }],
            non_empty_copy_count: 1,
            bytes: 4,
        };
        let slot = validate_fused_resident_readbacks(&bad_slot, 1, "test").unwrap_err();
        assert!(
            slot.to_string().contains("copy_slot"),
            "Fix: CUDA fused readback validation must reject views pointing outside staged copy slots: {slot}"
        );

        let bad_range = FusedResidentReadbacks {
            copies: smallvec![ResidentReadbackCopy {
                handle_id: 1,
                src: 16,
                byte_len: 4,
            }],
            views: smallvec![ResidentReadbackView {
                copy_slot: 0,
                byte_offset: 3,
                byte_len: 2,
            }],
            non_empty_copy_count: 1,
            bytes: 4,
        };
        let range = validate_fused_resident_readbacks(&bad_range, 1, "test").unwrap_err();
        assert!(
            range.to_string().contains("exceeds"),
            "Fix: CUDA fused readback validation must reject output views that overrun the fused copy: {range}"
        );

        let bad_bytes = FusedResidentReadbacks {
            copies: smallvec![ResidentReadbackCopy {
                handle_id: 1,
                src: 16,
                byte_len: 4,
            }],
            views: smallvec![ResidentReadbackView {
                copy_slot: 0,
                byte_offset: 0,
                byte_len: 4,
            }],
            non_empty_copy_count: 1,
            bytes: 3,
        };
        let bytes = validate_fused_resident_readbacks(&bad_bytes, 1, "test").unwrap_err();
        assert!(
            bytes.to_string().contains("bytes"),
            "Fix: CUDA fused readback validation must reject telemetry byte counts that drift from staged copies: {bytes}"
        );
    }

    #[test]
    fn monotonic_resident_readbacks_fuse_without_reordering_views() {
        let requested = [
            ResidentReadbackCopy {
                handle_id: 1,
                src: 100,
                byte_len: 8,
            },
            ResidentReadbackCopy {
                handle_id: 1,
                src: 104,
                byte_len: 4,
            },
            ResidentReadbackCopy {
                handle_id: 2,
                src: 16,
                byte_len: 2,
            },
        ];

        let fused = fuse_resident_readback_copies(&requested)
            .expect("Fix: monotonic resident readback fusion must not require sorting.");

        assert_eq!(
            fused.copies.len(),
            2,
            "Fix: monotonic same-handle intervals must still fuse on the sorted fast path."
        );
        assert_eq!(fused.copies[0].handle_id, 1);
        assert_eq!(fused.copies[0].src, 100);
        assert_eq!(fused.copies[0].byte_len, 8);
        assert_eq!(fused.copies[1].handle_id, 2);
        assert_eq!(
            fused.views[0].copy_slot, 0,
            "Fix: first monotonic request must map to the first fused copy."
        );
        assert_eq!(
            fused.views[1].byte_offset, 4,
            "Fix: overlapping monotonic request must retain its offset inside the fused copy."
        );
        assert_eq!(
            fused.views[2].copy_slot, 1,
            "Fix: monotonic distinct-handle request must map to its own fused copy."
        );
    }

    #[test]
    fn adjacent_raw_pointers_from_distinct_handles_do_not_fuse() {
        let requested = [
            ResidentReadbackCopy {
                handle_id: 1,
                src: 100,
                byte_len: 8,
            },
            ResidentReadbackCopy {
                handle_id: 2,
                src: 108,
                byte_len: 8,
            },
        ];

        let fused = fuse_resident_readback_copies(&requested)
            .expect("Fix: distinct-handle adjacent ranges must fuse-check without error");

        assert_eq!(
            fused.copies.len(),
            2,
            "Fix: adjacent raw pointers from distinct resident allocations must not coalesce."
        );
        assert_eq!(fused.bytes, 16);
    }
}
