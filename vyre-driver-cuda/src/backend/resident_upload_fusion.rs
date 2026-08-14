//! CUDA-facing resident H2D upload interval fusion adapter.
//!
//! Ordered overwrite fusion is backend-neutral. This module preserves the CUDA
//! domain names used by resident IO/dispatch while delegating the actual fusion
//! algorithm to `vyre-driver`.

use smallvec::SmallVec;
use vyre_driver::resident_transfer_fusion::{
    fuse_resident_upload_copies as driver_fuse_resident_upload_copies,
    push_resident_upload_copy as driver_push_resident_upload_copy,
    ResidentUploadBytes as DriverResidentUploadBytes,
    ResidentUploadCopy as DriverResidentUploadCopy,
};
use vyre_driver::BackendError;

/// Host bytes for one resident upload interval.
pub(crate) type ResidentUploadBytes<'a> = DriverResidentUploadBytes<'a>;

/// One validated host-to-device upload request.
pub(crate) type ResidentUploadCopy<'a> = DriverResidentUploadCopy<'a>;

/// Push one non-empty upload copy and account its requested bytes.
pub(crate) fn push_resident_upload_copy<'a>(
    copies: &mut SmallVec<[ResidentUploadCopy<'a>; 8]>,
    uploaded_bytes: &mut u64,
    handle_id: u64,
    dst_ptr: u64,
    bytes: &'a [u8],
    label: &str,
) -> Result<(), BackendError> {
    driver_push_resident_upload_copy(copies, uploaded_bytes, handle_id, dst_ptr, bytes, label)
}

/// Fuse same-handle adjacent or overlapping upload intervals.
pub(crate) fn fuse_resident_upload_copies<'a>(
    copies: SmallVec<[ResidentUploadCopy<'a>; 8]>,
) -> Result<(SmallVec<[ResidentUploadCopy<'a>; 8]>, u64), BackendError> {
    driver_fuse_resident_upload_copies(copies)
}

#[cfg(test)]
mod tests {
    use smallvec::SmallVec;
    use vyre_driver::resident_transfer_fixtures::assert_upload_fusion_preserves_ordered_writes;

    use super::{
        fuse_resident_upload_copies, push_resident_upload_copy, ResidentUploadBytes,
        ResidentUploadCopy,
    };

    #[test]
    fn empty_resident_upload_copy_does_not_schedule_dma() {
        let mut copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::new();
        let mut uploaded_bytes = 0_u64;

        push_resident_upload_copy(&mut copies, &mut uploaded_bytes, 7, 0xCAFE, &[], "unit")
            .expect("Fix: empty resident upload staging must not fail.");

        assert!(copies.is_empty());
        assert_eq!(uploaded_bytes, 0);
    }

    #[test]
    fn resident_upload_copy_accounts_non_empty_bytes_once() {
        let bytes = [1_u8, 2, 3, 4, 5];
        let mut copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::new();
        let mut uploaded_bytes = 0_u64;

        push_resident_upload_copy(&mut copies, &mut uploaded_bytes, 9, 0xBEEF, &bytes, "unit")
            .expect("Fix: non-empty resident upload staging must account bytes.");

        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].handle_id, 9);
        assert_eq!(copies[0].dst_ptr, 0xBEEF);
        assert_eq!(copies[0].bytes.as_slice(), bytes.as_slice());
        assert_eq!(uploaded_bytes, bytes.len() as u64);
    }

    #[test]
    fn resident_upload_copy_accounting_failure_is_transactional() {
        let bytes = [42_u8];
        let mut copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::new();
        let mut uploaded_bytes = u64::MAX;

        let error =
            push_resident_upload_copy(&mut copies, &mut uploaded_bytes, 9, 0xBEEF, &bytes, "unit")
                .expect_err("Fix: resident upload byte-accounting overflow must reject the copy.");

        assert!(
            error.to_string().contains("byte accounting overflowed"),
            "overflow diagnostic must identify the accounting bug: {error}"
        );
        assert!(
            copies.is_empty(),
            "Fix: failed resident upload accounting must not leave an unaccounted DMA copy queued."
        );
        assert_eq!(
            uploaded_bytes,
            u64::MAX,
            "Fix: failed resident upload accounting must not partially mutate byte counters."
        );
    }

    /// Grades this crate's adapter against the neutral fusion model. A copy of
    /// the corpus or of the ordered-write property here would let the adapter
    /// drift from the algorithm it delegates to and still pass.
    #[test]
    fn generated_resident_upload_fusion_preserves_ordered_write_semantics() {
        assert_upload_fusion_preserves_ordered_writes(0..4096, fuse_resident_upload_copies);
    }

    #[test]
    fn adjacent_raw_destinations_from_distinct_handles_do_not_fuse_uploads() {
        let first = [1u8, 2, 3, 4];
        let second = [5u8, 6, 7, 8];
        let copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::from_vec(vec![
            ResidentUploadCopy {
                handle_id: 1,
                dst_ptr: 100,
                bytes: ResidentUploadBytes::Borrowed(first.as_slice()),
            },
            ResidentUploadCopy {
                handle_id: 2,
                dst_ptr: 104,
                bytes: ResidentUploadBytes::Borrowed(second.as_slice()),
            },
        ]);

        let (fused, fused_bytes) = fuse_resident_upload_copies(copies)
            .expect("Fix: distinct-handle adjacent uploads must fuse-check without error");

        assert_eq!(
            fused.len(),
            2,
            "Fix: adjacent raw destinations from distinct resident allocations must not coalesce."
        );
        assert_eq!(fused_bytes, 8);
    }

    #[test]
    fn backward_overlapping_uploads_fuse_and_preserve_later_prefix_write() {
        let first = [4u8, 5, 6, 7];
        let second = [1u8, 2, 9, 8];
        let copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::from_vec(vec![
            ResidentUploadCopy {
                handle_id: 7,
                dst_ptr: 104,
                bytes: ResidentUploadBytes::Borrowed(first.as_slice()),
            },
            ResidentUploadCopy {
                handle_id: 7,
                dst_ptr: 102,
                bytes: ResidentUploadBytes::Borrowed(second.as_slice()),
            },
        ]);

        let (fused, fused_bytes) = fuse_resident_upload_copies(copies)
            .expect("Fix: backward-overlap resident uploads must fuse without error");

        assert_eq!(
            fused.len(),
            1,
            "Fix: backward-overlapping same-handle uploads must coalesce into one H2D copy."
        );
        assert_eq!(fused[0].dst_ptr, 102);
        assert_eq!(fused[0].bytes.as_slice(), &[1, 2, 9, 8, 6, 7]);
        assert_eq!(fused_bytes, 6);
    }

    #[test]
    fn later_full_overwrite_replaces_prior_upload_without_materializing_old_bytes() {
        let first = [1u8, 2, 3, 4];
        let second = [9u8, 8, 7, 6];
        let copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::from_vec(vec![
            ResidentUploadCopy {
                handle_id: 7,
                dst_ptr: 100,
                bytes: ResidentUploadBytes::Borrowed(first.as_slice()),
            },
            ResidentUploadCopy {
                handle_id: 7,
                dst_ptr: 100,
                bytes: ResidentUploadBytes::Borrowed(second.as_slice()),
            },
        ]);

        let (fused, fused_bytes) = fuse_resident_upload_copies(copies)
            .expect("Fix: full-overwrite resident uploads must fuse without error");

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].dst_ptr, 100);
        assert_eq!(fused[0].bytes.as_slice(), second.as_slice());
        assert!(
            matches!(fused[0].bytes, ResidentUploadBytes::Borrowed(_)),
            "Fix: later full overwrite should keep the newer borrowed payload instead of allocating fused owned bytes."
        );
        assert_eq!(fused_bytes, second.len() as u64);
    }

    #[test]
    fn later_wider_overwrite_replaces_prior_upload_without_prefix_merge_allocation() {
        let first = [4u8, 5, 6, 7];
        let second = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::from_vec(vec![
            ResidentUploadCopy {
                handle_id: 9,
                dst_ptr: 104,
                bytes: ResidentUploadBytes::Borrowed(first.as_slice()),
            },
            ResidentUploadCopy {
                handle_id: 9,
                dst_ptr: 100,
                bytes: ResidentUploadBytes::Borrowed(second.as_slice()),
            },
        ]);

        let (fused, fused_bytes) = fuse_resident_upload_copies(copies)
            .expect("Fix: wider full-overwrite resident uploads must fuse without error");

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].dst_ptr, 100);
        assert_eq!(fused[0].bytes.as_slice(), second.as_slice());
        assert!(
            matches!(fused[0].bytes, ResidentUploadBytes::Borrowed(_)),
            "Fix: wider full overwrite should replace the old interval instead of allocating a merged prefix buffer."
        );
        assert_eq!(fused_bytes, second.len() as u64);
    }
}
