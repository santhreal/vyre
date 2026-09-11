//! Canonical test model for resident transfer interval fusion, shared by the
//! neutral fusion tests and every backend adapter gate.
//!
//! [`crate::resident_transfer_fusion`] owns the coalescing policy, so it also
//! owns the model that decides whether a fusion was correct: the request
//! generators, the byte model that says what a resident address holds, and the
//! properties a fused plan has to satisfy. A backend adapter that restated any
//! of those could pass its own gate while grading a different algorithm, which
//! is the one failure a delegation gate exists to catch.
//!
//! The generators are two distinct corpora on purpose. The ordered corpus walks
//! deterministic strides and reverses every third seed, so it drives the
//! non-monotonic request orders that force the fuser to refuse a merge it cannot
//! prove. The pseudo-random corpus mixes handles, offsets and zero-length
//! requests instead. Both are kept because they reach different rejections.
//!
//! Enabled by the `test-fixtures` feature: it is scaffolding, not product code,
//! and a published build should not carry it.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::resident_transfer_fusion::{
    FusedResidentTransfers, ResidentTransferInterval, ResidentUploadBytes, ResidentUploadCopy,
};
use crate::BackendError;

/// Advance an xorshift64 state and return the new value.
///
/// The corpora need a generator that is reproducible from a seed alone, with no
/// dependency on a crate version's sampling internals, so a failing seed stays
/// reproducible after a dependency bump.
pub fn next_pseudo_random_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// The byte a resident address is modeled to hold.
///
/// Mixing the handle into the value is what lets a materialization check catch a
/// fused copy that reads the right offset out of the wrong resident allocation.
#[must_use]
pub fn synthetic_resident_byte(handle_id: u64, src: u64) -> u8 {
    handle_id
        .wrapping_mul(131)
        .wrapping_add(src.wrapping_mul(17))
        .wrapping_add(29) as u8
}

/// Deterministic-stride readback requests, reversed on every third seed.
#[must_use]
pub fn ordered_transfer_requests(seed: u64) -> Vec<ResidentTransferInterval> {
    let count = (seed as usize % 17) + 1;
    let mut requests = Vec::with_capacity(count);
    for i in 0..count {
        let handle_id = ((seed >> (i % 11)) + i as u64) % 5;
        let src = ((seed.wrapping_mul(31) + (i as u64 * 13)) % 64) * 4;
        let byte_len = ((seed as usize + i * 7) % 9) * 4;
        requests.push(ResidentTransferInterval {
            handle_id,
            src,
            byte_len,
        });
    }
    if seed % 3 == 0 {
        requests.reverse();
    }
    requests
}

/// Pseudo-random readback requests across four handles, including zero-length.
#[must_use]
pub fn generated_transfer_requests(seed: u64) -> Vec<ResidentTransferInterval> {
    let mut state = seed ^ 0xC0DA_CAFE_51DE_D2D2;
    let count = 1 + (next_pseudo_random_u64(&mut state) as usize % 16);
    let mut requests = Vec::with_capacity(count);
    for _ in 0..count {
        requests.push(ResidentTransferInterval {
            handle_id: next_pseudo_random_u64(&mut state) % 4,
            src: next_pseudo_random_u64(&mut state) % 64,
            byte_len: next_pseudo_random_u64(&mut state) as usize % 17,
        });
    }
    requests
}

/// Bytes covered by the handle-scoped union of the requested intervals.
///
/// Counted per distinct address rather than summed per request, so a fuser that
/// double-charges an overlap fails here instead of reporting a plausible total.
#[must_use]
pub fn expected_transfer_union_bytes(requests: &[ResidentTransferInterval]) -> u64 {
    let mut covered = HashSet::<(u64, u64)>::new();
    for request in requests {
        for offset in 0..request.byte_len as u64 {
            covered.insert((request.handle_id, request.src + offset));
        }
    }
    covered.len() as u64
}

/// Bytes an original request would have read directly.
#[must_use]
pub fn materialize_transfer_request(request: ResidentTransferInterval) -> Vec<u8> {
    (0..request.byte_len)
        .map(|offset| synthetic_resident_byte(request.handle_id, request.src + offset as u64))
        .collect()
}

/// Bytes a fused view slices back out of its fused copy.
#[must_use]
pub fn materialize_transfer_view(
    copies: &[ResidentTransferInterval],
    copy_slot: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Vec<u8> {
    if byte_len == 0 {
        return Vec::new();
    }
    let copy = copies[copy_slot];
    (0..byte_len)
        .map(|offset| {
            synthetic_resident_byte(copy.handle_id, copy.src + (byte_offset + offset) as u64)
        })
        .collect()
}

/// Assert a fused readback plan reproduces every requested output.
///
/// Checks request cardinality, non-empty copy accounting, union byte totals,
/// exhaustive merging of adjacent same-handle copies, per-view handle identity,
/// and byte-for-byte materialization of every view including the empty ones.
///
/// # Panics
///
/// Panics with a seed-tagged diagnostic when any of those fails.
pub fn assert_fused_transfers_preserve_requests(
    seed: u64,
    requested: &[ResidentTransferInterval],
    fused: &FusedResidentTransfers,
) {
    assert_eq!(
        fused.views.len(),
        requested.len(),
        "Fix: fused views must preserve request cardinality for seed {seed}."
    );
    assert_eq!(
        fused.non_empty_copy_count,
        fused.copies.len(),
        "Fix: fused copy count must match non-empty copy slots for seed {seed}."
    );
    assert_eq!(
        fused.bytes,
        expected_transfer_union_bytes(requested),
        "Fix: fused byte accounting must equal the handle-scoped interval union for seed {seed}."
    );

    for pair in fused.copies.windows(2) {
        let left = pair[0];
        let right = pair[1];
        let left_end = left.src + left.byte_len as u64;
        assert!(
            left.handle_id != right.handle_id || right.src > left_end,
            "Fix: fused copies must not leave mergeable same-handle intervals for seed {seed}."
        );
    }

    for (index, request) in requested.iter().enumerate() {
        let view = fused.views[index];
        assert_eq!(
            view.byte_len, request.byte_len,
            "Fix: fused view length must preserve request {index} for seed {seed}."
        );
        if request.byte_len == 0 {
            assert_eq!(
                materialize_transfer_view(
                    &fused.copies,
                    view.copy_slot,
                    view.byte_offset,
                    view.byte_len
                ),
                Vec::<u8>::new(),
                "Fix: zero-byte request {index} must materialize empty output for seed {seed}."
            );
            continue;
        }
        assert!(
            view.copy_slot < fused.copies.len(),
            "Fix: non-empty request {index} must map to a real fused copy for seed {seed}."
        );
        assert_eq!(
            fused.copies[view.copy_slot].handle_id, request.handle_id,
            "Fix: request {index} must not read bytes from a different resident handle for seed {seed}."
        );
        assert_eq!(
            materialize_transfer_view(
                &fused.copies,
                view.copy_slot,
                view.byte_offset,
                view.byte_len
            ),
            materialize_transfer_request(*request),
            "Fix: fused view must reproduce request {index} byte-for-byte for seed {seed}."
        );
    }
}

/// One generated host-to-device upload request.
pub struct UploadRequest {
    /// Resident allocation the bytes are written into.
    pub handle_id: u64,
    /// Device address of the first byte.
    pub dst_ptr: u64,
    /// Host payload, never empty.
    pub bytes: Vec<u8>,
}

/// Pseudo-random upload requests across four handles with overlapping writes.
#[must_use]
pub fn generated_upload_requests(seed: u64) -> Vec<UploadRequest> {
    let mut state = seed ^ 0x5151_C0DA_9E37_1234;
    let count = 1 + (next_pseudo_random_u64(&mut state) as usize % 16);
    let mut requests = Vec::with_capacity(count);
    for _ in 0..count {
        let handle_id = next_pseudo_random_u64(&mut state) % 4;
        let dst_ptr = next_pseudo_random_u64(&mut state) % 64;
        let len = 1 + (next_pseudo_random_u64(&mut state) as usize % 16);
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            bytes.push(next_pseudo_random_u64(&mut state) as u8);
        }
        requests.push(UploadRequest {
            handle_id,
            dst_ptr,
            bytes,
        });
    }
    requests
}

/// Resident memory after applying the requests in caller order.
#[must_use]
pub fn materialize_upload_requests(requests: &[UploadRequest]) -> HashMap<(u64, u64), u8> {
    let mut memory = HashMap::new();
    for request in requests {
        for (offset, &byte) in request.bytes.iter().enumerate() {
            memory.insert((request.handle_id, request.dst_ptr + offset as u64), byte);
        }
    }
    memory
}

/// Resident memory after applying the fused copies in fused order.
#[must_use]
pub fn materialize_fused_uploads(copies: &[ResidentUploadCopy<'_>]) -> HashMap<(u64, u64), u8> {
    let mut memory = HashMap::new();
    for copy in copies {
        for (offset, &byte) in copy.bytes.as_slice().iter().enumerate() {
            memory.insert((copy.handle_id, copy.dst_ptr + offset as u64), byte);
        }
    }
    memory
}

/// Assert an upload fusion entry point preserves ordered overwrite semantics.
///
/// For every seed in `seeds`, generates a corpus, drives `fuse`, and requires
/// that the fused writes leave resident memory byte-identical to the unfused
/// sequence, that fused accounting never exceeds requested bytes, and that no
/// mergeable monotonic same-handle interval survives.
///
/// `fuse` is a parameter rather than a fixed call so a backend adapter is graded
/// through its own entry point against the neutral model, which is what makes a
/// silently forked adapter fail.
///
/// # Panics
///
/// Panics with a seed-tagged diagnostic when fusion fails or violates any of
/// those properties.
pub fn assert_upload_fusion_preserves_ordered_writes<F>(seeds: std::ops::Range<u64>, fuse: F)
where
    F: for<'a> Fn(
        SmallVec<[ResidentUploadCopy<'a>; 8]>,
    ) -> Result<(SmallVec<[ResidentUploadCopy<'a>; 8]>, u64), BackendError>,
{
    for seed in seeds {
        let requests = generated_upload_requests(seed);
        let mut copies = SmallVec::<[ResidentUploadCopy<'_>; 8]>::new();
        for request in &requests {
            copies.push(ResidentUploadCopy {
                handle_id: request.handle_id,
                dst_ptr: request.dst_ptr,
                bytes: ResidentUploadBytes::Borrowed(request.bytes.as_slice()),
            });
        }

        let expected = materialize_upload_requests(&requests);
        let requested_bytes = requests
            .iter()
            .map(|request| request.bytes.len() as u64)
            .sum::<u64>();
        let (fused, fused_bytes) = fuse(copies)
            .expect("Fix: generated resident upload fusion must not overflow accounting");

        assert_eq!(
            materialize_fused_uploads(&fused),
            expected,
            "Fix: fused resident uploads must preserve ordered write semantics for seed {seed}."
        );
        assert!(
            fused_bytes <= requested_bytes,
            "Fix: fused resident upload byte accounting must not exceed requested bytes for seed {seed}."
        );
        for pair in fused.as_slice().windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let left_end = left.dst_ptr + left.bytes.len() as u64;
            assert!(
                left.handle_id != right.handle_id
                    || right.dst_ptr < left.dst_ptr
                    || right.dst_ptr > left_end,
                "Fix: resident upload fusion left a mergeable monotonic same-handle interval for seed {seed}."
            );
        }
    }
}
