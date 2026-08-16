//! CUDA dispatch path for long-lived resident buffers.

#[allow(dead_code)]
const _DISPATCH_MARKERS: &str = "dispatch_resident ptx";

mod async_dispatch;
mod batch;
mod borrowed;
mod dense_index_validation;
mod descriptor_cursor;
mod host_uploads;
mod sequence_api;
mod sequence_fused;
mod sequence_slots;
mod sync;
mod timed;

pub(crate) use crate::backend::resident_dispatch_accounting::CudaResidentDispatch;
pub(crate) use descriptor_cursor::next_dispatch_binding;

use std::sync::Arc;

use smallvec::SmallVec;
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::Program;

use crate::backend::plan::CudaDispatchPlan;
use crate::backend::resident::CudaResidentBuffer;

/// One resident dispatch step after its PTX, module key, and binding plan have
/// been resolved.
pub(crate) struct PreparedStep<'a> {
    pub(crate) program: &'a Program,
    pub(crate) handles: SmallVec<[CudaResidentBuffer; 8]>,
    pub(crate) config: &'a DispatchConfig,
    pub(crate) ptx_src: Arc<str>,
    pub(crate) module_key: crate::backend::module_cache::ModuleCacheKey,
    pub(crate) prepared: CudaDispatchPlan,
}

// Inline: `backend::resident_dispatch` is `pub(crate)`, so the dense-index
// validators, `stage_resident_fill_payload` and `prepare_resident_sequence_fills`
// are unreachable from an integration test.
#[cfg(test)]
mod tests {
    use super::dense_index_validation::{
        validate_dense_resident_input_indices, validate_dense_resident_output_indices,
    };
    use super::host_uploads::stage_resident_fill_payload;
    use super::sequence_slots::prepare_resident_sequence_fills;

    // Inline: covers `borrowed`, which no integration test can name.
    #[cfg(test)]
    #[allow(clippy::module_inception)]
    mod tests {
        use super::super::async_dispatch::resident_output_clear_for_readback;
        use super::super::borrowed::order_resident_fallback_inputs_by_logical_index;
        use super::{
            prepare_resident_sequence_fills, stage_resident_fill_payload,
            validate_dense_resident_input_indices, validate_dense_resident_output_indices,
        };
        use crate::backend::output_range::CudaOutputReadback;
        use crate::backend::resident::CudaResidentBuffer;
        use vyre_driver::ResidentOwner;

        #[test]
        fn resident_fallback_fill_payload_preserves_last_good_bytes_when_reservation_fails() {
            let mut payload = vec![0xC3, 0xC3, 0x7E, 0x11];

            let result = stage_resident_fill_payload(&mut payload, 0x5A, usize::MAX);

            assert!(
                result.is_err(),
                "oversized CUDA resident fill payload must fail preflight instead of mutating staging"
            );
            assert_eq!(
                payload,
                vec![0xC3, 0xC3, 0x7E, 0x11],
                "failed CUDA resident fill staging must preserve the last diagnostic payload"
            );
        }
        #[test]
        fn resident_fallback_fill_payload_reuses_capacity_and_overwrites_values() {
            let mut payload = Vec::new();
            {
                let bytes = stage_resident_fill_payload(&mut payload, 0xA5, 16)
                    .expect("Fix: reusable resident fallback fill staging should reserve bytes");
                assert_eq!(bytes, &[0xA5; 16]);
            }
            let initial_capacity = payload.capacity();

            {
                let bytes = stage_resident_fill_payload(&mut payload, 0x5A, 8)
                    .expect("Fix: smaller resident fallback fill staging should reuse capacity");
                assert_eq!(bytes, &[0x5A; 8]);
            }
            assert_eq!(
                payload.capacity(),
                initial_capacity,
                "CUDA resident fallback fill staging must reuse capacity across fills instead of allocating one Vec per fill"
            );

            {
                let bytes = stage_resident_fill_payload(&mut payload, 0x11, 0)
                    .expect("Fix: zero-byte resident fallback fill staging should be valid");
                assert!(bytes.is_empty());
            }
            assert_eq!(
                payload.capacity(),
                initial_capacity,
                "zero-byte fallback fills must not release reusable staging capacity"
            );
        }

        #[test]
        fn resident_output_clear_uses_observable_readback_range() {
            let clear = resident_output_clear_for_readback(
                0x1000,
                CudaOutputReadback {
                    device_offset: 128,
                    byte_len: 4096,
                },
                "out",
            )
            .expect("Fix: ranged resident output clear planning must accept valid offsets.");

            assert_eq!(
                clear,
                Some((0x1080, 4096)),
                "Fix: resident dispatch must clear the declared output byte range, not the padded allocation."
            );

            let full = resident_output_clear_for_readback(
                0x2000,
                CudaOutputReadback {
                    device_offset: 0,
                    byte_len: 8192,
                },
                "full",
            )
            .expect("Fix: full resident output clear planning must preserve full-buffer clears.");
            assert_eq!(full, Some((0x2000, 8192)));
        }

        #[test]
        fn resident_output_clear_skips_zero_byte_ranges_and_rejects_pointer_overflow() {
            let skipped = resident_output_clear_for_readback(
                0x1000,
                CudaOutputReadback {
                    device_offset: 256,
                    byte_len: 0,
                },
                "empty",
            )
            .expect("Fix: zero-byte resident output ranges should not enqueue memsets.");
            assert_eq!(skipped, None);

            let error = resident_output_clear_for_readback(
                u64::MAX,
                CudaOutputReadback {
                    device_offset: 1,
                    byte_len: 4,
                },
                "overflow",
            )
            .expect_err("Fix: resident output clear planning must reject device-pointer overflow.");

            assert!(
                error.to_string().contains("overflowed"),
                "overflow error must explain the CUDA resident clear pointer failure: {error}"
            );
        }

        #[test]
        fn resident_output_index_validation_rejects_sparse_or_duplicate_sorted_indexes() {
            validate_dense_resident_output_indices([0, 1, 2], 3, "test output")
                .expect("Fix: dense resident output indexes must validate.");
            let duplicate =
                validate_dense_resident_output_indices([0, 0, 2], 3, "test output").unwrap_err();
            assert!(
                duplicate.to_string().contains("duplicate"),
                "Fix: duplicate resident output indexes must fail before readback ordering can alias an output slot: {duplicate}"
            );
            let sparse =
                validate_dense_resident_output_indices([0, 2, 3], 3, "test output").unwrap_err();
            assert!(
                sparse.to_string().contains("dense"),
                "Fix: sparse resident output indexes must fail before readback ordering can skip an output slot: {sparse}"
            );
            let truncated =
                validate_dense_resident_output_indices([0, 1], 3, "test output").unwrap_err();
            assert!(
                truncated.to_string().contains("expected 3"),
                "Fix: truncated resident output indexes must fail before readback ordering can drop an output slot: {truncated}"
            );
        }

        #[test]
        fn resident_input_index_validation_rejects_sparse_duplicate_or_truncated_indexes() {
            validate_dense_resident_input_indices([0, 1, 2], 3, "test input")
                .expect("Fix: dense resident input indexes must validate.");
            let duplicate =
                validate_dense_resident_input_indices([0, 0, 2], 3, "test input").unwrap_err();
            assert!(
                duplicate.to_string().contains("duplicate"),
                "Fix: duplicate resident input indexes must fail before borrowed fallback can alias a logical input slot: {duplicate}"
            );
            let sparse = validate_dense_resident_input_indices([0, 2, 3], 3, "test input").unwrap_err();
            assert!(
                sparse.to_string().contains("dense"),
                "Fix: sparse resident input indexes must fail before borrowed fallback can skip a logical input slot: {sparse}"
            );
            let truncated = validate_dense_resident_input_indices([0, 1], 3, "test input").unwrap_err();
            assert!(
                truncated.to_string().contains("expected 3"),
                "Fix: truncated resident input indexes must fail before borrowed fallback can drop a logical input slot: {truncated}"
            );
        }

        #[test]
        fn resident_borrowed_fallback_orders_downloaded_inputs_by_logical_slot() {
            let mut inputs = vec![(2, vec![0xCC]), (0, vec![0xAA]), (1, vec![0xBB])];

            order_resident_fallback_inputs_by_logical_index(&mut inputs, 3)
                .expect("Fix: reordered resident fallback inputs should sort by logical input slot.");

            assert_eq!(
                inputs,
                vec![
                    (0, vec![0xAA]),
                    (1, vec![0xBB]),
                    (2, vec![0xCC]),
                ],
                "Fix: CUDA resident borrowed fallback must pass dispatch_borrowed inputs in Program::buffers logical order, not descriptor binding order."
            );

            let mut duplicate = vec![(0, vec![1]), (0, vec![2])];
            assert!(
                order_resident_fallback_inputs_by_logical_index(&mut duplicate, 2).is_err(),
                "Fix: resident fallback input ordering must reject duplicate logical input slots before launch."
            );
        }

        #[test]
        fn resident_sequence_fills_coalesce_duplicates_and_skip_full_upload_overwrites() {
            let owner = ResidentOwner::new().expect("Fix: owner ids must be available");
            let first = CudaResidentBuffer {
                handle: owner.handle(1),
                byte_len: 16,
            };
            let second = CudaResidentBuffer {
                handle: owner.handle(2),
                byte_len: 16,
            };
            let upload = [0xFE_u8; 16];

            let effective = prepare_resident_sequence_fills(
                &[(first, 0x11), (second, 0x22), (first, 0x33)],
                &[(second, upload.as_slice())],
            )
            .expect("Fix: generated resident sequence fill coalescing must succeed.");

            assert_eq!(
                effective.as_slice(),
                &[(first, 0x33)],
                "Fix: resident sequence fills must keep the last duplicate fill and drop fills fully overwritten by same-sequence uploads."
            );
        }

        #[test]
        fn resident_sequence_fills_handle_dense_duplicate_streams_without_changing_order() {
            let owner = ResidentOwner::new().expect("Fix: owner ids must be available");
            let handles: Vec<_> = (0..256)
                .map(|id| CudaResidentBuffer {
                    handle: owner.handle(id),
                    byte_len: 1,
                })
                .collect();
            let mut fills = Vec::new();
            for round in 0..8_u8 {
                fills.extend(handles.iter().copied().map(|handle| (handle, round)));
            }

            let upload = [0xAA_u8];
            let uploads: Vec<_> = handles
                .iter()
                .copied()
                .filter(|handle| handle.handle.id() % 2 == 0)
                .map(|handle| (handle, upload.as_slice()))
                .collect();

            let effective = prepare_resident_sequence_fills(&fills, &uploads)
                .expect("Fix: dense CUDA resident fill coalescing must reserve bounded indices.");

            assert_eq!(
                effective.len(),
                128,
                "Fix: uploaded handles must suppress same-sequence fills even under dense duplicate traffic."
            );
            for (position, (handle, value)) in effective.iter().copied().enumerate() {
                assert_eq!(
                    handle.handle.id() % 2,
                    1,
                    "Fix: uploaded resident handle {} must not retain a redundant fill.",
                    handle.handle
                );
                assert_eq!(
                    handle.handle.id() as usize,
                    position * 2 + 1,
                    "Fix: first-seen fill order must be stable after duplicate coalescing."
                );
                assert_eq!(
                    value, 7,
                    "Fix: duplicate resident fills must keep the final value for each handle."
                );
            }
        }
    }
}
