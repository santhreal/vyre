//! Dispatch output payloads shared by every backend.

use std::collections::TryReserveError;

/// Output of one dispatch: a vector per output buffer slot, each
/// vector holding the raw bytes read back from the GPU. Consumers
/// decode the bytes per the Program's output buffer declarations.
/// The outer vec is indexed in the same order as the Program's
/// `is_output: true` buffers.
pub type OutputBuffers = Vec<Vec<u8>>;

/// Read-back bytes of a batched submission: every dispatch's output row stored
/// end to end in one allocation, addressed by dispatch index.
///
/// A batch used to be returned as `Vec<Vec<Vec<u8>>>`, which allocated an inner
/// vector per dispatch to hold the single row that dispatch produces, plus a
/// vector for the row bytes themselves. Rows here share one buffer and are
/// handed out borrowed, so an `n`-dispatch batch costs two allocations instead
/// of `2 * n + 1`, and neither the caller nor the readback path can observe a
/// per-row capacity that outlives the row.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchOutputs {
    bytes: Vec<u8>,
    /// End offset in `bytes` of each row, in submission order. A row's start is
    /// the previous entry, so the first row starts at zero and no separate
    /// start array can drift out of step with this one.
    row_ends: Vec<usize>,
}

impl BatchOutputs {
    /// Reserve room for `rows` more rows holding `bytes` more bytes in total.
    ///
    /// # Errors
    ///
    /// Returns the allocator's error when either reservation cannot be
    /// satisfied, so a batch too large for the host fails before any device
    /// buffer is mapped.
    pub fn try_reserve(&mut self, rows: usize, bytes: usize) -> Result<(), TryReserveError> {
        crate::allocation::try_reserve_vec_to_capacity(&mut self.row_ends, rows)?;
        crate::allocation::try_reserve_vec_to_capacity(&mut self.bytes, bytes)
    }

    /// Append one dispatch's output row.
    pub fn push_row(&mut self, row: &[u8]) {
        self.bytes.extend_from_slice(row);
        self.row_ends.push(self.bytes.len());
    }

    /// Number of rows, one per dispatch in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.row_ends.len()
    }

    /// Whether the batch produced no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.row_ends.is_empty()
    }

    /// Borrow one dispatch's output row, or `None` past the end of the batch.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[u8]> {
        let end = *self.row_ends.get(index)?;
        let start = index.checked_sub(1).map_or(0, |prev| self.row_ends[prev]);
        Some(&self.bytes[start..end])
    }

    /// Borrow every row in submission order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
        (0..self.len()).map(|index| {
            self.row(index)
                .unwrap_or_else(|| unreachable!("row index below len is always present"))
        })
    }

    /// Total bytes read back across every row.
    ///
    /// Exact rather than summed, because the rows share one buffer whose length
    /// is the total by construction and so cannot overflow a `usize`.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// Slot-reuse accounting from output-buffer replacement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OutputSlotStats {
    /// Total output slots written after replacement.
    pub total_slots: usize,
    /// Existing output slots whose allocation was reused.
    pub reused_slots: usize,
    /// Existing output slots replaced by moving an oversized incoming allocation.
    pub moved_slots: usize,
    /// New output slots appended beyond the previous output vector length.
    pub appended_slots: usize,
}

/// Byte-pressure accounting from output-buffer replacement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OutputSlotByteStats {
    /// Bytes presented by incoming output buffers before replacement.
    pub incoming_bytes: usize,
    /// Bytes copied into retained caller-owned slots.
    pub copied_bytes: usize,
    /// Bytes moved into place by swapping oversized incoming allocations.
    pub moved_bytes: usize,
    /// Bytes appended beyond the previous output vector length.
    pub appended_bytes: usize,
    /// Total retained capacity of output slots after replacement.
    pub retained_capacity_bytes: usize,
}

/// Full output replacement accounting: slot decisions plus byte pressure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OutputReplacementStats {
    /// Slot-level reuse/move/append accounting.
    pub slots: OutputSlotStats,
    /// Byte-level copy/move/append/capacity accounting.
    pub bytes: OutputSlotByteStats,
}

/// Replace `outputs` with `incoming` while preserving already-allocated output
/// slots whenever their positions still exist.
pub fn replace_output_buffers_preserving_slots(
    incoming: OutputBuffers,
    outputs: &mut OutputBuffers,
) {
    let _ = replace_output_buffers_preserving_slots_with_stats(incoming, outputs);
}

/// Replace output buffers and return allocation-reuse accounting.
pub fn replace_output_buffers_preserving_slots_with_stats(
    incoming: OutputBuffers,
    outputs: &mut OutputBuffers,
) -> OutputSlotStats {
    replace_output_buffers_preserving_slots_with_memory_stats(incoming, outputs).slots
}

/// Replace output buffers and return allocation-reuse plus byte-pressure
/// accounting.
pub fn replace_output_buffers_preserving_slots_with_memory_stats(
    incoming: OutputBuffers,
    outputs: &mut OutputBuffers,
) -> OutputReplacementStats {
    let total_slots = incoming.len();
    let previous_slots = outputs.len();
    reserve_output_slots_for_replacement(outputs, total_slots);
    let mut incoming = incoming.into_iter();
    let mut retained_slots = 0usize;
    let mut reused_slots = 0usize;
    let mut moved_slots = 0usize;
    let mut incoming_bytes = 0usize;
    let mut copied_bytes = 0usize;
    let mut moved_bytes = 0usize;
    let mut appended_bytes = 0usize;
    for (slot, mut bytes) in outputs.iter_mut().zip(incoming.by_ref()) {
        incoming_bytes = add_bytes(incoming_bytes, bytes.len(), "incoming output bytes");
        if bytes.len() <= slot.capacity() {
            slot.clear();
            copied_bytes = add_bytes(copied_bytes, bytes.len(), "copied output bytes");
            slot.extend_from_slice(&bytes);
            reused_slots += 1;
        } else {
            moved_bytes = add_bytes(moved_bytes, bytes.len(), "moved output bytes");
            std::mem::swap(slot, &mut bytes);
            moved_slots += 1;
        }
        retained_slots += 1;
    }
    outputs.truncate(retained_slots);
    for bytes in incoming {
        incoming_bytes = add_bytes(incoming_bytes, bytes.len(), "incoming output bytes");
        appended_bytes = add_bytes(appended_bytes, bytes.len(), "appended output bytes");
        outputs.push(bytes);
    }
    let retained_capacity_bytes = outputs.iter().fold(0usize, |sum, output| {
        add_bytes(sum, output.capacity(), "retained output capacity bytes")
    });
    OutputReplacementStats {
        slots: OutputSlotStats {
            total_slots,
            reused_slots,
            moved_slots,
            appended_slots: total_slots.saturating_sub(previous_slots),
        },
        bytes: OutputSlotByteStats {
            incoming_bytes,
            copied_bytes,
            moved_bytes,
            appended_bytes,
            retained_capacity_bytes,
        },
    }
}

fn reserve_output_slots_for_replacement(outputs: &mut OutputBuffers, total_slots: usize) {
    outputs.reserve(total_slots.saturating_sub(outputs.len()));
}

fn add_bytes(current: usize, incoming: usize, _label: &str) -> usize {
    current.saturating_add(incoming)
}

/// Output plus timing captured by a backend-owned dispatch path.
///
/// `wall_ns` is always populated by the shared default implementation.
/// `device_ns` is populated only when a backend can measure elapsed device
/// stream time without crossing the driver boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimedDispatchResult {
    /// Output buffers in the same order as [`crate::backend::VyreBackend::dispatch`].
    pub outputs: OutputBuffers,
    /// Host-observed dispatch duration.
    pub wall_ns: u64,
    /// Device-observed elapsed time when the backend exposes a timer.
    pub device_ns: Option<u64>,
    /// Host time spent enqueueing backend work before the caller begins
    /// waiting for completion.
    pub enqueue_ns: Option<u64>,
    /// Host time spent waiting for completion and collecting output buffers.
    pub wait_ns: Option<u64>,
}

// Inline: `vyre_driver::backend` is `pub(crate)`, so no integration test can reach what this suite
// exercises.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DispatchConfig;

    /// Rows of one batch, generated so the row count, the row widths and the
    /// zero-length rows all vary independently of each other.
    fn generated_batch_rows(case: usize) -> Vec<Vec<u8>> {
        let row_count = (case % 7) + 1;
        (0..row_count)
            .map(|row| {
                let len = (case * 31 + row * 13) % 257;
                (0..len).map(|byte| (case + row + byte) as u8).collect()
            })
            .collect()
    }

    #[test]
    fn generated_batch_rows_are_recoverable_by_dispatch_index() {
        for case in 0..4096 {
            let rows = generated_batch_rows(case);
            let mut batch = BatchOutputs::default();
            let bytes: usize = rows.iter().map(Vec::len).sum();
            batch
                .try_reserve(rows.len(), bytes)
                .expect("Fix: batched row reservation must succeed for a generated batch");
            for row in &rows {
                batch.push_row(row);
            }

            assert_eq!(batch.len(), rows.len(), "case {case} lost a dispatch row");
            assert_eq!(
                batch.total_bytes(),
                bytes,
                "case {case} readback total drifted from the rows pushed"
            );
            for (index, row) in rows.iter().enumerate() {
                assert_eq!(
                    batch.row(index),
                    Some(row.as_slice()),
                    "case {case} row {index} did not read back the bytes it was given"
                );
            }
            assert_eq!(
                batch.row(rows.len()),
                None,
                "case {case} handed out a row past the end of the batch"
            );
            let borrowed: Vec<&[u8]> = batch.rows().collect();
            let expected: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
            assert_eq!(
                borrowed, expected,
                "case {case} iterated its rows out of submission order"
            );
        }
    }

    #[test]
    fn generated_batch_budget_accepts_exact_total_and_rejects_one_byte_less() {
        for case in 0..4096 {
            let mut batch = BatchOutputs::default();
            for row in generated_batch_rows(case) {
                batch.push_row(&row);
            }
            let exact_total = batch.total_bytes();

            let mut exact_config = DispatchConfig::default();
            exact_config.max_output_bytes = Some(exact_total);
            crate::program_walks::enforce_output_budget(&exact_config, batch.total_bytes())
                .expect("Fix: exact batched readback budget must be accepted");

            if exact_total == 0 {
                continue;
            }
            let mut too_small_config = DispatchConfig::default();
            too_small_config.max_output_bytes = Some(exact_total - 1);
            let error =
                crate::program_walks::enforce_output_budget(&too_small_config, batch.total_bytes())
                    .expect_err("Fix: batched readback budget one byte below total must reject");
            assert!(
                error.to_string().contains("max_output_bytes"),
                "batched budget rejection must name the violated policy, got {error}"
            );
        }
    }

    #[test]
    fn replace_output_buffers_preserves_existing_slots() {
        let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(4)];
        let outputs_addr = outputs.as_ptr() as usize;
        let first_slot_addr = outputs[0].as_ptr() as usize;
        let second_slot_addr = outputs[1].as_ptr() as usize;

        replace_output_buffers_preserving_slots(vec![vec![1, 2], vec![3]], &mut outputs);

        assert_eq!(outputs, vec![vec![1, 2], vec![3]]);
        assert_eq!(outputs.as_ptr() as usize, outputs_addr);
        assert_eq!(outputs[0].as_ptr() as usize, first_slot_addr);
        assert_eq!(outputs[1].as_ptr() as usize, second_slot_addr);
    }

    #[test]
    fn replace_output_buffers_truncates_without_dropping_reused_slots() {
        let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(4)];
        let outputs_addr = outputs.as_ptr() as usize;
        let first_slot_addr = outputs[0].as_ptr() as usize;

        replace_output_buffers_preserving_slots(vec![vec![9]], &mut outputs);

        assert_eq!(outputs, vec![vec![9]]);
        assert_eq!(outputs.as_ptr() as usize, outputs_addr);
        assert_eq!(outputs[0].as_ptr() as usize, first_slot_addr);
    }

    #[test]
    fn replace_output_buffers_moves_oversized_incoming_slot_without_copy() {
        let mut outputs = vec![Vec::with_capacity(1)];
        let incoming = vec![vec![1, 2, 3, 4]];
        let incoming_ptr = incoming[0].as_ptr() as usize;

        replace_output_buffers_preserving_slots(incoming, &mut outputs);

        assert_eq!(outputs, vec![vec![1, 2, 3, 4]]);
        assert_eq!(
            outputs[0].as_ptr() as usize,
            incoming_ptr,
            "oversized incoming output should be moved into place instead of copied through a too-small retained slot"
        );
    }

    #[test]
    fn replace_output_buffers_reports_reuse_move_and_append_stats() {
        let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(1)];

        let stats = replace_output_buffers_preserving_slots_with_stats(
            vec![vec![1, 2], vec![3, 4], vec![5]],
            &mut outputs,
        );

        assert_eq!(outputs, vec![vec![1, 2], vec![3, 4], vec![5]]);
        assert_eq!(
            stats,
            OutputSlotStats {
                total_slots: 3,
                reused_slots: 1,
                moved_slots: 1,
                appended_slots: 1,
            }
        );
    }

    #[test]
    fn replace_output_buffers_reserves_outer_slots_before_appending() {
        let mut outputs: OutputBuffers = Vec::with_capacity(3);
        outputs.push(Vec::with_capacity(4));
        outputs[0].extend_from_slice(&[0xaa]);
        let outer_ptr = outputs.as_ptr() as usize;
        let first_slot_ptr = outputs[0].as_ptr() as usize;

        let stats = replace_output_buffers_preserving_slots_with_memory_stats(
            vec![vec![1, 2], vec![3], vec![4, 5, 6]],
            &mut outputs,
        );

        assert_eq!(outputs, vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
        assert_eq!(
            outputs.as_ptr() as usize,
            outer_ptr,
            "outer output vector had enough capacity and must not reallocate while appending new readback slots"
        );
        assert_eq!(
            outputs[0].as_ptr() as usize,
            first_slot_ptr,
            "first output slot should be reused because the incoming bytes fit its retained allocation"
        );
        assert_eq!(stats.slots.appended_slots, 2);
        assert_eq!(stats.bytes.appended_bytes, 4);
    }

    #[test]
    fn replace_output_buffers_reports_byte_pressure_stats() {
        let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(1)];

        let stats = replace_output_buffers_preserving_slots_with_memory_stats(
            vec![vec![1, 2], vec![3, 4], vec![5]],
            &mut outputs,
        );

        assert_eq!(outputs, vec![vec![1, 2], vec![3, 4], vec![5]]);
        assert_eq!(
            stats.bytes,
            OutputSlotByteStats {
                incoming_bytes: 5,
                copied_bytes: 2,
                moved_bytes: 2,
                appended_bytes: 1,
                retained_capacity_bytes: 11,
            }
        );
    }
}
