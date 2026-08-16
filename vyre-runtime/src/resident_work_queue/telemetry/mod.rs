//! Host-side telemetry decoders for the megakernel ring and control buffers.
//!
//! The runtime already exposes low-level helpers such as
//! `read_done_count`, `read_epoch`, and `read_metrics`. This module adds a
//! single structured snapshot surface useful for wrappers like VyreOffload.

use super::policy::{
    PriorityRequeueAccounting, ResidentLaunchPolicy, ResidentLaunchRecommendation,
    ResidentLaunchRequest,
};
use super::protocol::{control, read_word, slot, ARG0_WORD, OPCODE_WORD, STATUS_WORD, TENANT_WORD};
use super::staging_reserve::{
    reserve_hash_map_capacity, reserve_vec_capacity as reserve_target_capacity,
};
use crate::PipelineError;

mod errors;
mod evidence;
mod ring_state;
mod sketch;
pub use evidence::{
    ResidentRuntimeEvidence, RuntimeEvidenceMetricCoverage, RuntimeEvidenceMetricFamily,
    TelemetryDecodeCapacityEvidence, RUNTIME_IO_EVIDENCE_SCHEMA_VERSION,
    TELEMETRY_DECODE_CAPACITY_SCHEMA_VERSION,
};
use ring_state::WindowAccumulator;
pub use ring_state::{
    ControlSnapshot, ResidentRuntimeCounters, ResidentWatchdogSnapshot, RingOccupancy,
    RingSlotSnapshot, RingStatus, RingTelemetry, TelemetryDecodeScratch, WindowTelemetry,
};
pub use sketch::{CountMinSketch, SketchTelemetry, SketchTelemetryScratch};

const SLOT_WORDS_USIZE: usize = 16;

fn try_read_slot_chunk_word(slot_bytes: &[u8], word_idx: u32) -> Result<u32, PipelineError> {
    let word_idx = telemetry_u32_to_usize(word_idx, "slot word index")?;
    let off = word_idx
        .checked_mul(4)
        .ok_or_else(|| errors::slot_word_offset_overflow())?;
    let end = off
        .checked_add(4)
        .ok_or_else(|| errors::slot_word_end_overflow())?;
    let bytes = slot_bytes
        .get(off..end)
        .ok_or_else(|| errors::missing_slot_word(word_idx, slot_bytes.len()))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn is_sorted_unique_u32(values: &[u32]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

impl ControlSnapshot {
    /// Strictly decode a structured control-buffer view into owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when any fixed control word is missing from
    /// the control snapshot.
    pub fn try_decode(control_bytes: &[u8]) -> Result<Self, PipelineError> {
        let mut out = Self::default();
        Self::try_decode_into(control_bytes, &mut out)?;
        Ok(out)
    }

    /// Strictly decode a structured control-buffer view.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when any fixed control word is missing from
    /// the control snapshot.
    pub fn try_decode_into(control_bytes: &[u8], out: &mut Self) -> Result<(), PipelineError> {
        validate_control_snapshot(control_bytes)?;
        out.shutdown =
            read_required_control_word(control_bytes, control_word_index(control::SHUTDOWN)?)? != 0;
        out.done_count =
            read_required_control_word(control_bytes, control_word_index(control::DONE_COUNT)?)?;
        out.epoch = read_required_control_word(control_bytes, control_word_index(control::EPOCH)?)?;
        out.metrics.clear();
        reserve_target_capacity(
            &mut out.metrics,
            telemetry_u32_to_usize(control::METRICS_SLOTS, "metrics slot count")?,
            "metrics",
        )?;
        for i in 0..control::METRICS_SLOTS {
            let count = read_required_control_word(
                control_bytes,
                control_offset_index(control::METRICS_BASE, i)?,
            )?;
            if count > 0 {
                out.metrics.push((i, count));
            }
        }
        out.tenant_fairness.clear();
        reserve_target_capacity(
            &mut out.tenant_fairness,
            telemetry_u32_to_usize(control::TENANT_FAIRNESS_SLOTS, "tenant fairness slot count")?,
            "tenant fairness",
        )?;
        for i in 0..control::TENANT_FAIRNESS_SLOTS {
            out.tenant_fairness.push(read_required_control_word(
                control_bytes,
                control_offset_index(control::TENANT_FAIRNESS_BASE, i)?,
            )?);
        }
        out.priority_fairness.clear();
        reserve_target_capacity(
            &mut out.priority_fairness,
            telemetry_u32_to_usize(
                control::PRIORITY_FAIRNESS_SLOTS,
                "priority fairness slot count",
            )?,
            "priority fairness",
        )?;
        for i in 0..control::PRIORITY_FAIRNESS_SLOTS {
            out.priority_fairness.push(read_required_control_word(
                control_bytes,
                control_offset_index(control::PRIORITY_FAIRNESS_BASE, i)?,
            )?);
        }
        Ok(())
    }
}

impl RingTelemetry {
    /// Decode the ring and control buffers into one structured snapshot.
    #[must_use]
    #[cfg(test)]
    pub fn decode(control_bytes: &[u8], ring_bytes: &[u8]) -> Self {
        Self::decode_with_window_opcodes(control_bytes, ring_bytes, &[])
    }

    /// Strictly decode ring and control bytes after validating ABI alignment.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when buffers are truncated or not aligned to
    /// the megakernel wire protocol.
    pub fn try_decode(control_bytes: &[u8], ring_bytes: &[u8]) -> Result<Self, PipelineError> {
        Self::try_decode_with_window_opcodes(control_bytes, ring_bytes, &[])
    }

    /// Decode the ring and control buffers, additionally grouping any slots
    /// whose opcode is present in `window_opcodes` into ticketed route-window
    /// telemetry records.
    #[must_use]
    #[cfg(test)]
    pub fn decode_with_window_opcodes(
        control_bytes: &[u8],
        ring_bytes: &[u8],
        window_opcodes: &[u32],
    ) -> Self {
        Self::try_decode_with_window_opcodes(control_bytes, ring_bytes, window_opcodes)
            .unwrap_or_default()
    }

    /// Decode the ring and control buffers into caller-owned telemetry and
    /// scratch storage.
    #[cfg(test)]
    pub fn decode_with_window_opcodes_into(
        control_bytes: &[u8],
        ring_bytes: &[u8],
        window_opcodes: &[u32],
        out: &mut Self,
        scratch: &mut TelemetryDecodeScratch,
    ) {
        Self::try_decode_with_window_opcodes_into(
            control_bytes,
            ring_bytes,
            window_opcodes,
            out,
            scratch,
        )
        .unwrap_or_else(|_| {
            *out = Self::default();
            scratch.clear();
        });
    }

    fn try_decode_with_window_opcodes_into_unchecked(
        control_bytes: &[u8],
        ring_bytes: &[u8],
        window_opcodes: &[u32],
        out: &mut Self,
        scratch: &mut TelemetryDecodeScratch,
    ) -> Result<(), PipelineError> {
        enum WindowOpcodeMatcher<'a> {
            None,
            Single(u32),
            DenseBitmap(u128),
            SmallSlice(&'a [u32]),
            LargeSlice(&'a [u32]),
        }

        ControlSnapshot::try_decode_into(control_bytes, &mut out.control)?;
        let slot_count = ring_bytes.len() / slot_byte_len()?;
        out.occupancy = RingOccupancy::default();
        out.slots.clear();
        reserve_target_capacity(&mut out.slots, slot_count, "ring slots")?;
        out.windows.clear();
        scratch.window_opcodes.clear();
        scratch.windows.clear();
        let window_opcode_lookup = if window_opcodes.is_empty() {
            &[][..]
        } else if is_sorted_unique_u32(window_opcodes) {
            window_opcodes
        } else {
            reserve_target_capacity(
                &mut scratch.window_opcodes,
                window_opcodes.len(),
                "window opcode scratch",
            )?;
            scratch.window_opcodes.extend_from_slice(window_opcodes);
            scratch.window_opcodes.sort_unstable();
            scratch.window_opcodes.dedup();
            scratch.window_opcodes.as_slice()
        };
        let window_opcode_matcher = match window_opcode_lookup {
            [] => WindowOpcodeMatcher::None,
            [opcode] => WindowOpcodeMatcher::Single(*opcode),
            opcodes if opcodes.len() > 1 && opcodes.iter().all(|opcode| *opcode < 128) => {
                let bitmap = opcodes
                    .iter()
                    .fold(0_u128, |acc, &opcode| acc | (1_u128 << opcode));
                WindowOpcodeMatcher::DenseBitmap(bitmap)
            }
            opcodes if opcodes.len() <= 8 => WindowOpcodeMatcher::SmallSlice(opcodes),
            opcodes => WindowOpcodeMatcher::LargeSlice(opcodes),
        };
        if !matches!(window_opcode_matcher, WindowOpcodeMatcher::None) {
            reserve_hash_map_capacity(
                &mut scratch.windows,
                slot_count,
                "window accumulator scratch",
            )?;
        }
        let decode_windows = !matches!(window_opcode_matcher, WindowOpcodeMatcher::None);

        let slot_byte_len = slot_byte_len()?;
        for (slot_idx, slot_bytes) in ring_bytes.chunks_exact(slot_byte_len).enumerate() {
            let slot_idx = u32::try_from(slot_idx).map_err(|source| {
                PipelineError::Backend(format!(
                    "megakernel telemetry slot index cannot fit u32: {source}. Fix: shard ring snapshots before host decode."
                ))
            })?;
            let status_raw = try_read_slot_chunk_word(slot_bytes, STATUS_WORD)?;
            let status = RingStatus::from_raw(status_raw);
            match status {
                RingStatus::Empty => out.occupancy.empty += 1,
                RingStatus::Published => out.occupancy.published += 1,
                RingStatus::Claimed => out.occupancy.claimed += 1,
                RingStatus::Done => out.occupancy.done += 1,
                RingStatus::WaitIo => out.occupancy.wait_io += 1,
                RingStatus::Yield => out.occupancy.yield_count += 1,
                RingStatus::Requeue => out.occupancy.requeue += 1,
                RingStatus::Fault => out.occupancy.fault += 1,
                RingStatus::Unknown(_) => out.occupancy.unknown += 1,
            }
            let tenant_id = try_read_slot_chunk_word(slot_bytes, TENANT_WORD)?;
            let opcode = try_read_slot_chunk_word(slot_bytes, OPCODE_WORD)?;
            let args_prefix = [
                try_read_slot_chunk_word(slot_bytes, ARG0_WORD)?,
                try_read_slot_chunk_word(slot_bytes, ARG0_WORD + 1)?,
                try_read_slot_chunk_word(slot_bytes, ARG0_WORD + 2)?,
            ];
            let is_window_opcode = match window_opcode_matcher {
                WindowOpcodeMatcher::None => false,
                WindowOpcodeMatcher::Single(expected) => opcode == expected,
                WindowOpcodeMatcher::DenseBitmap(bitmap) => {
                    opcode < 128 && ((bitmap >> opcode) & 1) == 1
                }
                WindowOpcodeMatcher::SmallSlice(window_opcodes) => window_opcodes.contains(&opcode),
                WindowOpcodeMatcher::LargeSlice(window_opcodes) => {
                    window_opcodes.binary_search(&opcode).is_ok()
                }
            };
            if decode_windows && is_window_opcode {
                let ticket = args_prefix[0];
                let class_tag = args_prefix[1];
                let entry =
                    scratch
                        .windows
                        .entry((ticket, opcode))
                        .or_insert_with(|| WindowAccumulator {
                            tenant_id,
                            opcode,
                            ..WindowAccumulator::default()
                        });
                match class_tag {
                    0 => entry.required_slots += 1,
                    1 => entry.lookahead_slots += 1,
                    _ => {}
                }
                match status {
                    RingStatus::Published => entry.published += 1,
                    RingStatus::Claimed => entry.claimed += 1,
                    RingStatus::Done => entry.done += 1,
                    RingStatus::WaitIo => entry.wait_io += 1,
                    RingStatus::Yield => entry.yield_count += 1,
                    RingStatus::Requeue => entry.requeue += 1,
                    RingStatus::Fault => entry.fault += 1,
                    RingStatus::Empty | RingStatus::Unknown(_) => {}
                }
            }
            out.slots.push(RingSlotSnapshot {
                slot_idx,
                status,
                tenant_id,
                opcode,
                args_prefix,
            });
        }

        reserve_target_capacity(&mut out.windows, scratch.windows.len(), "window output")?;
        for (&(ticket, _), acc) in &scratch.windows {
            out.windows.push(WindowTelemetry {
                ticket,
                tenant_id: acc.tenant_id,
                opcode: acc.opcode,
                required_slots: acc.required_slots,
                lookahead_slots: acc.lookahead_slots,
                published: acc.published,
                claimed: acc.claimed,
                done: acc.done,
                wait_io: acc.wait_io,
                yield_count: acc.yield_count,
                requeue: acc.requeue,
                fault: acc.fault,
            });
        }
        out.windows
            .sort_unstable_by_key(|window| (window.ticket, window.opcode));
        Ok(())
    }

    /// Strictly decode ring/control bytes and group selected window opcodes.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when buffers are truncated or not aligned to
    /// the megakernel wire protocol.
    pub fn try_decode_with_window_opcodes(
        control_bytes: &[u8],
        ring_bytes: &[u8],
        window_opcodes: &[u32],
    ) -> Result<Self, PipelineError> {
        validate_telemetry_buffers(control_bytes, ring_bytes)?;
        let mut out = Self::default();
        let mut scratch = TelemetryDecodeScratch::new();
        Self::try_decode_with_window_opcodes_into_unchecked(
            control_bytes,
            ring_bytes,
            window_opcodes,
            &mut out,
            &mut scratch,
        )?;
        Ok(out)
    }

    /// Strictly decode ring/control bytes into caller-owned telemetry and
    /// scratch storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when buffers are truncated or not aligned to
    /// the megakernel wire protocol.
    pub fn try_decode_with_window_opcodes_into(
        control_bytes: &[u8],
        ring_bytes: &[u8],
        window_opcodes: &[u32],
        out: &mut Self,
        scratch: &mut TelemetryDecodeScratch,
    ) -> Result<(), PipelineError> {
        validate_telemetry_buffers(control_bytes, ring_bytes)?;
        Self::try_decode_with_window_opcodes_into_unchecked(
            control_bytes,
            ring_bytes,
            window_opcodes,
            out,
            scratch,
        )?;
        Ok(())
    }

    /// Active slots matching a given opcode.
    #[must_use]
    #[cfg(test)]
    pub fn active_slots_for_opcode(&self, opcode: u32) -> Vec<&RingSlotSnapshot> {
        self.try_active_slots_for_opcode(opcode).unwrap_or_default()
    }

    /// Active slots matching a given opcode with fallible output staging.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when output storage cannot be reserved.
    pub fn try_active_slots_for_opcode(
        &self,
        opcode: u32,
    ) -> Result<Vec<&RingSlotSnapshot>, PipelineError> {
        let mut out = Vec::default();
        self.try_active_slots_for_opcode_into(opcode, &mut out)?;
        Ok(out)
    }

    /// Active slots matching a given opcode as an iterator.
    pub fn active_slots_for_opcode_iter(
        &self,
        opcode: u32,
    ) -> impl Iterator<Item = &RingSlotSnapshot> {
        self.slots
            .iter()
            .filter(move |slot| slot.opcode == opcode && slot.status.is_active())
    }

    /// Active slots matching a given opcode into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when output storage cannot be reserved.
    pub fn try_active_slots_for_opcode_into<'a>(
        &'a self,
        opcode: u32,
        out: &mut Vec<&'a RingSlotSnapshot>,
    ) -> Result<(), PipelineError> {
        out.clear();
        reserve_target_capacity(out, self.slots.len(), "active slot output")?;
        self.slots
            .iter()
            .filter(|slot| slot.opcode == opcode && slot.status.is_active())
            .for_each(|slot| out.push(slot));
        Ok(())
    }

    /// Unfinished ticketed windows.
    #[must_use]
    #[cfg(test)]
    pub fn active_windows(&self) -> Vec<&WindowTelemetry> {
        self.try_active_windows().unwrap_or_default()
    }

    /// Unfinished ticketed windows with fallible output staging.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when output storage cannot be reserved.
    pub fn try_active_windows(&self) -> Result<Vec<&WindowTelemetry>, PipelineError> {
        let mut out = Vec::default();
        self.try_active_windows_into(&mut out)?;
        Ok(out)
    }

    /// Unfinished ticketed windows into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when output storage cannot be reserved.
    pub fn try_active_windows_into<'a>(
        &'a self,
        out: &mut Vec<&'a WindowTelemetry>,
    ) -> Result<(), PipelineError> {
        out.clear();
        reserve_target_capacity(out, self.windows.len(), "active window output")?;
        self.windows
            .iter()
            .filter(|window| window.is_active())
            .for_each(|window| out.push(window));
        Ok(())
    }

    /// Summarize priority requeue/aging pressure visible in the ring snapshot.
    #[must_use]
    pub fn priority_accounting(&self) -> PriorityRequeueAccounting {
        PriorityRequeueAccounting {
            requeue_count: u64::from(self.occupancy.requeue),
            aged_promotions: 0,
            max_priority_age: 0,
        }
    }

    /// Fallibly aggregate queue, idle, fairness, and drain counters.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when counter aggregation overflows or decoded
    /// telemetry contains an impossible relationship.
    pub fn try_runtime_counters(&self) -> Result<ResidentRuntimeCounters, PipelineError> {
        let total_slots = self.occupancy.total_slots()?;
        let queue_depth = self.occupancy.queue_depth()?;
        let gpu_idle_slots = self.occupancy.empty;
        let gpu_idle_ppm = if total_slots == 0 {
            0
        } else {
            let raw_idle_ppm = (u64::from(gpu_idle_slots) * 1_000_000) / u64::from(total_slots);
            raw_idle_ppm.min(1_000_000) as u32
        };
        let frontier_density_bps = try_density_bps(queue_depth, total_slots)?;
        let active_slots = total_slots.saturating_sub(gpu_idle_slots);
        let occupancy_proxy_bps = try_density_bps(active_slots, total_slots)?;
        let tenant_fairness_total = try_sum_u32_as_u64(
            &self.control.tenant_fairness,
            "tenant fairness total",
            "shard tenant counters before telemetry aggregation",
        )?;
        let priority_fairness_total = try_sum_u32_as_u64(
            &self.control.priority_fairness,
            "priority fairness total",
            "shard priority counters before telemetry aggregation",
        )?;
        let tenant_fairness_skew = try_fairness_skew(&self.control.tenant_fairness)?;
        Ok(ResidentRuntimeCounters {
            total_slots,
            queue_depth,
            gpu_idle_slots,
            gpu_idle_ppm,
            frontier_density_bps,
            occupancy_proxy_bps,
            drained_slots: self.control.done_count,
            unreclaimed_done_slots: self.occupancy.done,
            tenant_fairness_total,
            tenant_fairness_skew,
            priority_fairness_total,
            requeue_slots: self.occupancy.requeue,
            fault_slots: self.occupancy.fault,
        })
    }

    /// Fallibly derive persistent-kernel health from two snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when counters wrap, move backwards, or cannot
    /// be aggregated without overflow.
    pub fn try_health_since(
        &self,
        previous: &RingTelemetry,
    ) -> Result<ResidentWatchdogSnapshot, PipelineError> {
        let counters = self.try_runtime_counters()?;
        let done_delta = self
            .control
            .done_count
            .checked_sub(previous.control.done_count)
            .ok_or_else(|| {
                errors::done_counter_backwards(previous.control.done_count, self.control.done_count)
            })?;
        let suspected_stall =
            counters.queue_depth > 0 && done_delta == 0 && counters.fault_slots == 0;
        Ok(ResidentWatchdogSnapshot {
            done_delta,
            queue_depth: counters.queue_depth,
            fault_slots: counters.fault_slots,
            requeue_slots: counters.requeue_slots,
            gpu_idle_ppm: counters.gpu_idle_ppm,
            suspected_stall,
        })
    }

    /// Feed telemetry into the shared launch policy.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the supplied adapter limits are malformed.
    pub fn recommend_launch(
        &self,
        mut request: ResidentLaunchRequest,
    ) -> Result<ResidentLaunchRecommendation, vyre_driver::BackendError> {
        let counters = self
            .try_runtime_counters()
            .map_err(errors::launch_telemetry_failed)?;
        if request.graph_node_count == 0 {
            request.graph_node_count = counters.total_slots;
        }
        if request.graph_edge_count == 0 {
            request.graph_edge_count = counters.queue_depth;
        }
        if request.frontier_density_bps == 0 {
            request.frontier_density_bps = counters.frontier_density_bps;
        }
        request.hot_opcode_count = self
            .control
            .metrics
            .iter()
            .filter(|(_, count)| *count > 0)
            .count()
            .try_into()
            .map_err(errors::hot_opcode_count_overflow)?;
        let mut hot_window_count = 0usize;
        for window in &self.windows {
            let demand = window
                .required_slots
                .checked_add(window.lookahead_slots)
                .ok_or_else(|| errors::route_window_demand_overflow())?;
            if demand >= 4 {
                hot_window_count = hot_window_count
                    .checked_add(1)
                    .ok_or_else(|| errors::hot_window_count_overflow())?;
            }
        }
        request.hot_window_count = hot_window_count
            .try_into()
            .map_err(errors::hot_window_count_too_wide)?;
        request.requeue_count = request
            .requeue_count
            .checked_add(u64::from(self.occupancy.requeue))
            .ok_or_else(errors::requeue_count_overflow)?;
        ResidentLaunchPolicy::standard().recommend(request)
    }
}

fn read_required_control_word(control_bytes: &[u8], word_idx: usize) -> Result<u32, PipelineError> {
    read_word(control_bytes, word_idx).ok_or_else(|| errors::missing_control_word(word_idx))
}

fn try_density_bps(numerator: u32, denominator: u32) -> Result<u16, PipelineError> {
    if denominator == 0 {
        return Ok(0);
    }
    let bps = (u64::from(numerator) * 10_000) / u64::from(denominator);
    u16::try_from(bps.min(u64::from(u16::MAX))).map_err(errors::density_bps_overflow)
}

fn validate_telemetry_buffers(
    control_bytes: &[u8],
    ring_bytes: &[u8],
) -> Result<(), PipelineError> {
    validate_control_snapshot(control_bytes)?;
    let slot_bytes = slot_byte_len()?;
    if ring_bytes.len() % slot_bytes != 0 {
        return Err(errors::ring_slot_alignment(ring_bytes.len(), slot_bytes));
    }
    let slot_count = ring_bytes.len() / slot_bytes;
    if u32::try_from(slot_count).is_err() {
        return Err(errors::ring_slot_count_too_wide(slot_count));
    }
    Ok(())
}

fn validate_control_snapshot(control_bytes: &[u8]) -> Result<(), PipelineError> {
    let min_control =
        super::protocol::control_byte_len(0).ok_or_else(|| errors::control_length_overflow())?;
    if control_bytes.len() < min_control || control_bytes.len() % 4 != 0 {
        return Err(errors::bad_control_snapshot(
            control_bytes.len(),
            min_control,
        ));
    }
    Ok(())
}

fn slot_byte_len() -> Result<usize, PipelineError> {
    SLOT_WORDS_USIZE
        .checked_mul(4)
        .ok_or_else(|| errors::slot_byte_width_overflow())
}

fn telemetry_u32_to_usize(value: u32, label: &'static str) -> Result<usize, PipelineError> {
    usize::try_from(value).map_err(|source| errors::telemetry_u32_to_usize(value, label, source))
}

fn control_word_index(word: u32) -> Result<usize, PipelineError> {
    usize::try_from(word).map_err(|source| errors::control_word_index(word, source))
}

fn control_offset_index(base: u32, offset: u32) -> Result<usize, PipelineError> {
    let word = base
        .checked_add(offset)
        .ok_or_else(|| errors::control_word_offset_overflow())?;
    control_word_index(word)
}

fn try_sum_u32_as_u64(
    counters: &[u32],
    label: &'static str,
    fix: &'static str,
) -> Result<u64, PipelineError> {
    counters.iter().try_fold(0u64, |acc, &count| {
        acc.checked_add(u64::from(count))
            .ok_or_else(|| errors::counter_sum_overflow(label, fix))
    })
}

fn try_fairness_skew(counters: &[u32]) -> Result<u32, PipelineError> {
    let mut min_nonzero = u32::MAX;
    let mut max = 0u32;
    for &count in counters {
        if count != 0 {
            min_nonzero = min_nonzero.min(count);
            max = max.max(count);
        }
    }
    if min_nonzero == u32::MAX {
        Ok(0)
    } else {
        max.checked_sub(min_nonzero)
            .ok_or_else(|| errors::fairness_skew_invalid(max, min_nonzero))
    }
}

// Inline: the suite drives the `#[cfg(test)]` panicking accessors (`decode`,
// `active_slots_for_opcode`, `active_windows`), which an integration test cannot
// reach. The panicking counter and health accessors it also used to drive are
// gone; their cases now run through `try_runtime_counters` and `try_health_since`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_work_queue::descriptor::WindowClass;
    use crate::resident_work_queue::policy::{
        ResidentExecutionMode, ResidentLaunchRequest, ResidentQueueTopology,
    };
    use crate::resident_work_queue::protocol::{opcode, SLOT_WORDS};
    use crate::resident_work_queue::ResidentWorkQueue;

    mod decode_contracts {
        use super::*;

        #[test]
        fn decode_empty_ring_counts_slots() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let telemetry = RingTelemetry::decode(&control, &ring);
            assert_eq!(telemetry.occupancy.empty, 4);
            assert_eq!(telemetry.occupancy.published, 0);
            assert_eq!(telemetry.slots.len(), 4);
            assert!(telemetry.windows.is_empty());
        }

        #[test]
        fn strict_decode_rejects_trailing_partial_slot() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
            ring.push(0);
            let err = RingTelemetry::try_decode(&control, &ring)
                .expect_err("Fix: strict telemetry must reject malformed ring snapshots");
            assert!(matches!(err, PipelineError::Backend(_)));
        }

        #[test]
        fn strict_decode_rejects_misaligned_control_snapshot() {
            let mut control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            control.push(0xFF);
            let ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
            let err = RingTelemetry::try_decode(&control, &ring)
                .expect_err("Fix: strict telemetry must reject malformed control snapshots");
            assert!(matches!(err, PipelineError::Backend(_)));
        }

        /// The control decode error path is the only thing standing between a truncated
        /// device readback and a snapshot that reads as a healthy idle kernel, so the
        /// error must both name the malformed buffer and carry the corrective action.
        #[test]
        fn control_try_decode_rejects_short_snapshot_without_panic() {
            let err = ControlSnapshot::try_decode(&[]).expect_err(
                "Fix: strict control telemetry decode must reject missing control words",
            );
            let message = err.to_string();
            assert!(
                message.contains("control snapshot"),
                "Fix: strict control decode errors must explain the malformed control buffer: {err}"
            );
            assert!(
                message.contains("Fix: capture the full control buffer"),
                "Fix: strict control decode errors must carry the corrective action: {err}"
            );
        }

        /// `try_decode_into` is the caller-owned-storage twin, and it must reject the
        /// same truncated buffer with the same corrective action instead of leaving a
        /// half-written snapshot behind.
        #[test]
        fn control_try_decode_into_rejects_short_snapshot_and_leaves_output_untouched() {
            let mut control = ResidentWorkQueue::try_encode_control(false, 3, 5).unwrap();
            let done_count_offset = (control::DONE_COUNT as usize) * 4;
            control[done_count_offset..done_count_offset + 4].copy_from_slice(&41u32.to_le_bytes());
            let mut out = ControlSnapshot::try_decode(&control)
                .expect("Fix: a well-formed control buffer must decode");
            assert_eq!(out.done_count, 41);

            let err = ControlSnapshot::try_decode_into(&[], &mut out).expect_err(
                "Fix: strict control telemetry decode_into must reject missing control words",
            );
            assert!(
                err.to_string()
                    .contains("Fix: capture the full control buffer"),
                "Fix: strict control decode_into errors must carry the corrective action: {err}"
            );
            assert_eq!(
                out.done_count, 41,
                "Fix: a rejected control buffer must not overwrite the caller's previous snapshot"
            );
        }

        #[test]
        fn strict_decode_into_rejects_trailing_partial_slot_without_mutating_output() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
            ring.push(0);
            let mut telemetry = RingTelemetry::default();
            let mut scratch = TelemetryDecodeScratch::new();

            let err = RingTelemetry::try_decode_with_window_opcodes_into(
                &control,
                &ring,
                &[],
                &mut telemetry,
                &mut scratch,
            )
            .expect_err("Fix: strict telemetry decode_into must reject partial ring slots");

            assert!(
                err.to_string().contains("whole ring slots"),
                "Fix: strict telemetry decode_into errors must explain partial ring slots: {err}"
            );
            assert!(telemetry.slots.is_empty());
        }

        #[test]
        fn decode_published_slot_reads_prefix() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 1, 9, opcode::ATOMIC_ADD, &[5, 7, 11])
                .unwrap();
            let telemetry = RingTelemetry::decode(&control, &ring);
            let slot = &telemetry.slots[1];
            assert_eq!(slot.status, RingStatus::Published);
            assert_eq!(slot.tenant_id, 9);
            assert_eq!(slot.opcode, opcode::ATOMIC_ADD);
            assert_eq!(slot.args_prefix, [5, 7, 11]);
        }
    }

    mod recommendation_runtime_contracts {
        use super::*;

        #[test]
        fn telemetry_recommendation_promotes_hot_opcodes_and_requeue_pressure() {
            let mut control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            for opcode_idx in 0..8u32 {
                let off = ((control::METRICS_BASE + opcode_idx) as usize) * 4;
                control[off..off + 4].copy_from_slice(&1u32.to_le_bytes());
            }
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let status_off = (STATUS_WORD as usize) * 4;
            ring[status_off..status_off + 4].copy_from_slice(&slot::REQUEUE.to_le_bytes());
            let telemetry = RingTelemetry::decode(&control, &ring);
            let rec = telemetry
                .recommend_launch(ResidentLaunchRequest::direct(4096, 64, 256))
                .expect("Fix: telemetry launch recommendation must accept valid limits");
            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
            assert!(rec.promote_hot_opcodes);
            assert!(rec.age_priority_work);
            assert_eq!(telemetry.priority_accounting().requeue_count, 1);
        }

        #[test]
        fn runtime_counters_report_queue_idle_fairness_and_drain() {
            let mut control = ResidentWorkQueue::try_encode_control(false, 7, 0).unwrap();
            let tenant_a = (control::TENANT_FAIRNESS_BASE as usize) * 4;
            let tenant_b = ((control::TENANT_FAIRNESS_BASE + 1) as usize) * 4;
            let priority_a = (control::PRIORITY_FAIRNESS_BASE as usize) * 4;
            let done_count = (control::DONE_COUNT as usize) * 4;
            control[done_count..done_count + 4].copy_from_slice(&7u32.to_le_bytes());
            control[tenant_a..tenant_a + 4].copy_from_slice(&3u32.to_le_bytes());
            control[tenant_b..tenant_b + 4].copy_from_slice(&9u32.to_le_bytes());
            control[priority_a..priority_a + 4].copy_from_slice(&5u32.to_le_bytes());

            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 2, 11, opcode::ATOMIC_ADD, &[1, 2, 3])
                .unwrap();
            let slot_status =
                |slot_idx: usize| slot_idx * (SLOT_WORDS as usize) * 4 + (STATUS_WORD as usize) * 4;
            let requeue = slot_status(0);
            ring[requeue..requeue + 4].copy_from_slice(&slot::REQUEUE.to_le_bytes());
            let done = slot_status(1);
            ring[done..done + 4].copy_from_slice(&slot::DONE.to_le_bytes());

            let counters = RingTelemetry::decode(&control, &ring)
                .try_runtime_counters()
                .expect("Fix: a four-slot ring must aggregate without overflow");
            assert_eq!(counters.total_slots, 4);
            assert_eq!(counters.queue_depth, 2);
            assert_eq!(counters.gpu_idle_slots, 1);
            assert_eq!(counters.gpu_idle_ppm, 250_000);
            assert_eq!(counters.frontier_density_bps, 5_000);
            assert_eq!(counters.occupancy_proxy_bps, 7_500);
            assert_eq!(counters.drained_slots, 7);
            assert_eq!(counters.unreclaimed_done_slots, 1);
            assert_eq!(counters.tenant_fairness_total, 12);
            assert_eq!(counters.tenant_fairness_skew, 6);
            assert_eq!(counters.priority_fairness_total, 5);
            assert_eq!(counters.requeue_slots, 1);
        }

        #[test]
        fn telemetry_launch_recommendation_uses_frontier_density_for_topology() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(8).unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 0, 7, opcode::ATOMIC_ADD, &[1, 2, 3])
                .unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 1, 7, opcode::ATOMIC_ADD, &[1, 2, 3])
                .unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 2, 7, opcode::ATOMIC_ADD, &[1, 2, 3])
                .unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 3, 7, opcode::ATOMIC_ADD, &[1, 2, 3])
                .unwrap();

            let telemetry = RingTelemetry::decode(&control, &ring);
            let rec = telemetry
                .recommend_launch(ResidentLaunchRequest::direct(8, 64, 256))
                .expect("Fix: telemetry launch recommendation must accept valid limits");

            assert_eq!(
                telemetry
                    .try_runtime_counters()
                    .expect("Fix: an eight-slot ring must aggregate without overflow")
                    .frontier_density_bps,
                5_000
            );
            assert_eq!(rec.topology, ResidentQueueTopology::DenseFrontier);
        }

        #[test]
        fn telemetry_decode_into_reports_caller_owned_capacity_evidence() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 0, 7, opcode::ATOMIC_ADD, &[11, 0, 0])
                .unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 1, 7, opcode::ATOMIC_ADD, &[11, 1, 0])
                .unwrap();
            let mut telemetry = RingTelemetry::default();
            let mut scratch = TelemetryDecodeScratch::new();

            RingTelemetry::try_decode_with_window_opcodes_into(
                &control,
                &ring,
                &[opcode::ATOMIC_ADD, opcode::ATOMIC_ADD],
                &mut telemetry,
                &mut scratch,
            )
            .expect("Fix: strict telemetry decode should accept aligned ring/control snapshots");
            let evidence = telemetry.decode_capacity_evidence(&scratch);

            assert_eq!(
                evidence.schema_version,
                TELEMETRY_DECODE_CAPACITY_SCHEMA_VERSION
            );
            assert_eq!(evidence.decoded_slot_count, 2);
            assert!(evidence.slot_output_capacity >= 2);
            assert_eq!(evidence.decoded_window_count, 1);
            assert!(evidence.window_output_capacity >= 1);
            assert!(evidence.window_opcode_scratch_capacity >= 2);
            assert!(evidence.window_accumulator_scratch_capacity >= 2);
            assert!(evidence.uses_caller_owned_scratch);
            assert!(evidence.is_complete());
        }

        #[test]
        fn launch_recommendation_rejects_route_window_demand_overflow_without_panic() {
            let telemetry = RingTelemetry {
                windows: vec![WindowTelemetry {
                    ticket: 1,
                    tenant_id: 1,
                    opcode: opcode::ATOMIC_ADD,
                    required_slots: u32::MAX,
                    lookahead_slots: 1,
                    published: 0,
                    claimed: 0,
                    done: 0,
                    wait_io: 0,
                    yield_count: 0,
                    requeue: 0,
                    fault: 0,
                }],
                ..RingTelemetry::default()
            };

            let error = telemetry
                .recommend_launch(ResidentLaunchRequest::direct(4096, 64, 256))
                .expect_err(
                    "Fix: route-window demand overflow must not panic during launch recommendation",
                );
            assert!(
                error
                    .to_string()
                    .contains("route-window slot demand overflowed"),
                "Fix: launch recommendation overflow errors must identify route-window demand: {error}"
            );
        }

        #[test]
        fn metrics_and_observable_regions_remain_non_overlapping_in_snapshot() {
            let mut control = ResidentWorkQueue::try_encode_control(false, 1, 4).unwrap();
            let metric_off = (control::METRICS_BASE as usize) * 4;
            control[metric_off..metric_off + 4].copy_from_slice(&0xAA55AA55u32.to_le_bytes());
            let observable_off = (control::OBSERVABLE_BASE as usize) * 4;
            control[observable_off..observable_off + 4]
                .copy_from_slice(&0x11223344u32.to_le_bytes());

            let ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
            let telemetry = RingTelemetry::decode(&control, &ring);
            assert!(
                telemetry.control.metrics.contains(&(0, 0xAA55AA55)),
                "metrics decoder must preserve metric slot 0 value"
            );
            assert_eq!(
                ResidentWorkQueue::read_observable(&control, 0),
                0x11223344,
                "observable reads must not alias metric region words"
            );
        }
    }

    mod sketch_watchdog_contracts {
        use super::*;

        #[test]
        fn sketch_into_reuses_counter_storage() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            ResidentWorkQueue::publish_slot(&mut ring, 1, 9, opcode::ATOMIC_ADD, &[5, 7, 11])
                .unwrap();
            let telemetry = RingTelemetry::decode(&control, &ring);
            let mut scratch = SketchTelemetryScratch::new(3, 16).unwrap();

            telemetry.sketch_into(3, 16, &mut scratch).unwrap();
            let ring_ptr = scratch.ring_opcode.counters().as_ptr();
            let active_ptr = scratch.active_opcode.counters().as_ptr();
            let tenant_ptr = scratch.tenant.counters().as_ptr();
            let status_ptr = scratch.status.counters().as_ptr();
            let metrics_ptr = scratch.dispatch_metrics.counters().as_ptr();
            let first_active = scratch.active_slots;

            telemetry.sketch_into(3, 16, &mut scratch).unwrap();

            assert_eq!(scratch.ring_opcode.counters().as_ptr(), ring_ptr);
            assert_eq!(scratch.active_opcode.counters().as_ptr(), active_ptr);
            assert_eq!(scratch.tenant.counters().as_ptr(), tenant_ptr);
            assert_eq!(scratch.status.counters().as_ptr(), status_ptr);
            assert_eq!(scratch.dispatch_metrics.counters().as_ptr(), metrics_ptr);
            assert_eq!(scratch.total_slots, 4);
            assert_eq!(scratch.active_slots, first_active);
            assert!(scratch.ring_opcode.estimate(opcode::ATOMIC_ADD) >= 1);
        }

        #[test]
        fn watchdog_health_flags_active_queue_without_drain_progress() {
            let mut previous_control = ResidentWorkQueue::try_encode_control(false, 7, 0).unwrap();
            let done_count = (control::DONE_COUNT as usize) * 4;
            previous_control[done_count..done_count + 4].copy_from_slice(&7u32.to_le_bytes());
            let previous_ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            let previous = RingTelemetry::decode(&previous_control, &previous_ring);

            let mut current_control = previous_control.clone();
            let mut current_ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            ResidentWorkQueue::publish_slot(
                &mut current_ring,
                0,
                7,
                opcode::ATOMIC_ADD,
                &[1, 2, 3],
            )
            .unwrap();
            let stalled = RingTelemetry::decode(&current_control, &current_ring)
                .try_health_since(&previous)
                .expect("Fix: two well-formed snapshots must derive health without overflow");
            assert_eq!(stalled.done_delta, 0);
            assert_eq!(stalled.queue_depth, 1);
            assert!(stalled.suspected_stall);

            current_control[done_count..done_count + 4].copy_from_slice(&9u32.to_le_bytes());
            let progressed = RingTelemetry::decode(&current_control, &current_ring)
                .try_health_since(&previous)
                .expect("Fix: two well-formed snapshots must derive health without overflow");
            assert_eq!(progressed.done_delta, 2);
            assert!(!progressed.suspected_stall);
        }

        #[test]
        fn watchdog_try_health_rejects_done_counter_wrap_without_panic() {
            let mut previous_control = ResidentWorkQueue::try_encode_control(false, 7, 0).unwrap();
            let done_count = (control::DONE_COUNT as usize) * 4;
            previous_control[done_count..done_count + 4].copy_from_slice(&9u32.to_le_bytes());
            let previous_ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            let previous = RingTelemetry::decode(&previous_control, &previous_ring);

            let mut current_control = previous_control.clone();
            current_control[done_count..done_count + 4].copy_from_slice(&7u32.to_le_bytes());
            let current_ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            let current = RingTelemetry::decode(&current_control, &current_ring);

            let error = current
                .try_health_since(&previous)
                .expect_err("Fix: wrapped done counters must return structured watchdog errors");
            assert!(
                error.to_string().contains("moved backwards"),
                "Fix: watchdog wrap errors must identify the counter relationship: {error}"
            );
        }
    }

    mod window_contracts {
        use super::*;

        #[test]
        fn decode_window_opcodes_groups_ticketed_slots() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let window_opcode = 0xF101;
            ResidentWorkQueue::publish_slot(
                &mut ring,
                0,
                3,
                window_opcode,
                &[7, WindowClass::Required.into_wire(), 42],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                1,
                3,
                window_opcode,
                &[7, WindowClass::Lookahead.into_wire(), 99],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                2,
                3,
                window_opcode,
                &[7, WindowClass::Required.into_wire(), 123],
            )
            .unwrap();
            let telemetry =
                RingTelemetry::decode_with_window_opcodes(&control, &ring, &[window_opcode]);
            assert_eq!(telemetry.windows.len(), 1);
            let window = &telemetry.windows[0];
            assert_eq!(window.ticket, 7);
            assert_eq!(window.tenant_id, 3);
            assert_eq!(window.opcode, window_opcode);
            assert_eq!(window.required_slots, 2);
            assert_eq!(window.lookahead_slots, 1);
            assert_eq!(window.published, 3);
            assert!(window.is_active());
            assert_eq!(telemetry.active_windows().len(), 1);
            assert_eq!(telemetry.active_slots_for_opcode(window_opcode).len(), 3);
            assert_eq!(
                telemetry
                    .active_slots_for_opcode_iter(window_opcode)
                    .count(),
                3
            );
            let mut active_windows = Vec::with_capacity(4);
            let mut active_slots = Vec::with_capacity(4);
            let windows_ptr = active_windows.as_ptr();
            let slots_ptr = active_slots.as_ptr();
            telemetry
                .try_active_windows_into(&mut active_windows)
                .expect(
                    "Fix: active-window staging must fit the caller-owned buffer reserved above",
                );
            telemetry
                .try_active_slots_for_opcode_into(window_opcode, &mut active_slots)
                .expect("Fix: active-slot staging must fit the caller-owned buffer reserved above");
            assert_eq!(active_windows.len(), 1);
            assert_eq!(active_slots.len(), 3);
            assert_eq!(active_windows.as_ptr(), windows_ptr);
            assert_eq!(active_slots.as_ptr(), slots_ptr);
        }

        #[test]
        fn decode_window_opcodes_matches_dense_bitmap_opcodes() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let first_window_opcode = 3u32;
            let second_window_opcode = 9u32;
            ResidentWorkQueue::publish_slot(
                &mut ring,
                0,
                3,
                first_window_opcode,
                &[11, WindowClass::Required.into_wire(), 42],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                1,
                3,
                second_window_opcode,
                &[11, WindowClass::Lookahead.into_wire(), 99],
            )
            .unwrap();
            let telemetry = RingTelemetry::decode_with_window_opcodes(
                &control,
                &ring,
                &[first_window_opcode, second_window_opcode],
            );
            assert_eq!(telemetry.windows.len(), 2);
            assert_eq!(
                telemetry.active_slots_for_opcode(first_window_opcode).len(),
                1
            );
            assert_eq!(
                telemetry
                    .active_slots_for_opcode(second_window_opcode)
                    .len(),
                1
            );
        }

        #[test]
        fn decode_with_scratch_reuses_snapshot_storage() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let window_opcode = 0xF101;
            ResidentWorkQueue::publish_slot(
                &mut ring,
                0,
                3,
                window_opcode,
                &[7, WindowClass::Required.into_wire(), 42],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                1,
                3,
                window_opcode,
                &[7, WindowClass::Lookahead.into_wire(), 99],
            )
            .unwrap();

            let mut telemetry = RingTelemetry {
                control: ControlSnapshot {
                    metrics: Vec::with_capacity(control::METRICS_SLOTS as usize),
                    tenant_fairness: Vec::with_capacity(control::TENANT_FAIRNESS_SLOTS as usize),
                    priority_fairness: Vec::with_capacity(
                        control::PRIORITY_FAIRNESS_SLOTS as usize,
                    ),
                    ..ControlSnapshot::default()
                },
                slots: Vec::with_capacity(4),
                windows: Vec::with_capacity(1),
                ..RingTelemetry::default()
            };
            let mut scratch = TelemetryDecodeScratch::new();

            RingTelemetry::decode_with_window_opcodes_into(
                &control,
                &ring,
                &[window_opcode],
                &mut telemetry,
                &mut scratch,
            );
            let metrics_ptr = telemetry.control.metrics.as_ptr();
            let tenant_ptr = telemetry.control.tenant_fairness.as_ptr();
            let priority_ptr = telemetry.control.priority_fairness.as_ptr();
            let slots_ptr = telemetry.slots.as_ptr();
            let windows_ptr = telemetry.windows.as_ptr();

            RingTelemetry::try_decode_with_window_opcodes_into(
                &control,
                &ring,
                &[window_opcode],
                &mut telemetry,
                &mut scratch,
            )
            .expect("Fix: scratch telemetry decode must accept valid control/ring snapshots");

            assert_eq!(telemetry.control.metrics.as_ptr(), metrics_ptr);
            assert_eq!(telemetry.control.tenant_fairness.as_ptr(), tenant_ptr);
            assert_eq!(telemetry.control.priority_fairness.as_ptr(), priority_ptr);
            assert_eq!(telemetry.slots.as_ptr(), slots_ptr);
            assert_eq!(telemetry.windows.as_ptr(), windows_ptr);
            assert_eq!(telemetry.windows.len(), 1);
            assert_eq!(telemetry.slots.len(), 4);
        }

        #[test]
        fn decode_sorted_window_opcodes_reuses_scratch_without_resort_growth() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
            let first_opcode = 0xF101;
            let second_opcode = 0xF102;
            ResidentWorkQueue::publish_slot(
                &mut ring,
                0,
                3,
                first_opcode,
                &[7, WindowClass::Required.into_wire(), 42],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                1,
                3,
                second_opcode,
                &[9, WindowClass::Lookahead.into_wire(), 99],
            )
            .unwrap();

            let mut telemetry = RingTelemetry::default();
            let mut scratch = TelemetryDecodeScratch::new();
            let sorted_unique = [first_opcode, second_opcode];
            RingTelemetry::decode_with_window_opcodes_into(
                &control,
                &ring,
                &sorted_unique,
                &mut telemetry,
                &mut scratch,
            );
            let opcode_capacity = scratch.window_opcodes.capacity();
            let window_capacity = scratch.windows.capacity();

            RingTelemetry::decode_with_window_opcodes_into(
                &control,
                &ring,
                &sorted_unique,
                &mut telemetry,
                &mut scratch,
            );

            assert_eq!(scratch.window_opcodes.capacity(), opcode_capacity);
            assert_eq!(scratch.windows.capacity(), window_capacity);
            assert_eq!(telemetry.windows.len(), 2);
            assert!(telemetry
                .windows
                .iter()
                .any(|window| window.opcode == first_opcode && window.ticket == 7));
            assert!(telemetry
                .windows
                .iter()
                .any(|window| window.opcode == second_opcode && window.ticket == 9));
        }

        #[test]
        fn terminal_window_is_not_reported_as_active() {
            let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
            let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
            let window_opcode = 0xF101;
            ResidentWorkQueue::publish_slot(
                &mut ring,
                0,
                3,
                window_opcode,
                &[9, WindowClass::Required.into_wire(), 42],
            )
            .unwrap();
            ResidentWorkQueue::publish_slot(
                &mut ring,
                1,
                3,
                window_opcode,
                &[9, WindowClass::Lookahead.into_wire(), 99],
            )
            .unwrap();
            let mut mark_done = |slot_idx: usize| {
                let start = slot_idx * (SLOT_WORDS as usize) * 4 + (STATUS_WORD as usize) * 4;
                ring[start..start + 4].copy_from_slice(&slot::DONE.to_le_bytes());
            };
            mark_done(0);
            mark_done(1);
            let telemetry =
                RingTelemetry::decode_with_window_opcodes(&control, &ring, &[window_opcode]);
            assert_eq!(telemetry.windows.len(), 1);
            assert!(!telemetry.windows[0].is_active());
            assert!(telemetry.active_windows().is_empty());
            assert!(telemetry.active_slots_for_opcode(window_opcode).is_empty());
        }
    }
}
