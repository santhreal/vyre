//! The timestamp resolve a device must pass before it reports timestamp queries.

use std::time::Duration;

/// A resolve on a created device already waited for its submission, so the
/// mapping callback is a driver hand-off rather than device work. This bounds a
/// driver that never delivers it, so device acquisition cannot hang on the
/// capability probe.
const TIMESTAMP_PROBE_READBACK_TIMEOUT: Duration = Duration::from_secs(10);

/// Resolve the timed-dispatch query layout once on a created device and report
/// whether it produced usable timestamps.
///
/// A Metal adapter advertises `TIMESTAMP_QUERY` and
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS`, admits the capability check, and resolves
/// an end-of-pass timestamp of zero. Without this resolve the first timed
/// dispatch reports a delta underflow, which reads as a broken dispatch rather
/// than as an adapter that cannot be timed.
pub(super) fn timestamp_resolve_is_monotonic(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> std::result::Result<(), String> {
    use crate::engine::record_and_readback::timestamp::{
        timestamp_ticks, TIMESTAMP_QUERY_COUNT, TIMESTAMP_READBACK_BYTES,
    };

    const WGSL: &str = r"
@compute @workgroup_size(1)
fn main() {
}
";

    let scope = super::ValidationScope::push(device);
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("vyre timestamp resolve probe"),
        ty: wgpu::QueryType::Timestamp,
        count: TIMESTAMP_QUERY_COUNT,
    });
    let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vyre timestamp resolve probe resolve"),
        size: TIMESTAMP_READBACK_BYTES,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vyre timestamp resolve probe readback"),
        size: TIMESTAMP_READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vyre timestamp resolve probe"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vyre timestamp resolve probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vyre timestamp resolve probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vyre timestamp resolve probe"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_pipeline(&pipeline);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.write_timestamp(&query_set, 2);
    encoder.write_timestamp(&query_set, 3);
    encoder.resolve_query_set(&query_set, 0..TIMESTAMP_QUERY_COUNT, &resolve_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &resolve_buffer,
        0,
        &readback_buffer,
        0,
        TIMESTAMP_READBACK_BYTES,
    );
    let submission = queue.submit(std::iter::once(encoder.finish()));

    let _ = super::poll_device_wait_for(device, submission.clone())
        .map_err(|error| error.to_string())?;
    // After the submission wait, so encoder-recording validation has surfaced
    // and the scope resolves without a second blocking poll.
    match scope.pop_now() {
        Ok(None) => {}
        Ok(Some(error)) => return Err(format!("validation error: {error}")),
        // The resolved ticks are the verdict. An error scope that has not
        // resolved is not evidence against the device, and withdrawing a
        // working capability over it would cost every timed dispatch.
        Err(reason) => tracing::debug!(
            target: "vyre.wgpu.timestamps",
            %reason,
            "timestamp probe error scope did not resolve"
        ),
    }

    let (sender, receiver) = crossbeam_channel::bounded(1);
    readback_buffer
        .slice(0..TIMESTAMP_READBACK_BYTES)
        .map_async(wgpu::MapMode::Read, move |result| {
            drop(sender.send(result));
        });
    // The submission is complete, so this wait returns at once and delivers the
    // mapping callback requested after it.
    let _ = super::poll_device_wait_for(device, submission)
        .map_err(|error| format!("device poll before timestamp probe readback failed: {error}"))?;
    match receiver.recv_timeout(TIMESTAMP_PROBE_READBACK_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(format!("probe readback mapping failed: {error:?}")),
        Err(error) => return Err(format!("probe readback did not complete: {error}")),
    }

    let ticks = {
        let mapped = readback_buffer
            .slice(0..TIMESTAMP_READBACK_BYTES)
            .get_mapped_range();
        if mapped.len() < TIMESTAMP_READBACK_BYTES as usize {
            let len = mapped.len();
            return Err(format!(
                "probe readback returned {len} bytes, expected {TIMESTAMP_READBACK_BYTES}"
            ));
        }
        timestamp_ticks(&mapped)
    };
    readback_buffer.unmap();

    timestamp_ticks_are_usable(&ticks)
}

/// Pairs a timed dispatch subtracts: the compute pass, the encoder bracket, and
/// the whole submission. A resolve is usable only if every one of them advances.
const TIMESTAMP_MONOTONIC_PAIRS: [(usize, usize); 3] = [(1, 0), (3, 2), (3, 0)];

/// A resolve is usable when each recorded pair advances and the whole set is
/// not zero. An all-zero resolve advertises a device time of zero for every
/// dispatch, which is not a measurement.
fn timestamp_ticks_are_usable(ticks: &[u64]) -> std::result::Result<(), String> {
    if ticks.iter().all(|tick| *tick == 0) {
        return Err("resolve returned only zero timestamps".to_string());
    }
    for (end, start) in TIMESTAMP_MONOTONIC_PAIRS {
        let (Some(&end_tick), Some(&start_tick)) = (ticks.get(end), ticks.get(start)) else {
            return Err(format!(
                "resolve returned {} timestamps, too few to compare query {end} against {start}",
                ticks.len()
            ));
        };
        if end_tick < start_tick {
            return Err(format!(
                "query {end} resolved {end_tick}, earlier than query {start} at {start_tick}"
            ));
        }
    }
    Ok(())
}

// Inline: the resolve verdict and the pair table are crate-private, so an
// integration test cannot name either one.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair the timed-dispatch layout compares, driven from the layout
    /// itself rather than a hand-listed set, so a query count change turns this
    /// red instead of leaving a pair unchecked.
    #[test]
    fn a_pair_that_runs_backwards_is_not_a_usable_resolve() {
        let count = crate::engine::record_and_readback::timestamp::TIMESTAMP_QUERY_COUNT as usize;
        let highest = TIMESTAMP_MONOTONIC_PAIRS
            .iter()
            .flat_map(|(end, start)| [*end, *start])
            .max()
            .expect("Fix: a timed dispatch must compare at least one pair");
        assert!(
            highest < count,
            "Fix: pair {highest} is compared but the resolve records only {count} queries"
        );
        let baseline: Vec<u64> = (0..count).map(|index| 1_000 + index as u64).collect();
        timestamp_ticks_are_usable(&baseline)
            .expect("Fix: an increasing resolve is the usable case");

        for (end, start) in TIMESTAMP_MONOTONIC_PAIRS {
            let mut ticks = baseline.clone();
            ticks[end] = baseline[start] - 1;
            let reason = timestamp_ticks_are_usable(&ticks)
                .expect_err("Fix: a pair whose end precedes its start is not a usable resolve");
            assert!(
                reason.contains(&format!("query {end} resolved")),
                "Fix: the reason must name the query that ran backwards, got {reason}"
            );
        }
    }

    #[test]
    fn a_zero_end_timestamp_is_reported_as_no_capability() {
        // The measured Metal resolve: a beginning-of-pass tick and a
        // zero end-of-pass tick, which reached a caller as a delta underflow.
        let reason = timestamp_ticks_are_usable(&[1_177_748_395_222_791, 0, 1, 2])
            .expect_err("Fix: a zero end-of-pass tick is not a usable resolve");
        assert!(
            reason.contains("query 1 resolved 0"),
            "Fix: the reason must name the zero tick, got {reason}"
        );
    }

    #[test]
    fn an_all_zero_resolve_is_not_a_measurement() {
        let reason = timestamp_ticks_are_usable(&[0, 0, 0, 0])
            .expect_err("Fix: an all-zero resolve reports a zero device time for every dispatch");
        assert!(
            reason.contains("only zero timestamps"),
            "Fix: the reason must say the resolve was all zero, got {reason}"
        );
    }

    #[test]
    fn a_short_resolve_names_the_pair_it_cannot_compare() {
        let highest = TIMESTAMP_MONOTONIC_PAIRS
            .iter()
            .flat_map(|(end, start)| [*end, *start])
            .max()
            .expect("Fix: a timed dispatch must compare at least one pair");
        let (end, start) = TIMESTAMP_MONOTONIC_PAIRS
            .into_iter()
            .find(|(end, start)| *end >= highest || *start >= highest)
            .expect("Fix: the highest compared index belongs to some pair");
        let ticks = vec![1u64; highest];
        let reason = timestamp_ticks_are_usable(&ticks)
            .expect_err("Fix: too few timestamps cannot prove a monotonic layout");
        assert!(
            reason.contains(&format!("too few to compare query {end} against {start}")),
            "Fix: the reason must name the pair it could not read, got {reason}"
        );
    }
}
