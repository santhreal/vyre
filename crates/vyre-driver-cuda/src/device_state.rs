//! Live clock, thermal and power state for one CUDA device.
//!
//! The CUDA driver API reports peak clocks as device attributes and reports no
//! current clock, temperature or power draw at all, so the live figures come from
//! the management interface the driver installs beside it. Compile-time
//! measurement retains them next to its samples: a candidate that measured slow
//! on a thermally limited device measured the device, not the schedule.

use std::process::Command;

use vyre_megakernel::measure::{DeviceState, ThrottleState};

/// Fields queried, in the order [`parse_device_state_row`] reads them.
const QUERY_FIELDS: &str = "clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,clocks_throttle_reasons.active";

/// Throttle reasons that mean the device is running below the clock it was
/// asked for. Idle and application-clock settings are excluded: neither limits a
/// launch that is running.
const LIMITING_REASONS: u64 = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;

/// Query live state for the device at `ordinal`.
///
/// # Errors
///
/// Returns an error when the management interface cannot be run, exits nonzero,
/// or reports a row this parser does not recognize.
pub(crate) fn query_device_state(ordinal: usize) -> Result<DeviceState, String> {
    let output = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={QUERY_FIELDS}"),
            "--format=csv,noheader,nounits",
            "-i",
            &ordinal.to_string(),
        ])
        .output()
        .map_err(|error| {
            format!(
                "CUDA device-state query failed for ordinal {ordinal}: {error}. Fix: make `nvidia-smi` reachable on a GPU host so compile-time measurement retains clock and power state."
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "CUDA device-state query exited with status {} for ordinal {ordinal}: {}. Fix: repair NVIDIA driver visibility on this host.",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        format!("CUDA device-state query output for ordinal {ordinal} was not UTF-8: {error}")
    })?;
    let row = stdout.lines().next().ok_or_else(|| {
        format!(
            "CUDA device-state query reported no row for ordinal {ordinal}. Fix: verify the ordinal is visible to `nvidia-smi -L`."
        )
    })?;
    parse_device_state_row(row).ok_or_else(|| {
        format!(
            "CUDA device-state query for ordinal {ordinal} reported an unreadable row `{}`. Fix: update the parser to the queried field schema `{QUERY_FIELDS}`.",
            row.trim()
        )
    })
}

/// State the device reports, or [`DeviceState::unreported`] with a warning when
/// the management interface cannot answer.
///
/// A CUDA device dispatches without a management interface, so an unanswerable
/// query is not a compilation failure: the samples stay valid and the session
/// records the drift it observes across its own rounds instead.
#[must_use]
pub(crate) fn device_state(ordinal: usize) -> DeviceState {
    match query_device_state(ordinal) {
        Ok(state) => state,
        Err(error) => {
            tracing::warn!("{error}");
            DeviceState::unreported()
        }
    }
}

/// Parse one row of the query above.
///
/// Clocks arrive in megahertz, temperature in whole degrees Celsius and power in
/// watts; the recorded state is in kilohertz, thousandths of a degree and
/// thousandths of a watt. A field the driver reports as `[N/A]` parses as
/// unreported for that term alone.
#[must_use]
pub(crate) fn parse_device_state_row(row: &str) -> Option<DeviceState> {
    let mut fields = row.split(',').map(str::trim);
    let graphics_mhz = fields.next()?;
    let memory_mhz = fields.next()?;
    let temperature_c = fields.next()?;
    let power_w = fields.next()?;
    let throttle = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    Some(DeviceState {
        graphics_clock_khz: scaled_u32(graphics_mhz, 1_000),
        memory_clock_khz: scaled_u32(memory_mhz, 1_000),
        temperature_millicelsius: temperature_c
            .parse::<i32>()
            .map_or(0, |celsius| celsius.saturating_mul(1_000)),
        power_draw_milliwatts: power_w
            .parse::<f64>()
            .ok()
            .filter(|watts| watts.is_finite() && *watts >= 0.0)
            .map_or(0, |watts| (watts * 1_000.0) as u32),
        throttle: parse_throttle(throttle),
    })
}

/// A whole-number field scaled to its recorded unit, or zero when the driver
/// reports nothing for it.
fn scaled_u32(field: &str, scale: u32) -> u32 {
    field
        .parse::<u32>()
        .map_or(0, |value| value.saturating_mul(scale))
}

/// Read the active-reason bitmask the driver reports.
fn parse_throttle(field: &str) -> ThrottleState {
    let digits = field.strip_prefix("0x").unwrap_or(field);
    let Ok(mask) = u64::from_str_radix(digits, 16) else {
        return ThrottleState::Unreported;
    };
    if mask & LIMITING_REASONS == 0 {
        ThrottleState::Clear
    } else {
        ThrottleState::Throttled
    }
}
