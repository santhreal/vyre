//! WHY: the parsed row becomes measurement provenance inside an artifact. A
//! misread field is worse than an absent one, because a reader cannot tell it
//! from a real measurement: a throttled device that parses as clear turns a
//! thermally limited launch into evidence about a schedule. Every field and every
//! throttle class is exercised here, including the absent forms the driver
//! reports for a device that does not expose a term.
//!
//! What this does not catch: whether the management interface itself reports the
//! truth. That is the driver's contract.

use vyre_megakernel::measure::ThrottleState;

use super::device_state::parse_device_state_row;

/// One row in the exact field order the query asks for.
fn row(graphics: &str, memory: &str, temperature: &str, power: &str, throttle: &str) -> String {
    format!("{graphics}, {memory}, {temperature}, {power}, {throttle}")
}

#[test]
fn every_field_is_scaled_to_its_recorded_unit() {
    let state = parse_device_state_row(&row("1980", "9501", "45", "118.44", "0x0000000000000000"))
        .expect("a complete row must parse");

    assert_eq!(
        state.graphics_clock_khz, 1_980_000,
        "megahertz to kilohertz"
    );
    assert_eq!(state.memory_clock_khz, 9_501_000);
    assert_eq!(
        state.temperature_millicelsius, 45_000,
        "degrees to thousandths"
    );
    assert_eq!(
        state.power_draw_milliwatts, 118_440,
        "watts to thousandths, keeping the reported fraction"
    );
    assert_eq!(state.throttle, ThrottleState::Clear);
    assert!(
        !state.is_unreported(),
        "a device that answered every term is not unreported"
    );
}

#[test]
fn an_absent_term_is_absent_and_the_rest_still_parses() {
    let state = parse_device_state_row(&row(
        "[N/A]",
        "9501",
        "[N/A]",
        "[N/A]",
        "0x0000000000000000",
    ))
    .expect("a row with absent terms must still parse");

    assert_eq!(state.graphics_clock_khz, 0, "an absent clock reports zero");
    assert_eq!(
        state.memory_clock_khz, 9_501_000,
        "the reported term stands"
    );
    assert_eq!(state.temperature_millicelsius, 0);
    assert_eq!(state.power_draw_milliwatts, 0);
}

#[test]
fn every_limiting_reason_reads_as_throttled_and_no_other_does() {
    // Bit positions the management interface documents for active clock reasons.
    for (mask, why, expected) in [
        (0x0_u64, "no reason active", ThrottleState::Clear),
        (0x1, "gpu idle", ThrottleState::Clear),
        (0x2, "applications clock setting", ThrottleState::Clear),
        (0x100, "display clock setting", ThrottleState::Clear),
        (0x4, "software power cap", ThrottleState::Throttled),
        (0x8, "hardware slowdown", ThrottleState::Throttled),
        (0x20, "software thermal slowdown", ThrottleState::Throttled),
        (0x40, "hardware thermal slowdown", ThrottleState::Throttled),
        (0x80, "hardware power brake", ThrottleState::Throttled),
        (
            0x21,
            "idle beside a thermal limit",
            ThrottleState::Throttled,
        ),
    ] {
        let state =
            parse_device_state_row(&row("1980", "9501", "45", "118.44", &format!("{mask:#x}")))
                .expect("a row with a reason mask must parse");
        assert_eq!(
            state.throttle, expected,
            "{why} ({mask:#x}) must read as {expected:?}"
        );
    }
}

#[test]
fn a_reason_field_the_driver_did_not_answer_is_unreported() {
    for field in ["[N/A]", "Not Supported", ""] {
        let state = parse_device_state_row(&row("1980", "9501", "45", "118.44", field))
            .expect("a row with an unreadable reason must still parse");
        assert_eq!(
            state.throttle,
            ThrottleState::Unreported,
            "`{field}` states nothing about the clock"
        );
    }
}

#[test]
fn a_row_that_is_not_the_queried_schema_is_refused() {
    for (why, text) in [
        ("one field short", "1980, 9501, 45, 118.44"),
        ("one field extra", "1980, 9501, 45, 118.44, 0x0, 7"),
        ("empty", ""),
    ] {
        assert!(
            parse_device_state_row(text).is_none(),
            "a row that is {why} must be refused rather than half-read: `{text}`"
        );
    }
}
