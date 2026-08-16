//! Contracts for `vyre_driver::accounting`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::accounting::{
    checked_add_u64_usize_offset_lazy, checked_mul_u32_value, checked_mul_u64_lazy,
    checked_sub_u64_lazy, checked_sub_usize_lazy, checked_usize_byte_range_end_lazy,
    checked_usize_to_u64_lazy,
};

use std::cell::Cell;

use vyre_driver::accounting::{
    checked_add_u64_usize_offset_lazy, checked_mul_u32_value, checked_mul_u64_lazy,
    checked_sub_u64_lazy, checked_sub_usize_lazy, checked_usize_byte_range_end_lazy,
    checked_usize_to_u64_lazy,
};

#[test]
fn checked_mul_u64_lazy_is_lazy_on_success() {
    let overflow_called = Cell::new(false);

    let value = checked_mul_u64_lazy(8, 4, || {
        overflow_called.set(true);
        "overflow"
    });

    assert_eq!(value, Ok(32));
    assert!(!overflow_called.get());
}

#[test]
fn checked_mul_u64_lazy_reports_overflow() {
    let value = checked_mul_u64_lazy(u64::MAX, 2, || "overflow");

    assert_eq!(value, Err("overflow"));
}

#[test]
fn checked_mul_u32_value_multiplies_without_wraparound() {
    let value = checked_mul_u32_value(128, 8, "overflow");

    assert_eq!(value, Ok(1024));
}

#[test]
fn checked_mul_u32_value_reports_overflow() {
    let value = checked_mul_u32_value(u32::MAX, 2, "overflow");

    assert_eq!(value, Err("overflow"));
}

#[test]
fn checked_sub_u64_lazy_reports_underflow() {
    let value = checked_sub_u64_lazy(1, 2, || "underflow");

    assert_eq!(value, Err("underflow"));
}

#[test]
fn checked_sub_usize_lazy_reports_underflow() {
    let value = checked_sub_usize_lazy(4, 8, || "underflow");

    assert_eq!(value, Err("underflow"));
}

#[test]
fn checked_usize_to_u64_lazy_converts_host_width() {
    let value = checked_usize_to_u64_lazy(64, || "overflow");

    assert_eq!(value, Ok(64));
}

#[test]
fn checked_usize_byte_range_end_lazy_is_lazy_on_success() {
    let overflow_called = Cell::new(false);
    let bounds_called = Cell::new(false);

    let end = checked_usize_byte_range_end_lazy(
        8,
        4,
        16,
        || {
            overflow_called.set(true);
            "overflow"
        },
        |_| {
            bounds_called.set(true);
            "bounds"
        },
    );

    assert_eq!(end, Ok(12));
    assert!(!overflow_called.get());
    assert!(!bounds_called.get());
}

#[test]
fn checked_usize_byte_range_end_lazy_passes_computed_end_to_bounds_error() {
    let end = checked_usize_byte_range_end_lazy(8, 5, 12, || usize::MAX, |end| end);

    assert_eq!(end, Err(13));
}

#[test]
fn checked_add_u64_usize_offset_lazy_is_lazy_on_success() {
    let conversion_called = Cell::new(false);
    let overflow_called = Cell::new(false);

    let value = checked_add_u64_usize_offset_lazy(
        64,
        8,
        || {
            conversion_called.set(true);
            "conversion"
        },
        || {
            overflow_called.set(true);
            "overflow"
        },
    );

    assert_eq!(value, Ok(72));
    assert!(!conversion_called.get());
    assert!(!overflow_called.get());
}

#[test]
fn checked_add_u64_usize_offset_lazy_reports_pointer_overflow() {
    let value = checked_add_u64_usize_offset_lazy(u64::MAX, 1, || "conversion", || "overflow");

    assert_eq!(value, Err("overflow"));
}
