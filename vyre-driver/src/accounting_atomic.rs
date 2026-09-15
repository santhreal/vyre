//! Backend-neutral atomic accounting primitives.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::BackendError;

/// Add `value` to a `u64` counter without allowing wraparound or saturation.
///
/// # Errors
///
/// Returns [`BackendError`] from `overflow` when the addition would overflow.
pub fn checked_atomic_add_u64(
    counter: &AtomicU64,
    value: u64,
    overflow: impl Fn(u64, u64) -> BackendError,
) -> Result<(), BackendError> {
    checked_atomic_add_u64_with_order(
        counter,
        value,
        Ordering::Relaxed,
        Ordering::Relaxed,
        Ordering::Relaxed,
        overflow,
    )
}

/// Add `value` to a `u64` counter with caller-selected atomic orderings.
///
/// # Errors
///
/// Returns `E` from `overflow` when the addition would overflow.
pub fn checked_atomic_add_u64_with_order<E>(
    counter: &AtomicU64,
    value: u64,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    overflow: impl Fn(u64, u64) -> E,
) -> Result<(), E> {
    checked_atomic_add_u64_guarded_with_order(
        counter,
        value,
        load_order,
        success_order,
        failure_order,
        overflow,
        |_| Ok(()),
    )
}

macro_rules! define_checked_atomic_add_guarded {
    ($name:ident, $atomic:ty, $value:ty) => {
        #[doc = concat!(
                                    "Add a `",
                                    stringify!($value),
                                    "` value with overflow checking and a pre-CAS next-value guard."
                                )]
        ///
        /// # Errors
        ///
        /// Returns `E` from `overflow` when the addition would overflow, or from
        /// `validate_next` when the computed next value violates a caller invariant.
        pub fn $name<E>(
            counter: &$atomic,
            value: $value,
            load_order: Ordering,
            success_order: Ordering,
            failure_order: Ordering,
            overflow: impl Fn($value, $value) -> E,
            mut validate_next: impl FnMut($value) -> Result<(), E>,
        ) -> Result<(), E> {
            let mut observed = counter.load(load_order);
            loop {
                let next = observed
                    .checked_add(value)
                    .ok_or_else(|| overflow(observed, value))?;
                validate_next(next)?;
                match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
                    Ok(_) => return Ok(()),
                    Err(actual) => observed = actual,
                }
            }
        }
    };
}

define_checked_atomic_add_guarded!(checked_atomic_add_u64_guarded_with_order, AtomicU64, u64);

/// Add `value` to a `usize` counter without allowing wraparound.
///
/// # Errors
///
/// Returns [`BackendError`] from `overflow` when the addition would overflow.
pub fn checked_atomic_add_usize(
    counter: &AtomicUsize,
    value: usize,
    overflow: impl Fn(usize, usize) -> BackendError,
) -> Result<(), BackendError> {
    checked_atomic_add_usize_with_order(
        counter,
        value,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        overflow,
    )
}

/// Add `value` to a `usize` counter with caller-selected atomic orderings.
///
/// # Errors
///
/// Returns `E` from `overflow` when the addition would overflow.
pub fn checked_atomic_add_usize_with_order<E>(
    counter: &AtomicUsize,
    value: usize,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    overflow: impl Fn(usize, usize) -> E,
) -> Result<(), E> {
    checked_atomic_add_usize_guarded_with_order(
        counter,
        value,
        load_order,
        success_order,
        failure_order,
        overflow,
        |_| Ok(()),
    )
}

define_checked_atomic_add_guarded!(
    checked_atomic_add_usize_guarded_with_order,
    AtomicUsize,
    usize
);

/// Subtract `value` from a `u64` counter without allowing underflow.
///
/// # Errors
///
/// Returns [`BackendError`] from `underflow` when the subtraction would underflow.
pub fn checked_atomic_sub_u64(
    counter: &AtomicU64,
    value: u64,
    underflow: impl Fn(u64, u64) -> BackendError,
) -> Result<(), BackendError> {
    checked_atomic_sub_u64_with_order(
        counter,
        value,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        underflow,
    )
}

macro_rules! define_checked_atomic_sub {
    ($name:ident, $atomic:ty, $value:ty) => {
        #[doc = concat!(
                                            "Subtract from a `",
                                            stringify!($value),
                                            "` atomic counter with caller-selected orderings."
                                        )]
        ///
        /// # Errors
        ///
        /// Returns `E` from `underflow` when subtraction would underflow.
        pub fn $name<E>(
            counter: &$atomic,
            value: $value,
            load_order: Ordering,
            success_order: Ordering,
            failure_order: Ordering,
            underflow: impl Fn($value, $value) -> E,
        ) -> Result<(), E> {
            if value == 0 {
                return Ok(());
            }
            let mut observed = counter.load(load_order);
            loop {
                let next = observed
                    .checked_sub(value)
                    .ok_or_else(|| underflow(observed, value))?;
                match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
                    Ok(_) => return Ok(()),
                    Err(actual) => observed = actual,
                }
            }
        }
    };
}

define_checked_atomic_sub!(checked_atomic_sub_u64_with_order, AtomicU64, u64);

/// Subtract `value` from a `usize` counter without allowing underflow.
///
/// # Errors
///
/// Returns [`BackendError`] from `underflow` when the subtraction would underflow.
pub fn checked_atomic_sub_usize(
    counter: &AtomicUsize,
    value: usize,
    underflow: impl Fn(usize, usize) -> BackendError,
) -> Result<(), BackendError> {
    checked_atomic_sub_usize_with_order(
        counter,
        value,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        underflow,
    )
}

define_checked_atomic_sub!(checked_atomic_sub_usize_with_order, AtomicUsize, usize);

macro_rules! define_checked_atomic_update {
    ($name:ident, $atomic:ty, $value:ty) => {
        #[doc = concat!(
                                            "Apply a checked update to a `",
                                            stringify!($value),
                                            "` atomic counter with caller-selected orderings."
                                        )]
        ///
        /// Returns the value observed before the successful publish. `update`
        /// computes the next value, and `on_retry` may abort a failed CAS retry.
        pub fn $name<E>(
            counter: &$atomic,
            load_order: Ordering,
            success_order: Ordering,
            failure_order: Ordering,
            mut update: impl FnMut($value) -> Result<$value, E>,
            mut on_retry: impl FnMut($value, $value) -> Result<(), E>,
        ) -> Result<$value, E> {
            let mut observed = counter.load(load_order);
            loop {
                let next = update(observed)?;
                match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
                    Ok(previous) => return Ok(previous),
                    Err(actual) => {
                        on_retry(observed, actual)?;
                        observed = actual;
                    }
                }
            }
        }
    };
}

define_checked_atomic_update!(checked_atomic_update_u64_with_order, AtomicU64, u64);
define_checked_atomic_update!(checked_atomic_update_u32_with_order, AtomicU32, u32);

/// Subtract `value` from a `usize` counter, repairing underflow to zero.
///
/// This is only for release-path accounting where the caller has already
/// decided that a corrupt counter must be repaired rather than propagated as a
/// dispatch error. `on_repair` is called after a successful repair CAS.
pub fn repair_atomic_sub_usize_with_order(
    counter: &AtomicUsize,
    value: usize,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    on_repair: impl FnMut(usize, usize),
) {
    let _ = repair_atomic_sub_usize_fetch_with_order(
        counter,
        value,
        load_order,
        success_order,
        failure_order,
        on_repair,
    );
}

/// Subtract `value` from a `usize` counter, repairing underflow to zero and
/// returning the observed value before the successful publish.
///
/// This preserves `fetch_sub`-style previous-value semantics for padded or
/// wrapper counters while keeping the underflow repair policy single-sourced.
pub fn repair_atomic_sub_usize_fetch_with_order(
    counter: &AtomicUsize,
    value: usize,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    mut on_repair: impl FnMut(usize, usize),
) -> usize {
    if value == 0 {
        return counter.load(load_order);
    }
    let mut observed = counter.load(load_order);
    loop {
        let Some(next) = observed.checked_sub(value) else {
            match counter.compare_exchange_weak(observed, 0, success_order, failure_order) {
                Ok(_) => {
                    on_repair(observed, value);
                    return observed;
                }
                Err(actual) => {
                    observed = actual;
                    continue;
                }
            }
        };
        match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
            Ok(_) => return observed,
            Err(actual) => observed = actual,
        }
    }
}

/// Add `value` to a `usize` atomic counter, pinning it at `usize::MAX` instead
/// of wrapping, and return the observed value before the successful publish.
///
/// `on_pinned` is called exactly once when a successful publish moves a
/// non-pinned counter to `usize::MAX`.
pub fn pinning_atomic_add_usize_with_order(
    counter: &AtomicUsize,
    value: usize,
    success_order: Ordering,
    failure_order: Ordering,
    on_pinned: impl FnOnce(usize, usize),
) -> usize {
    if value == 0 {
        return counter.load(failure_order);
    }
    let mut current = counter.load(failure_order);
    loop {
        let next = current.checked_add(value).unwrap_or(usize::MAX);
        match counter.compare_exchange_weak(current, next, success_order, failure_order) {
            Ok(previous) => {
                if next == usize::MAX && previous != usize::MAX {
                    on_pinned(previous, value);
                }
                return previous;
            }
            Err(observed) => current = observed,
        }
    }
}

macro_rules! define_pinning_atomic_increment {
    ($name:ident, $atomic:ty, $value:ty) => {
        #[doc = concat!(
                                                            "Increment a `",
                                                            stringify!($value),
                                                            "` atomic counter, pinning it at `",
                                                            stringify!($value),
                                                            "::MAX` instead of wrapping."
                                                        )]
        ///
        /// Returns `true` when the counter was incremented and `false` when it was
        /// already pinned. `on_pinned` is called exactly once on the pinned path.
        pub fn $name(
            counter: &$atomic,
            success_order: Ordering,
            failure_order: Ordering,
            on_pinned: impl FnOnce(),
        ) -> bool {
            let mut current = counter.load(failure_order);
            loop {
                let Some(next) = current.checked_add(1) else {
                    on_pinned();
                    return false;
                };
                match counter.compare_exchange_weak(current, next, success_order, failure_order) {
                    Ok(_) => return true,
                    Err(observed) => current = observed,
                }
            }
        }
    };
}

define_pinning_atomic_increment!(pinning_atomic_increment_u64, AtomicU64, u64);
define_pinning_atomic_increment!(pinning_atomic_increment_u32, AtomicU32, u32);

/// Allocate the current `u64` atomic sequence value and publish the next value.
///
/// When incrementing would overflow, publishes `rebase_to` instead of wrapping.
/// Returns the allocated value observed before the publish. `on_rebase` is
/// called exactly once for each successful overflow rebase.
pub fn rebasing_atomic_next_u64(
    counter: &AtomicU64,
    rebase_to: u64,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    mut on_rebase: impl FnMut(u64, u64),
) -> u64 {
    let mut observed = counter.load(load_order);
    loop {
        let next = match observed.checked_add(1) {
            Some(next) => next,
            None => rebase_to,
        };
        match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
            Ok(_) => {
                if next == rebase_to && observed == u64::MAX {
                    on_rebase(observed, rebase_to);
                }
                return observed;
            }
            Err(actual) => observed = actual,
        }
    }
}

/// Allocate the current `u64` atomic sequence value and publish `current + 1`.
///
/// # Errors
///
/// Returns `E` from `overflow` when the sequence cannot advance without
/// wrapping.
pub fn checked_atomic_next_u64_with_order<E>(
    counter: &AtomicU64,
    load_order: Ordering,
    success_order: Ordering,
    failure_order: Ordering,
    overflow: impl Fn(u64) -> E,
) -> Result<u64, E> {
    let mut observed = counter.load(load_order);
    loop {
        let next = observed.checked_add(1).ok_or_else(|| overflow(observed))?;
        match counter.compare_exchange_weak(observed, next, success_order, failure_order) {
            Ok(_) => return Ok(observed),
            Err(actual) => observed = actual,
        }
    }
}

/// Raise a `u64` atomic counter to at least `value` using one atomic max update.
///
/// Returns the previous value observed by the atomic operation.
pub fn atomic_max_u64(counter: &AtomicU64, value: u64, order: Ordering) -> u64 {
    counter.fetch_max(value, order)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn checked_atomic_accounting_reports_overflow_and_underflow_without_saturation() {
        let add_counter = AtomicU64::new(u64::MAX - 1);
        checked_atomic_add_u64(&add_counter, 1, |_, _| unreachable!("one fits"))
            .expect("Fix: atomic add should accept exact non-overflow");
        assert_eq!(add_counter.load(Ordering::Relaxed), u64::MAX);
        let add_error = checked_atomic_add_u64(&add_counter, 1, |current, attempted| {
            crate::BackendError::InvalidProgram {
                fix: format!("Fix: overflow {current} {attempted}"),
            }
        })
        .expect_err("overflowing atomic add should fail");
        assert!(add_error.to_string().contains("overflow"));
        assert_eq!(add_counter.load(Ordering::Relaxed), u64::MAX);

        let sub_counter = AtomicU64::new(1);
        checked_atomic_sub_u64(&sub_counter, 1, |_, _| unreachable!("one fits"))
            .expect("Fix: atomic sub should accept exact subtraction");
        assert_eq!(sub_counter.load(Ordering::Acquire), 0);
        let sub_error = checked_atomic_sub_u64(&sub_counter, 1, |current, attempted| {
            crate::BackendError::InvalidProgram {
                fix: format!("Fix: underflow {current} {attempted}"),
            }
        })
        .expect_err("underflowing atomic sub should fail");
        assert!(sub_error.to_string().contains("underflow"));
        assert_eq!(sub_counter.load(Ordering::Acquire), 0);

        let usize_add_counter = AtomicUsize::new(usize::MAX - 1);
        checked_atomic_add_usize(&usize_add_counter, 1, |_, _| unreachable!("one fits"))
            .expect("Fix: usize atomic add should accept exact non-overflow");
        assert_eq!(usize_add_counter.load(Ordering::Acquire), usize::MAX);
        let usize_add_error =
            checked_atomic_add_usize(&usize_add_counter, 1, |current, attempted| {
                crate::BackendError::InvalidProgram {
                    fix: format!("Fix: usize overflow {current} {attempted}"),
                }
            })
            .expect_err("overflowing usize atomic add should fail");
        assert!(usize_add_error.to_string().contains("usize overflow"));
        assert_eq!(usize_add_counter.load(Ordering::Acquire), usize::MAX);

        let usize_counter = AtomicUsize::new(0);
        let usize_error = checked_atomic_sub_usize(&usize_counter, 1, |current, attempted| {
            crate::BackendError::InvalidProgram {
                fix: format!("Fix: usize underflow {current} {attempted}"),
            }
        })
        .expect_err("underflowing usize atomic sub should fail");
        assert!(usize_error.to_string().contains("usize underflow"));
        assert_eq!(usize_counter.load(Ordering::Acquire), 0);
    }

    #[test]
    fn ordered_atomic_helpers_preserve_domain_errors() {
        let add_counter = AtomicU64::new(40);
        checked_atomic_add_u64_with_order(
            &add_counter,
            2,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "overflow",
        )
        .expect("Fix: reject adds that would overflow; use checked accounting API on hostile sizes - ordered atomic add should accept non-overflow");
        assert_eq!(add_counter.load(Ordering::Acquire), 42);

        let sub_counter = AtomicU64::new(42);
        checked_atomic_sub_u64_with_order(
            &sub_counter,
            2,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "underflow",
        )
        .expect("Fix: reject subs that would underflow; use checked accounting API on hostile sizes - ordered atomic sub should accept non-underflow");
        assert_eq!(sub_counter.load(Ordering::Acquire), 40);

        let usize_counter = AtomicUsize::new(10);
        checked_atomic_add_usize_with_order(
            &usize_counter,
            5,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "usize overflow",
        )
        .expect("Fix: reject usize atomics that overflow/underflow; return Err from guarded helpers - ordered usize atomic add should accept non-overflow");
        assert_eq!(usize_counter.load(Ordering::Acquire), 15);
        checked_atomic_sub_usize_with_order(
            &usize_counter,
            3,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "usize underflow",
        )
        .expect("Fix: reject usize atomics that overflow/underflow; return Err from guarded helpers - ordered usize atomic sub should accept non-underflow");
        assert_eq!(usize_counter.load(Ordering::Acquire), 12);
    }

    #[test]
    fn guarded_atomic_add_helpers_validate_next_value_before_publish() {
        let u64_counter = AtomicU64::new(8);
        let u64_error = checked_atomic_add_u64_guarded_with_order(
            &u64_counter,
            5,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "overflow",
            |next| {
                if next <= 12 {
                    Ok(())
                } else {
                    Err("budget")
                }
            },
        )
        .expect_err("guarded u64 add should reject over-budget next value");
        assert_eq!(u64_error, "budget");
        assert_eq!(u64_counter.load(Ordering::Acquire), 8);

        checked_atomic_add_u64_guarded_with_order(
            &u64_counter,
            4,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "overflow",
            |next| if next <= 12 { Ok(()) } else { Err("budget") },
        )
        .expect("Fix: reject guarded adds that overflow; surface Err to caller instead of panicking - guarded u64 add should publish accepted next value");
        assert_eq!(u64_counter.load(Ordering::Acquire), 12);

        let usize_counter = AtomicUsize::new(3);
        let usize_error = checked_atomic_add_usize_guarded_with_order(
            &usize_counter,
            2,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| "overflow",
            |next| {
                if next < 5 {
                    Ok(())
                } else {
                    Err("usize budget")
                }
            },
        )
        .expect_err("guarded usize add should reject over-budget next value");
        assert_eq!(usize_error, "usize budget");
        assert_eq!(usize_counter.load(Ordering::Acquire), 3);
    }

    #[test]
    fn pinning_atomic_increment_helpers_never_wrap() {
        let u64_counter = AtomicU64::new(u64::MAX - 1);
        assert!(pinning_atomic_increment_u64(
            &u64_counter,
            Ordering::Relaxed,
            Ordering::Relaxed,
            || unreachable!("first increment should fit")
        ));
        assert_eq!(u64_counter.load(Ordering::Relaxed), u64::MAX);
        let mut u64_pinned = false;
        assert!(!pinning_atomic_increment_u64(
            &u64_counter,
            Ordering::Relaxed,
            Ordering::Relaxed,
            || u64_pinned = true
        ));
        assert!(u64_pinned);
        assert_eq!(u64_counter.load(Ordering::Relaxed), u64::MAX);

        let u32_counter = AtomicU32::new(u32::MAX - 1);
        assert!(pinning_atomic_increment_u32(
            &u32_counter,
            Ordering::Relaxed,
            Ordering::Relaxed,
            || unreachable!("first increment should fit")
        ));
        assert_eq!(u32_counter.load(Ordering::Relaxed), u32::MAX);
        let mut u32_pinned = false;
        assert!(!pinning_atomic_increment_u32(
            &u32_counter,
            Ordering::Relaxed,
            Ordering::Relaxed,
            || u32_pinned = true
        ));
        assert!(u32_pinned);
        assert_eq!(u32_counter.load(Ordering::Relaxed), u32::MAX);
    }

    #[test]
    fn atomic_max_helper_raises_without_lowering() {
        let counter = AtomicU64::new(10);
        assert_eq!(atomic_max_u64(&counter, 42, Ordering::Relaxed), 10);
        assert_eq!(counter.load(Ordering::Relaxed), 42);
        assert_eq!(atomic_max_u64(&counter, 7, Ordering::Relaxed), 42);
        assert_eq!(counter.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn rebasing_atomic_next_returns_observed_and_rebases_on_overflow() {
        let counter = AtomicU64::new(7);
        let mut rebase_count = 0;
        assert_eq!(
            rebasing_atomic_next_u64(
                &counter,
                1,
                Ordering::Acquire,
                Ordering::AcqRel,
                Ordering::Acquire,
                |_, _| rebase_count += 1,
            ),
            7
        );
        assert_eq!(counter.load(Ordering::Acquire), 8);
        assert_eq!(rebase_count, 0);

        counter.store(u64::MAX, Ordering::Release);
        assert_eq!(
            rebasing_atomic_next_u64(
                &counter,
                1,
                Ordering::Acquire,
                Ordering::AcqRel,
                Ordering::Acquire,
                |observed, rebase_to| {
                    assert_eq!(observed, u64::MAX);
                    assert_eq!(rebase_to, 1);
                    rebase_count += 1;
                },
            ),
            u64::MAX
        );
        assert_eq!(counter.load(Ordering::Acquire), 1);
        assert_eq!(rebase_count, 1);
    }

    #[test]
    fn checked_atomic_next_returns_observed_and_rejects_wraparound() {
        let counter = AtomicU64::new(41);
        assert_eq!(
            checked_atomic_next_u64_with_order(
                &counter,
                Ordering::Acquire,
                Ordering::AcqRel,
                Ordering::Acquire,
                |_| "overflow",
            )
            .expect("Fix: allocation of next atomic value must not overflow; return None/Err on hostile input - checked atomic next should allocate non-overflowing value"),
            41
        );
        assert_eq!(counter.load(Ordering::Acquire), 42);

        counter.store(u64::MAX, Ordering::Release);
        let error = checked_atomic_next_u64_with_order(
            &counter,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |observed| {
                assert_eq!(observed, u64::MAX);
                "overflow"
            },
        )
        .expect_err("checked atomic next should reject u64 wraparound");
        assert_eq!(error, "overflow");
        assert_eq!(counter.load(Ordering::Acquire), u64::MAX);
    }

    #[test]
    fn repair_atomic_sub_usize_repairs_underflow_to_zero_once() {
        let counter = AtomicUsize::new(10);
        let mut repairs = 0;
        repair_atomic_sub_usize_with_order(
            &counter,
            4,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_, _| repairs += 1,
        );
        assert_eq!(counter.load(Ordering::Acquire), 6);
        assert_eq!(repairs, 0);

        repair_atomic_sub_usize_with_order(
            &counter,
            99,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |observed, attempted| {
                assert_eq!(observed, 6);
                assert_eq!(attempted, 99);
                repairs += 1;
            },
        );
        assert_eq!(counter.load(Ordering::Acquire), 0);
        assert_eq!(repairs, 1);
    }
}
