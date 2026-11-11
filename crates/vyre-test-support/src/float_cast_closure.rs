//! The refusal every emitter owes for a float cast with no defined conversion.
//!
//! `cast_is_valid` in the foundation rejects a cast from `f32` to a target
//! that has no numeric conversion. An emitter reached through a no-validation
//! path never sees that decision, so each one has to fail closed on its own:
//! a narrow integer target would otherwise take a conversion that does not
//! narrow and produce a full-range word claimed as a byte, and a wide target
//! would reinterpret the float through a 32-bit coerce that drops the high
//! word. Both miscompile silently.
//!
//! The refusal is one contract across every emitter, and it was written out
//! once per emitter suite. A copy is a place where the expected message can be
//! relaxed to match one target's regression while every other target still
//! reports the strict form, which is the failure this module removes.
//!
//! The target list stays per emitter: which targets have no defined
//! conversion is a property of what that emitter can express, not of the
//! refusal.

use core::fmt::Debug;

use vyre_foundation::ir::DataType;

/// Assert that emitting a cast from `f32` to `target` was refused, with the
/// message that names the cast and states a corrective action.
///
/// `emitted` is whatever the emitter returned. A successful emit is the defect
/// this asserts against, so the `Ok` value is only ever reported in the
/// failure.
///
/// # Panics
/// Panics when the emitter produced a module, or refused with a message that
/// does not name the cast, the absent conversion, or a fix.
pub fn float_cast_must_fail_closed<T: Debug, E: Debug>(target: &DataType, emitted: Result<T, E>) {
    let error = match emitted {
        Err(error) => error,
        Ok(emitted) => panic!(
            "Fix: f32 -> {target:?} has no defined float conversion, so emit must refuse it \
             rather than produce {emitted:?}"
        ),
    };
    let message = format!("{error:?}");
    assert!(
        message.contains("cast from f32 to")
            && message.contains("no defined conversion")
            && message.contains("Fix:"),
        "Fix: the refusal of f32 -> {target:?} must name the cast, state that no conversion is \
         defined, and give a corrective action; got: {message}"
    );
}
