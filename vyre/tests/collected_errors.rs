//! The result a roster check returns after it has looked at every member.
//!
//! Each check in this suite walks the whole workspace before it answers, so a
//! caller sees every violation at once instead of the first one and then a
//! rerun. Five checks ended the same way, and a copy of a return shape is where
//! one of them starts returning `Ok` on a non-empty list.

/// `Ok` when nothing was collected, every message otherwise.
pub(crate) fn collected(errors: Vec<String>) -> Result<(), Vec<String>> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
