//! Rendering an unwind payload as a diagnostic string.
//!
//! Every proof lane that catches an unwind reports the same thing: what the
//! panicking arm said, or that it said nothing a reader can print. A payload is
//! `&'static str` or `String` for a `panic!`, and anything at all for a
//! non-string panic, so the third case is a stated fact rather than a lost
//! message.

/// The message `payload` carries, or a note that it carries none.
#[must_use]
pub fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
