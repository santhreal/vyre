//! Binding-name to PTX identifier mangling.
//!
//! Owns the one rule that turns a descriptor binding name into the `.param`
//! identifier suffix the entry point declares and the body loads from. It
//! owns no other naming: register spelling lives in `reg` and label
//! allocation in the emission state.

/// Sanitize a binding name into a valid PTX identifier suffix. Empty
/// names fall back to `slot{N}` so every binding still gets a unique
/// suffix.
pub(super) fn sanitize_param_name(name: &str, slot: u32) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        format!("slot{slot}")
    } else {
        cleaned
    }
}
