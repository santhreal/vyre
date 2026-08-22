//! Source fixtures the host oracle elimination tests share.
//!
//! Every case in this family needs a function body the gate must judge: a host
//! loop that derives bytes from input words. A case varies the item that wraps
//! that body, the name it gives it, and whether the item is test scoped. The
//! body itself is never what a case varies, so it is written here and each case
//! supplies the rest. Indentation is cosmetic to the parser, so one form serves
//! a free function, a method and a trait default alike.

/// A host oracle body that folds each input word into output bytes.
///
/// `op` is the arithmetic applied to each word, so a case that must be
/// distinguishable from another case states its own operation.
pub(super) fn oracle_body(op: &str) -> String {
    format!(
        "    let mut out = Vec::new();
    for &x in input {{
        out.extend_from_slice(&x.{op}.to_le_bytes());
    }}
    out"
    )
}

/// A host oracle body that adds one to each input word.
pub(super) fn incrementing_oracle_body() -> String {
    oracle_body("wrapping_add(1)")
}
