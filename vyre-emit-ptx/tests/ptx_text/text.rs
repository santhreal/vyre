//! Readers for the parts of emitted PTX a shared-memory contract is stated
//! against.
//!
//! Both the zero-init pin and the extent-padding property read the same two
//! constructs out of the emitted text: the `.shared` declarations, and the
//! entry prologue that zeroes them. The two spellings drifted apart already -
//! one parsed a byte length with a bare `expect("byte length")` and the other
//! with an actionable message - so the reader is stated once and the contracts
//! disagree about the emitter rather than about how to read it.
//!
//! A reader returns what it found and asserts nothing. What a declaration or a
//! prologue has to contain is the contract's business, not the parser's.

/// Every `.shared .align 4 .b8 <symbol>[<bytes>];` declaration, in emit order.
///
/// A line the emitter writes in any other shape is not a shared declaration
/// and is skipped, so a declaration whose alignment or element width changes
/// reads as absent and turns the contract red rather than passing on a line it
/// did not understand.
pub(crate) fn shared_declarations(ptx: &str) -> Vec<(String, u32)> {
    ptx.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix(".shared .align 4 .b8 ")?;
            let (symbol, rest) = rest.split_once('[')?;
            let bytes = rest.strip_suffix("];")?;
            Some((
                symbol.to_string(),
                bytes
                    .parse()
                    .expect("Fix: emit a decimal byte length in a shared declaration"),
            ))
        })
        .collect()
}

/// Text from the zero prologue up to the barrier that closes it, or `None`
/// when the emitter wrote no prologue.
pub(crate) fn zero_prologue(ptx: &str) -> Option<&str> {
    let start = ptx.find("    // Workgroup memory holds zero at entry.\n")?;
    let tail = &ptx[start..];
    let end = tail.find("    bar.sync 0;\n")?;
    Some(&tail[..end])
}
