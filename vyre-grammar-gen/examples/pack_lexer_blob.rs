//! Build the C11 lexer DFA, pack it into a binary blob, and decode it back.
//!
//! `PackedBlob` is the wire format that `vyre-libs::parsing` uploads as a
//! ReadOnly storage buffer, so this example shows the exact bytes the GPU sees
//! and proves the round trip is lossless. The blob carries its own header
//! (dimensions plus a BLAKE3-128 payload digest), which is why decoding can
//! fail loudly instead of handing the GPU a truncated table.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example pack_lexer_blob -p vyre-grammar-gen
//! ```

use vyre_grammar_gen::{build_c11_lexer_dfa, decode_dfa_from_bytes, BlobKind, PackedBlob};

fn main() {
    let dfa = build_c11_lexer_dfa();
    println!(
        "C11 lexer DFA: {} states x {} classes, {} transition words",
        dfa.num_states,
        dfa.num_classes,
        dfa.transitions.len()
    );

    let accepting = dfa.token_ids.iter().filter(|&&id| id != 0).count();
    println!("{accepting} accepting states");

    let blob = PackedBlob::from_dfa(&dfa);
    assert_eq!(blob.kind, BlobKind::LexerDfa);
    println!("packed blob: {} bytes", blob.bytes.len());

    let decoded = match decode_dfa_from_bytes(&blob.bytes) {
        Ok(decoded) => decoded,
        Err(error) => {
            eprintln!("decoding the blob failed: {error:?}");
            std::process::exit(1);
        }
    };

    assert_eq!(decoded.num_states, dfa.num_states);
    assert_eq!(decoded.num_classes, dfa.num_classes);
    assert_eq!(decoded.transitions, dfa.transitions);
    assert_eq!(decoded.token_ids, dfa.token_ids);
    println!("round trip is byte-for-byte identical");

    // A single flipped payload byte must be rejected, not silently accepted.
    let mut corrupted = blob.bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0xff;
    match decode_dfa_from_bytes(&corrupted) {
        Ok(_) => {
            eprintln!("a corrupted blob decoded successfully, which must never happen");
            std::process::exit(1);
        }
        Err(error) => println!("corrupted blob rejected: {error:?}"),
    }
}
