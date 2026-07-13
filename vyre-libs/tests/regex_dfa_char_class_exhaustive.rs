//! Exhaustive alphabet-correctness for the regex DFA's character classes, the
//! primitive every token detector is built on (`ghp_[A-Za-z0-9]{n}`, `[a-f0-9]`
//! hex, …). For each class pattern the unanchored DFA must accept a single byte iff
//! that byte is a member, checked against a plain-Rust membership oracle over ALL
//! 256 byte values (not a sampled subset, a class boundary bug hides in exactly the
//! one un-sampled byte). This is the foundation the leftmost-longest and anchoring
//! suites assume; locking it makes those suites' "body is `[class]`" premise sound.
#![cfg(feature = "matching-regex")]

use vyre_libs::scan::regex_dfa::build_regex_dfa_unanchored;

/// `true` iff the unanchored DFA for `pattern` (a single one-char class) accepts
/// after consuming exactly `byte` from the start state. For a one-char pattern the
/// start-state transition lands directly on an accepting state iff `byte` matches.
fn class_admits(dfa_transitions: &[u32], dfa_accept: &[u32], byte: u8) -> bool {
    let state = dfa_transitions[byte as usize]; // start state 0: index = 0*256 + byte
    dfa_accept[state as usize] != 0
}

#[test]
fn character_class_accepts_exactly_its_members_over_all_256_bytes() {
    // (regex class, membership oracle) (oracle is the ground truth).
    let classes: &[(&str, fn(u8) -> bool)] = &[
        ("[a-z]", |b| b.is_ascii_lowercase()),
        ("[A-Z]", |b| b.is_ascii_uppercase()),
        ("[0-9]", |b| b.is_ascii_digit()),
        ("[A-Za-z0-9]", |b| b.is_ascii_alphanumeric()),
        ("[a-fA-F0-9]", |b| b.is_ascii_hexdigit()),
        // Negated + explicit-set forms exercise different lowering paths.
        ("[^0-9]", |b| !b.is_ascii_digit()),
        ("[aeiou]", |b| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')),
    ];

    for (pattern, oracle) in classes {
        let pipeline = build_regex_dfa_unanchored(&[pattern], 1024, 1 << 16)
            .unwrap_or_else(|e| panic!("class {pattern:?} must compile: {e:?}"));
        let transitions = &pipeline.dfa.transitions;
        let accept = &pipeline.dfa.accept;

        let mut mismatches = Vec::new();
        for byte in 0u8..=255 {
            let admitted = class_admits(transitions, accept, byte);
            if admitted != oracle(byte) {
                mismatches.push((byte, admitted, oracle(byte)));
            }
        }
        assert!(
            mismatches.is_empty(),
            "class {pattern:?} disagrees with its membership oracle on {} byte(s): {:?} \
             (each tuple = (byte, dfa_admitted, oracle_says_member))",
            mismatches.len(),
            mismatches
                .iter()
                .take(8)
                .map(|(b, d, o)| (*b, *b as char, *d, *o))
                .collect::<Vec<_>>()
        );
    }
}
