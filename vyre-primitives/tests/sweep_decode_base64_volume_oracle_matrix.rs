//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
//!
//! DOMAIN: the differential covers VALID standard base64 (RFC 4648), the only
//! domain where a correctness differential is meaningful. Non-canonical `=`
//! placement (a pad byte anywhere but the final 1-2 positions of the last quad) is
//! NOT valid base64 and has no canonical decoding, so asserting equality of two
//! best-effort schemes there pins coincidence, not a contract (that was the prior
//! RED state: a broken shift-and-skip oracle vs the RFC-correct positional impl,
//! diverging on every `=`). The corpus therefore generates adversarial-random valid
//! base64 across all three padding forms (0/1/2 `=`), and the oracle is an
//! INDEPENDENT streaming bit-accumulator decoder (a different algorithm from the
//! impl's per-quad positional decode) so agreement is real cross-checking, not a
//! reimplementation. The impl's RFC-correctness on padded vectors is additionally
//! locked by the inline KATs in src/decode/base64.rs (TWFu→Man, TQ==→M,
//! Zm9vYmFy→foobar); its fail-loud contract on non-multiple-of-4 input is locked by
//! `cpu_base64_decode_fails_loud_on_invalid_length`.
#![forbid(unsafe_code)]
// The differential drives `cpu_base64_decode`, a CPU reference oracle gated on
// `cfg(any(test, feature = "cpu-parity"))`. In an integration test the lib is built
// without `--cfg test`, so the oracle is reachable ONLY under `cpu-parity`; gating on
// `decode` alone made `cargo test --features decode` fail to compile (unresolved
// import). Declare the true dependency so the suite runs wherever both features are on
// and is cleanly skipped otherwise.
#![cfg(all(feature = "decode", feature = "cpu-parity"))]

use vyre_primitives::decode::base64::cpu_base64_decode;

const CASES: usize = 16384;
/// The 64 standard base64 body symbols (no `=`); `=` is added only as valid
/// terminal padding by the generator.
const BODY_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn build_std_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for (idx, &ch) in BODY_ALPHABET.iter().enumerate() {
        table[ch as usize] = idx as u8;
    }
    table
}

/// Independent RFC-4648 decoder using a streaming 6-bit accumulator (distinct from
/// the impl's per-quad positional layout). For valid base64 the leftover `< 8` bits
/// after the final byte correspond exactly to the terminal padding and are dropped;
/// decoding stops at the first `=` (which, for valid input, is terminal).
fn oracle_base64(input: &[u8]) -> Vec<u8> {
    let table = build_std_table();
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut bits = 0u32;
    let mut nbits = 0u32;
    for &byte in input {
        if byte == b'=' {
            break;
        }
        bits = (bits << 6) | u32::from(table[byte as usize]);
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    out
}

/// Generate adversarial-random but VALID standard base64: full random-body quads,
/// with the LAST quad optionally carrying valid terminal padding (0, 1, or 2 `=`).
/// A 1-pad quad is `[b,b,b,=]`, a 2-pad quad is `[b,b,=,=]`: the only positions the
/// spec permits `=`. Bodies draw from the 64 non-pad symbols, so every input is
/// decodable and both sides have a defined answer.
fn hostile_b64(seed: u32) -> Vec<u8> {
    let quads = 1 + (seed % 48);
    let mut state = seed ^ 0xB64B_64B4;
    let mut next = move || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        state
    };
    let mut out = Vec::with_capacity(quads as usize * 4);
    for q in 0..quads {
        // Padding is legal only in the final quad.
        let pads = if q == quads - 1 {
            (next() % 3) as usize
        } else {
            0
        };
        let body_chars = 4 - pads;
        for _ in 0..body_chars {
            out.push(BODY_ALPHABET[(next() as usize) % BODY_ALPHABET.len()]);
        }
        for _ in 0..pads {
            out.push(b'=');
        }
    }
    out
}

#[test]
fn sweep_decode_base64_volume_oracle_matrix() {
    let mut saw_pad_1 = false;
    let mut saw_pad_2 = false;
    let mut saw_pad_0 = false;
    for idx in 0..CASES {
        let input = hostile_b64(idx as u32);
        let expected = oracle_base64(&input);
        let actual = cpu_base64_decode(&input);
        assert_eq!(
            actual,
            expected,
            "Fix: base64_decode volume case {idx} len={} input={}",
            input.len(),
            String::from_utf8_lossy(&input)
        );
        // Track that all three padding forms are actually exercised, so a
        // generator regression that stopped emitting padded quads can't make the
        // matrix vacuously pass on unpadded-only inputs.
        match input.iter().rev().take_while(|&&b| b == b'=').count() {
            0 => saw_pad_0 = true,
            1 => saw_pad_1 = true,
            2 => saw_pad_2 = true,
            _ => unreachable!("generator never emits more than 2 terminal pads"),
        }
    }
    assert!(
        saw_pad_0 && saw_pad_1 && saw_pad_2,
        "corpus must cover all three padding forms: pad0={saw_pad_0} pad1={saw_pad_1} pad2={saw_pad_2}"
    );
}

#[test]
fn oracle_matches_known_answer_vectors() {
    // Pin the INDEPENDENT oracle itself against RFC-4648 known answers, so the
    // differential cannot pass by the oracle happening to mirror an impl bug.
    assert_eq!(oracle_base64(b"TWFu"), b"Man");
    assert_eq!(oracle_base64(b"TWE="), b"Ma");
    assert_eq!(oracle_base64(b"TQ=="), b"M");
    assert_eq!(oracle_base64(b"Zm9vYmFy"), b"foobar");
    assert_eq!(oracle_base64(b"c3VyZS4="), b"sure.");
}
