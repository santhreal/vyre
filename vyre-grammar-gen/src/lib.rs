//! # vyre-grammar-gen
//!
//! Host-side C11 grammar table generator for the vyre GPU C parser.
//! Produces DFA lexer + LR(1) action/goto tables as binary blobs that
//! `vyre-libs::parsing` loads as `ReadOnly` storage buffers.
//!
//! See `README.md` for the pipeline and binary-blob wire format.
//!
//! ## Safe defaults
//!
//! **Input size:** No hard cap enforced by the library. `DfaBuilder::new(states,
//! classes)` allocates `states * classes * 4` bytes for the transition table;
//! callers are responsible for bounding those dimensions. `decode_dfa_from_bytes`
//! and `decode_lr_from_bytes` read only the slice they are handed - no unbounded
//! allocation beyond what the `payload_len` header field indicates.
//!
//! **Recursion depth:** No recursion in any public library function. All
//! algorithms (`preprocess_c_host`, DFA construction, wire encoding/decoding) are
//! iterative; stack depth is O(1) with respect to input size.
//!
//! **Outbound network:** None. The library makes no network calls. All I/O is
//! via the caller-supplied byte slices or `String`/`Vec` return values.
//!
//! **Process spawning:** None. The library never spawns child processes or
//! invokes `std::process`. The `vyre-grammar-gen` binary in `src/main.rs`
//! writes files to disk, but the library itself does not.
//!
//! **Filesystem writes:** None by the library. `preprocess_c_host`,
//! `build_c11_lexer_dfa`, `PackedBlob::from_dfa`, `PackedBlob::from_lr`
//! (which returns `Result<Self, String>` and validates table dimensions before
//! packing), and all decode functions operate purely in memory and return
//! owned values. Only the `main.rs` binary writes files (via `std::fs::write`).
//!
//! **Credential exposure:** None. No credentials, tokens, or secrets are
//! read, logged, or transmitted. BLAKE3-128 is used solely for payload
//! integrity verification of binary blobs, not for any authentication purpose.


pub mod c11_lexer;
pub mod chunk_lexer_cpu;
pub mod dfa;
pub mod host_preprocess;
pub mod lex_c11_max_munch;
pub mod lr;
pub mod max_munch_cpu;
pub mod wire;
