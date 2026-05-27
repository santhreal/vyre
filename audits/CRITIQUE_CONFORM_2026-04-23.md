# Conformance Subsystem Security Audit  -  2026-04-23

**Scope:** Every `.rs` file in `vyre/conform/` (runner, spec, enforce, generate, test-harness).
**Auditor:** Kimi Code CLI
**Date:** 2026-04-23

---

## Hunt Responses (Direct Answers)

### Hunt 1  -  Does `BundleCertificate::verify` re-check the witness corpus against the backend, or does it just compare a hash to a hash?
**Answer:** `verify_bundle_with_backend` **does re-execute** every witness in the canonicalised corpus through the live backend (`backend_dispatch`), hashes the resulting output streams, and compares that aggregate hash to `cert.reference_output_blake3`. A forged cert with a valid hash but stale witnesses would **not** pass, because the live backend produces new outputs.

**However**, the verifier does **not** cryptographically verify the Ed25519 signature that attests to the cert. An attacker who can forge the signature field (or simply stuff random hex into it) can modify the cert metadata, re-issue it, and the hash-only verifier will accept it. The re-execution is correct; the signature verification is missing.

### Hunt 2  -  Are all hash inputs domain-separated?
**Answer:** **No.** `canonicalise_corpus`, `hash_output_streams`, `issue_certificate`, and `U32Witness::fingerprint_canonical` all use bare `blake3::Hasher::new()` without a domain tag. Only `BackendFingerprint::from_observation` prefixes with `v1\0`. The absence of domain separation means a byte-identical encoding in one context (e.g., a program wire image) could be replayed into another context (e.g., a witness corpus) and produce the same digest. While blake3 is collision-resistant, the architectural gap violates the "extend, don't hack" principle and removes a defense-in-depth layer against cross-context preimage confusion.

### Hunt 3  -  `ConformanceCase::label` uniqueness
**Answer:** The type `ConformanceCase` does not exist in this codebase. The nearest analog is `CorpusWitness::name` in `bundle_cert.rs`. **Uniqueness is not enforced.** `canonicalise_corpus` sorts by name but permits duplicates. A forged corpus containing two witnesses with the same name (one benign, one malicious) will hash deterministically and pass `verify_bundle_with_backend`, but a downstream UI or cache that indexes by name may silently overwrite the malicious entry, hiding a divergence.

### Hunt 4  -  Does the runner skip ops that don't match a capability filter?
**Answer:** **It depends on which runner.**
- The CLI binary (`vyre-conform-runner/src/main.rs`) does **not** skip based on capability filters; it dispatches every op directly. However, it **omits the entire `vyre_primitives` catalog** from `unified_entries()`, so those ops are silently skipped.
- The shared lens module (`vyre-test-harness/src/lens.rs`) **does** skip via `program_caps::check_backend_capabilities`. `LensOutcome::is_ok()` returns `true` for skips, so tests that assert `is_ok()` pass with zero actual coverage when a capability is missing.
- The `parity_matrix.rs` integration test passes when all backends fail to dispatch (zero divergences, 100% skips) because it only asserts `divergences.is_empty()`.

**Capability-filter holes = silent under-coverage is a confirmed finding.**

### Hunt 5  -  Cert-signing key storage location
**Answer:** **There is no persistent key storage location.** The `prove` command generates an ephemeral Ed25519 key in memory and discards it after signing. The artifact is written to the user-supplied `--out` path.

**However**, the ephemeral key is derived from `blake3(program_hash:pid:SystemTime::now())`. The timestamp has very low entropy and is guessable. An attacker who knows the approximate CI runtime can brute-force the 32-byte seed, recover the private key, and forge `prove` artifacts. The signature is therefore security theater, not a cryptographic guarantee.

---

## Findings

### C1  -  CRITICAL | `vyre-conform-runner/src/bundle_cert.rs:299` + `vyre-conform-runner/src/cert.rs:160`
**Description:** `verify_bundle_with_backend`, `verify_bundle_against_reference`, and `verify_structural` never cryptographically verify the Ed25519 signature. They accept any hex string in `signature_ed25519` and `pubkey`. An attacker can forge a certificate, modify hashes or metadata, insert a random signature, and every verification path in this crate will accept it.

**Fix:** Add a `verify_signature` helper using `ed25519_dalek::VerifyingKey` and `ed25519_dalek::Signature`. Invoke it at the top of `verify_bundle_with` (before hash checks) and add a `verify_cryptographic(cert, pubkey)` function for `Certificate`. Reject with a new `SignatureInvalid` error variant.

**Test hint:** Issue a valid cert, mutate one hex digit in `reference_output_blake3`, re-sign with the original key, and assert `verify_bundle_with_backend` returns `SignatureInvalid`. Then issue a completely forged cert signed by a new attacker keypair and assert the same rejection.

---

### C2  -  CRITICAL | `vyre-conform-runner/src/main.rs:396-403`
**Description:** The `prove` command derives the ephemeral Ed25519 signing key from `blake3(format!("{}:{}:{:?}", program_hash, pid, SystemTime::now()))`. The program hash is public, the PID is small (≤2^22), and the system clock has microsecond resolution. An attacker with approximate CI timestamps can brute-force the 32-byte seed in practical time, recover the private key, and forge signed `prove` artifacts.

**Fix:** Replace deterministic derivation with `SigningKey::generate(&mut OsRng)` from `rand`. If reproducibility across CI reruns is required, load a long, high-entropy master secret from an environment variable or OS keyring and derive the signing key via HKDF, never from public metadata.

**Test hint:** Run `prove` twice within the same second; assert the `public_key` fields differ (proving true randomness). Alternatively, mock `SystemTime::now()` to a fixed value and demonstrate that the resulting key is still unpredictable without the secret.

---

### H1  -  HIGH | `vyre-conform-runner/src/main.rs:162-179`
**Description:** `unified_entries()` only chains `vyre_libs::harness::all_entries()` and `vyre_intrinsics::harness::all_entries()`. It omits `vyre_primitives::harness::all_entries()`. Both `vyre-conform dispatch --ops all` and `vyre-conform prove` therefore silently skip the entire primitive op catalog (bitset, reduce, label, predicate, fixpoint, etc.). This is a massive silent under-coverage hole.

**Fix:** Chain `vyre_primitives::harness::all_entries()` into `unified_entries()`, matching the coverage in `parity_matrix.rs`.

**Test hint:** Register a synthetic primitive op with a unique id. Run `vyre-conform dispatch --ops all` and assert its id appears in the emitted JSON. Run `prove` and assert the artifact's `pairs` array contains the primitive op.

---

### H2  -  HIGH | `vyre-test-harness/src/lens.rs:125-191` + `vyre-test-harness/src/lens.rs:201-306`
**Description:** `cpu_vs_backend` and `fixpoint` lenses skip ops when `program_caps::check_backend_capabilities` reports a missing capability, returning `LensOutcome::Skip`. `LensOutcome::is_ok()` returns `true` for skips. Any test or CI gate that asserts `is_ok()` will pass even if 100% of ops were skipped due to a capability-filter hole (e.g., a new `bf16` requirement that the filter does not yet know about).

**Fix:** Remove `Skip` from `is_ok()`, or introduce a separate `is_pass()` method that is `true` only for `Pass`. Update all test assertions to require `is_pass()` and separately allow-list expected skips by name.

**Test hint:** Register a synthetic op requiring a fake capability not handled by `check_backend_capabilities`. Run the lens against a backend and assert the test suite fails, not silently passes via `is_ok()`.

---

### H3  -  HIGH | `vyre-conform-runner/tests/parity_matrix.rs:352-356`
**Description:** The parity matrix test asserts only `summary.divergences.is_empty()`. It does not assert that `summary.ops_covered > 0` or that the skip count is below a threshold. If every backend fails to instantiate (driver missing, factory error) or every op is skipped, the test passes with zero actual coverage.

**Fix:** Add `assert!(summary.ops_covered > 0, "Fix: zero ops covered")` and `assert!(summary.skipped.len() < summary.ops_total * backends_runnable, "Fix: excessive skips")` before the divergence check.

**Test hint:** Patch `registered_backends()` to return an empty slice, or set an environment variable that forces every factory to fail. The test must exit non-zero, not pass.

---

### H4  -  HIGH | `vyre-conform-runner/src/main.rs:181-306`
**Description:** `compare_backend_against_reference` returns `PairResult { passed: true, message: "0 witness case(s) matched ..." }` when `test_inputs()` returns an empty vector. An op that registers a `test_inputs` function producing zero cases receives a passing certificate with zero coverage.

**Fix:** At the top of the case loop, if `cases.is_empty()`, return `passed: false` with message `"Fix: op has zero witness cases  -  empty fixture is not coverage."`.

**Test hint:** Register an op whose `test_inputs` returns `vec![]`. Run `vyre-conform prove` and assert the op appears in the failing pairs list with an empty-fixture error.

---

### H5  -  HIGH | `vyre-conform-runner/src/bundle_cert.rs:152-175`
**Description:** `canonicalise_corpus` sorts `CorpusWitness` by `name` but never checks for duplicates. A forged corpus containing two witnesses with the same name (one benign, one malicious) will hash deterministically and pass `verify_bundle_with_backend`, but a downstream display layer that indexes by name will silently overwrite the malicious entry.

**Fix:** In `canonicalise_corpus` (or `issue_bundle_cert`), reject duplicates:
```rust
if sorted.windows(2).any(|w| w[0].name == w[1].name) {
    return Err(BundleCertError::DuplicateWitnessName);
}
```

**Test hint:** Attempt to issue a cert with corpus `[CorpusWitness { name: "dup", inputs: a }, CorpusWitness { name: "dup", inputs: b }]`. Assert `issue_bundle_cert` returns `DuplicateWitnessName`.

---

### H6  -  HIGH | `vyre-conform-enforce/src/prover.rs:65-116`
**Description:** `LawProver` uses hardcoded deterministic PRNG seeds (`0x1337_BEEF`, `0xBEEF_1337`). The sample sequences are fully predictable. A malicious op author can craft a witness set that passes exactly the sampled pairs while violating the law on all other pairs, fraudulently claiming commutativity, associativity, or identity on the certificate.

**Fix:** Replace `Xorshift32` with a CSPRNG seeded from `OsRng`, or switch to exhaustive testing for witness sets below a threshold (e.g., ≤256). If stochastic verification is retained, clearly document it as a screening step, not a certification step.

**Test hint:** Build a witness set and a non-commutative function `f` such that `f(a,b) == f(b,a)` for the first 64 `Xorshift32(0x1337_BEEF)` samples but fails on sample 65. Assert `verify_commutative` incorrectly returns `Holds`.

---

### H7  -  HIGH | `vyre-conform-enforce/tests/composition_discipline.rs:284-287`
**Description:** `hash_expr` for `Expr::Call` hashes only the discriminant (`112`) and `args.len()`, ignoring both the `op_id` and the argument expressions. Two programs that call different ops with different arguments but the same arity produce identical structural fingerprints. An attacker can trivially craft an op whose fingerprint collides with an op in a different namespace, bypassing the cross-namespace subsumption gate.

**Fix:** Recurse into `args` and hash the `op_id` bytes:
```rust
Expr::Call { op_id, args } => {
    mix(h, 112);
    for b in op_id.as_bytes() { mix(h, *b as u64); }
    mix(h, args.len() as u64);
    for arg in args { hash_expr(arg, h); }
}
```

**Test hint:** Create two programs: one with `Expr::Call { op_id: "a", args: [Expr::u32(1)] }` and one with `Expr::Call { op_id: "b", args: [Expr::u32(2)] }`. Assert their structural fingerprints differ after the fix.

---

### M1  -  MEDIUM | `vyre-conform-runner/src/bundle_cert.rs:152-193` + `vyre-conform-runner/src/cert.rs:130-131` + `vyre-conform-spec/src/witness.rs:44-51`
**Description:** Bare `blake3::Hasher::new()` is used without domain-separation prefixes in `canonicalise_corpus`, `hash_output_streams`, `issue_certificate`, and `U32Witness::fingerprint_canonical`. A byte-identical payload in one context can produce the same digest in another context, enabling cross-context preimage confusion.

**Fix:** Prefix every hash stream with a unique domain tag:
- `hasher.update(b"vyre.bundle-corpus.v1")`
- `hasher.update(b"vyre.bundle-outputs.v1")`
- `hasher.update(b"vyre.occ.program.v1")`
- `hasher.update(b"vyre.occ.witness.v1")`

**Test hint:** Compute the old and new hashes for the same inputs and assert they differ. Verify that a cert issued with the new domain tags fails verification against the old code.

---

### M2  -  MEDIUM | `vyre-conform-runner/src/cert.rs:160-179`
**Description:** `verify_structural` validates `program_blake3` and `witness_set_blake3` as 64-hex-char strings, but it does not validate `signature_ed25519` (must be 128 hex chars for 64 bytes) or `pubkey` (must be 64 hex chars for 32 bytes). A 32-character "signature" passes structural check.

**Fix:** Add length and hex-digit checks for `signature_ed25519` and `pubkey`, returning `BadFingerprint` when invalid.

**Test hint:** Create a cert with `signature_ed25519 = "ab".repeat(16)` (32 chars) and `pubkey = "cd".repeat(16)` (32 chars). Assert `verify_structural` returns `BadFingerprint` for both fields.

---

### M3  -  MEDIUM | `vyre-conform-spec/src/cert/fingerprint.rs:54-66`
**Description:** `BackendFingerprint::from_observation` builds a `\0`-delimited canonical string but does not reject null bytes in `backend` or `adapter`. An attacker can inject `\0` to shift delimiter boundaries and produce a fingerprint collision with a different backend configuration.

**Fix:** Validate that `backend` and `adapter` contain no `\0` bytes in `ProbeObservation::new`, or switch to a length-prefixed canonical encoding.

**Test hint:** Construct two observations:
- A: `backend="wgpu\0v1", adapter="nvidia"`
- B: `backend="wgpu", adapter="v1nvidia"`
Assert they produce different fingerprints; if they collide, the bug is confirmed.

---

### M4  -  MEDIUM | `vyre-test-harness/src/lens.rs:179`
**Description:** `cpu_vs_backend` uses raw byte comparison (`cpu != gpu`) instead of `vyre_conform_runner::fp_parity::compare_output_buffers`. F32 buffers within the spec-allowed ULP window are reported as divergences, causing false positives that lead developers to add `UniversalDiffExemption`s, which then hide real regressions.

**Fix:** Replace the raw `!=` check with `compare_output_buffers(&program, &cpu, &gpu)`.

**Test hint:** Create an op with a single F32 `exp` witness that diverges by 10 ULP (within the 64-ULP transcendental window). Run `cpu_vs_backend` and assert `Pass`, not `Fail`.

---

### M5  -  MEDIUM | `vyre-conform-runner/src/fp_parity.rs:146-148`
**Description:** `program_has_transcendental` is intraprocedural  -  it does not recurse into the bodies of ops invoked via `Expr::Call`. A program that wraps a transcendental in a helper op is classified as non-transcendental and receives the tight 4-ULP tolerance instead of 64-ULP, causing false-positive divergence reports.

**Fix:** Resolve `Expr::Call` op_ids against the registry and recursively inspect the callee's `program.entry()` for transcendentals. Cache results to avoid O(N²) traversal.

**Test hint:** Build a program that calls `vyre-libs::math::exp` via `Expr::Call`. Assert `f32_ulp_tolerance` returns `64`, not `4`.

---

### M6  -  MEDIUM | `vyre-conform-enforce/tests/composition_discipline.rs:185-191`
**Description:** `structural_fingerprint` returns a 64-bit custom FNV-1a hash. A 64-bit digest provides only ~2³² collision resistance (birthday bound). An attacker can craft a new op that collides with an existing op's fingerprint in a different namespace, bypassing the "no reimplementation" gate.

**Fix:** Replace the 64-bit hash with a 256-bit blake3 hash over a canonical binary encoding of the AST shape.

**Test hint:** Write a brute-force search that generates random small programs until two distinct programs in different namespaces produce the same 64-bit fingerprint. After the fix, the same search should exhaust its budget without a collision.

---

### M7  -  MEDIUM | `vyre-conform-enforce/tests/composition_discipline.rs:488-531`
**Description:** `every_op_has_test_fixtures_or_is_explicitly_exempt` only flags ops where **both** `test_inputs` and `expected_output` are `None`. An op that supplies `test_inputs` but no `expected_output` (or vice versa) passes the gate but is incomplete.

**Fix:** Change the condition to `if entry.test_inputs.is_none() || entry.expected_output.is_none()`.

**Test hint:** Register an op with `test_inputs: Some(...)` and `expected_output: None`. Assert the gate fails, naming the missing fixture.

---

### L1  -  LOW | `vyre-conform-runner/src/bundle_cert.rs:57-74`
**Description:** `BundleCertificate` stores `witness_count: u64`, but `verify_bundle_with` never validates that the supplied corpus length matches this field. A tampered cert can claim a misleading witness count without affecting verification.

**Fix:** In `verify_bundle_with`, add:
```rust
if sorted_corpus.len() as u64 != cert.witness_count {
    return Err(BundleCertError::WitnessCountMismatch { expected: cert.witness_count, observed: sorted_corpus.len() as u64 });
}
```

**Test hint:** Mutate `cert.witness_count` to `99` and assert `verify_bundle_against_reference` returns `WitnessCountMismatch`.

---

### L2  -  LOW | `vyre-conform-runner/src/bundle_cert.rs:195-201`
**Description:** `hex32` swallows the `Result` from `write!` via `let _ = ...`. While `String::write_str` is effectively infallible today, this pattern violates the "never swallow errors" standard and will hide a regression if the implementation changes.

**Fix:** Replace the manual loop with `hex::encode(bytes)` (already a dependency), or use `write!(&mut out, "{b:02x}").unwrap();`.

**Test hint:** Run `cargo clippy -- -D warnings`; the unused `Result` should be flagged.

---

### L3  -  LOW | `vyre-conform-runner/src/main.rs:181-306`
**Description:** `compare_backend_against_reference` does not catch panics from `backend.dispatch` or `vyre_reference::reference_eval`. A misbehaving backend or reference can abort the entire `prove` or `dispatch` process. `parity_matrix.rs` already wraps backend dispatch in `catch_unwind`; the CLI runner should do the same.

**Fix:** Wrap dispatch and comparison in `std::panic::catch_unwind(AssertUnwindSafe(...))` and convert a panic into a `PairResult { passed: false, message: "panic: ..." }`.

**Test hint:** Inject a test backend whose `dispatch` panics. Run `vyre-conform dispatch` and assert it exits code 1 with a JSON result containing `passed: false`, rather than aborting the process.

---

## Severity Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH     | 7 |
| MEDIUM   | 7 |
| LOW      | 3 |
| **Total**  | **19** |

## Top 3 to Escalate

1. **C1  -  Missing Ed25519 signature verification** (`bundle_cert.rs` + `cert.rs`). The entire certificate subsystem is security theater; anyone can forge a cert and the verifier will accept it.
2. **C2  -  Low-entropy ephemeral key derivation in `prove`** (`main.rs`). The signing key is brute-forceable from public metadata, making the `prove` artifact signature worthless and forgeable.
3. **H1  -  `vyre_primitives` catalog omitted from CLI runner** (`main.rs`). The `dispatch` and `prove` commands silently skip an entire op family, meaning untested primitives ship to production.
