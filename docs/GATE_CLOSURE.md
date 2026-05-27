# Gate Closure Mechanics

Defines release gate closure mechanics and the yank path if a claimed
release version cannot close its evidence gates.

## The gates

The release gate graduates vyre from "conformance-certified CPU/GPU
parity" to ">=1000x competition gate proven per-cell." Five gates must
all close before tagging:

| Gate | Owner | Measurable |
|---|---|---|
| **G1  -  Zero-stub** | `cargo_full run --bin xtask -- gate1` | `clippy::todo/unimplemented/panic/expect/unwrap` deny passes across every `src/` tree. |
| **G2  -  Conformance** | `vyre-conform-runner prove` | Certification artifact signed by OsRng-seeded Ed25519 (CONFORM C2) covers every registered op, with the hash-chain plus signature both verifying. |
| **G3  -  Region chain** | `cargo_full test -p vyre-libs --test region_chain_invariant` | Every Tier-3 op's Region chain terminates at registered generators (VISION V7). |
| **G4  -  ≥1000× competition** | `cargo_full bench -p consumer --features gpu --bench vs_competition` | Every matrix cell meets its threshold in `libs/tools/consumer/benches/thresholds.toml` (consumer BENCHMARK.md). |
| **G5  -  LAW 7 organisation** | `cargo_full run --bin xtask -- lego-audit` + file-length gate | No file >500 LOC without a split-tracking entry, no cross-dialect reachthrough (VISION V5 guard), every dialect has a README. |

## Closure order

1. **G1** must be clean every commit. Hard CI gate for the selected release train.
2. **G2** + **G3** run on every tag cut. Breakage blocks the tag.
3. **G4** runs on the release branch. A missed cell **blocks the tag** and triggers the E.2 path.
4. **G5** runs on every merge; regressions block merge.

## E.2  -  Yank protocol if G4 cannot close

If G4 cannot close by the release deadline:

1. **Yank the affected selected release version from crates.io** with the bounded `cargo_full` release wrapper on every affected Vyre crate. Yanking keeps existing consumers building but blocks new installs  -  the signal downstream authors need to stop building against a version whose claims are unvalidated.

2. **Publish a holding notice** at the top of each README pointing at the open cell and naming an owner + ETA. Example:

   ```
   > **The selected Vyre release version has been yanked.** The ≥1000× competition gate did
   > not close on (rule_class=regex-backref-free, corpus=10GB,
   > gpu=rtx-4090). Owner: `@<handle>`. Tracking: issue #<n>.
   > New installs should wait on the next evidence-closed patch release (ETA <date>).
   ```

3. **Do not tag** until every gate is green or the user has explicitly waived the affected cell in writing. Signed-off waivers record which cells are known-degraded; the certification artifact lists them.

4. **Keep writing code.** The yank does not pause development; it only gates the *release*. Agents continue closing audit findings, landing optimisations, and growing the Tier-3 surface. The tag cuts when the gates go green.

### Why yank instead of patch-release

A patch release shipping an un-certified ≥1000× claim would make every downstream author relying on the release number silently wrong. Yank is the cheapest honest signal. Downstream authors who already built against the yanked version keep working; nobody new adopts a version whose product claim is under audit.

## Verifying a gate snapshot

```bash
# G1
cargo_full run --bin xtask -- gate1

# G2
cargo_full run -p vyre-conform-runner -- prove --out certs/g2-<tag>.json
cargo_full run -p vyre-conform-runner -- verify certs/g2-<tag>.json \
    --pubkey <trusted.hex>

# G3
cargo_full test -p vyre-libs --test region_chain_invariant

# G4
cargo_full bench -p consumer --features gpu --bench vs_competition
./scripts/check_no_hardcoded_thresholds.sh

# G5
cargo_full run --bin xtask -- lego-audit
./scripts/check_max_file_size.sh
```

Every command must exit 0 against the release commit. CI mirrors this sequence on every tag cut.

## Open source-change items

- The release command list now points at scripts that exist in this
  workspace: `check_no_hardcoded_thresholds.sh` and
  `check_max_file_size.sh`.
- `verify_cert_signature_hex` exists and is pinned by
  `conform/vyre-conform-runner/tests/cert_regression_pin.rs`; any
  verify-path gap is source work, not a doc-only closure.
