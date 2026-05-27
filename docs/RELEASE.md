# vyre release process

## 0.4.1 playbook (quick-reference)

Operator with `cargo_full login <crates.io token>` runs:

```bash
# Pre-flight
cd libs/performance/matching/vyre
CARGO_TARGET_DIR=target-release cargo_full test -j1 --workspace
CARGO_TARGET_DIR=target-release cargo_full run -j1 -p vyre-bench --release -- \
    run --backend cuda --suite release --measured-samples 30 \
    --warmup-samples 3 --enforce-budgets
CARGO_TARGET_DIR=target-release target-release/release/vyre-bench \
    run --backend wgpu --suite release --measured-samples 30 \
    --warmup-samples 3 --enforce-budgets

# Alpha: bump to 0.4.1-alpha.1 across publishable crates + workspace deps,
# publish in dependency order; wait for each crate to appear in the crates.io index before the next dependent crate
for crate in vyre-lints vyre-macros vyre-spec vyre-foundation \
             vyre-lower vyre-primitives vyre-emit-ptx vyre-reference \
             vyre-self-substrate vyre-driver vyre vyre-aot \
             vyre-driver-cuda vyre-driver-reference vyre-emit-naga \
             vyre-intrinsics vyre-runtime vyre-driver-spirv \
             vyre-driver-wgpu vyre-emit-spirv vyre-harness \
             vyre-libs vyre-debug vyre-test-harness; do
    cargo_full publish -p $crate --locked
    bash scripts/wait-crates-index.sh "$crate" "0.4.1-alpha.1"
done

# Smoke test on a fresh machine
cargo_full install vyre --version =0.4.1-alpha.1 --root /tmp/s
/tmp/s/bin/vyre --version

# 24-hour soak; monitor github issues + security@santh.dev
# On clean soak, bump versions to 0.4.1, repeat publish loop
```

CUDA is the release speed backend. WGPU is the portability/correctness fallback.
Do not convert a WGPU performance miss into a CUDA release blocker when WGPU
produces correct outputs.

The CUDA release path must keep `BufferDecl::output_byte_range` semantics
intact for direct dispatch and cudaGraph replay. Narrow and zero-length
readbacks are performance fixes: megakernel control-report workloads rely on
them to avoid copying ring/debug/IO buffers back to host.

Yank + fix recipe for security issues:

```bash
cargo_full yank --vers 0.4.1 <crate>
# fix + publish 0.4.2
```

Full topological ordering, rollback, and RFC review notes below.

---



Every publishable crate in the vyre workspace publishes to crates.io on every tagged release. This document is the single source of truth for cutting a release; any deviation is a CI or process bug, not a valid shortcut.

For conflicts between release docs, plans, audits, generated docs, and
internal archives, use [`docs/DOCUMENTATION_GOVERNANCE.md`](docs/DOCUMENTATION_GOVERNANCE.md).

## Topological publish order

Crates must be published in dependency order so that every path dependency can
resolve from crates.io before its consumer publishes.

1. `vyre-lints`
2. `vyre-macros`
3. `vyre-spec`
4. `vyre-foundation`
5. `vyre-lower`
6. `vyre-primitives`
7. `vyre-emit-ptx`
8. `vyre-reference`
9. `vyre-self-substrate`
10. `vyre-driver`
11. `vyre`
12. `vyre-aot`
13. `vyre-driver-cuda`
14. `vyre-driver-reference`
15. `vyre-emit-naga`
16. `vyre-intrinsics`
17. `vyre-runtime`
18. `vyre-driver-spirv`
19. `vyre-driver-wgpu`
20. `vyre-emit-spirv`
21. `vyre-harness`
22. `vyre-libs`
23. `vyre-debug`
24. `vyre-test-harness`

## Pre-release checklist

1. `cargo_full check --workspace --all-targets --all-features`  -  zero errors, zero warnings.
2. `cargo_full test --workspace --release --all-features`  -  every test passes.
3. `cargo_full clippy --workspace --all-targets --all-features -- -D warnings`  -  clean.
4. `cargo_full +nightly udeps --workspace`  -  no unused deps.
5. `cargo_full deny check`  -  licenses + advisories + sources green.
6. `cargo_full public-api --all-features`  -  diff against `docs/public-api/*.txt` baselines zero unexpected.
7. `cargo_full semver-checks check-release`  -  every publishable crate passes.
8. `cargo_full run -p vyre-bench --release -- run --backend cuda --suite release --measured-samples 30 --warmup-samples 3 --enforce-budgets`  -  CUDA release path passes.
9. `cargo_full run -p vyre-bench --release -- run --backend wgpu --suite release --measured-samples 30 --warmup-samples 3 --enforce-budgets`  -  WGPU fallback path passes correctness.
10. `cargo_full bench -p vyre-bench`  -  runs, produces numbers.
11. `cargo_full test -p vyre-driver-cuda cuda_honors_`  -  CUDA output-range readback semantics verified.
12. `cargo_full test -p vyre-driver-cuda cuda_graph_honors_output_byte_ranges_like_direct_dispatch`  -  cudaGraph readback semantics verified.
13. Every crate's `CHANGELOG.md` has an entry for the new version.
14. Workspace `Cargo.toml` version bumps are coherent (no crate on an older version than what it depends on).
15. `CITATION.cff` version field matches the tag.

## Publish

For each crate, in the order above:

```bash
CARGO_TARGET_DIR=target-release cargo_full publish --dry-run --locked -p <crate>
cargo_full publish --locked -p <crate>
bash scripts/wait-crates-index.sh <crate> <version>
```

`cargo_full publish --dry-run` for a crate with internal dependencies only works
after the dependency versions are already present in the registry. For the
alpha release, dry-run and publish each crate in topological order, then wait
for the index before moving to the next crate. A full pre-publish dry-run of all
24 crates against crates.io is impossible when crates.io still only has older
Vyre versions.

Pre-publish package metadata checks must still be clean before the first
publish:

```bash
cargo_full metadata --no-deps --format-version 1 | jq -e '
  [.packages[]
   | select(.publish != [])
   | select((.repository == null) or (.homepage == null) or
            (.documentation == null) or (.readme == null))] | length == 0'

cargo_full metadata --no-deps --format-version 1 | jq -e '
  [.packages[] | select(.publish != []) as $p
   | $p.dependencies[]
   | select(.path != null and (.req == "*" or .req == "" or .req == null) and
            (.kind == null))] | length == 0'
```

During this pass, dry-run succeeded for `vyre-lints`, `vyre-macros`, and
`vyre-spec`. The next crate, `vyre-foundation`, correctly required
`vyre-macros = "^0.4.1"` from crates.io and therefore cannot dry-run until
`vyre-macros 0.4.1` is actually published and indexed.

## Tag + release notes

After the last crate publishes:

```bash
git tag vyre-v0.4.1
git tag vyre-0.4.1-dataflow consumer-0.0.1
git push origin vyre-v0.4.1 vyre-0.4.1-dataflow consumer-0.0.1
gh release create vyre-v0.4.1 \
    --title "vyre v0.4.1  -  CUDA-first branchy condition execution" \
    --notes-file docs/release/v0.4.1.md
```

The The dataflow consumer repository/release path must also create `dataflow consumer-v0.0.1` before the combined release-train tag is considered complete. Release notes are generated from the per-crate `CHANGELOG.md` entries for the new version. Never hand-write them separately from the changelogs.

## Rollback

If a crate publishes broken, **yank**, do not unpublish (crates.io does not permit unpublish after 72h). Cut a patch release the same day.

```bash
cargo_full yank --vers 0.4.1 <crate>
```

Fix forward; publish `0.4.2` with the correction.

## Post-release

1. Update the `README.md` banner at workspace root with the new version.
2. Update the user-facing install documentation.
3. Open issues for every finding surfaced during the release that didn't block ship.
4. Single Telegram ping to `@SanthCEObot` with the release URL.

## Publish DAG

```text
vyre-lints
vyre-macros
vyre-spec
  └──> vyre-foundation ──> vyre-lower ──> emitters/backends/runtime/libs
vyre-driver
  ├──> vyre-driver-cuda
  ├──> vyre-driver-reference
  ├──> vyre-driver-spirv
  └──> vyre-driver-wgpu
vyre
  ├──> vyre-aot
  ├──> vyre-debug
  ├──> vyre-harness
  ├──> vyre-libs
  └──> vyre-test-harness
```

Publish order is the numbered order above. Do not publish blocked workspace
tools (`xtask`, `vyre-bench`, `vyre-frontend-c`, or `vyre-conform-*`) unless
their manifests are explicitly changed to publishable crates.

Verify the DAG automatically from `Cargo.toml` metadata:

```sh
cargo_full run --bin xtask -- release-order
```

The `xtask` prints the publish order as a topological sort of the
crate graph; any deviation from the order above signals a dependency
cycle or a missing crate.

## Community / post-0.4.1 crates

These publish independently on their own cadence after 0.4.1:

- `vyre-pipeline-cache`  -  content-addressed SPIR-V blob store.
- `vyre-autodiff`  -  reverse-mode AD transform (roadmap R-1).
- `vyre-verify`  -  Kani theorem harness (roadmap R-2).
- `vyre-libs-llm`  -  FlashAttention-v2 + KV-cache + MoE (roadmap R-3).
- Community dialect packs under `vyre-libs-*` (e.g.
  `vyre-libs-quant`, `vyre-libs-sparse`, `vyre-libs-collective`).
- Community-registered backends following `vyre-driver-*` naming.

Every community crate must pin `vyre = "0.4.1"` or later and pass the
conformance-certificate gate (see `conform/`).

## Release evidence

Release readiness for this document is proven through the platform and dataflow evidence manifest and generated artifacts under `release/evidence/`. Claims here must map to concrete gate output, benchmark output, conformance output, parser corpus output, or documentation proof files before the release requirement can be closed.
