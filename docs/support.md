# Getting Support

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

## Supported release line

The active open-source support line is the `0.7.2` train published from the
`santhreal/vyre` workspace. Pre-1.0 means public APIs can still move within the
semver policy in [`docs/semver-policy.md`](semver-policy.md). Experimental
crates (`vyre-runtime` experimental surfaces, beta frontends) may change faster
than frozen foundation and wire contracts.

## Report a reproducible bug

Open a GitHub issue at
[santhreal/vyre](https://github.com/santhreal/vyre/issues/new/choose). Include:

- Vyre crate versions (`vyre`, backend crates if selected)
- Host OS, GPU, and driver/backend (`cuda`, `wgpu`, `metal`, `spirv`, `cpu-ref`)
- Minimal input program or fixture
- Expected result and observed result
- Exact diagnostic codes and full messages
- Relevant conformance or test command output

Do not attach secrets, credentials, or proprietary model weights.

## Ask a design or usage question

Use [GitHub Discussions](https://github.com/santhreal/vyre/discussions). Read
the indexed guides in [`docs/INDEX.md`](INDEX.md) first so the question can
name the current contract. Useful starting points:

- Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Megakernel ownership: [`megakernel-wiring.md`](megakernel-wiring.md)
- FAQ: [`faq.md`](faq.md)
- Errors: [`error-codes.md`](error-codes.md)

## Report a security issue

Follow [`SECURITY.md`](../SECURITY.md). Do not open a public issue for an
unpatched vulnerability.

## Contribute

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md). Operation, backend, and
conformance changes have separate contribution paths. Architecture edge changes
must update `docs/CRATE_OWNERSHIP.toml` in the same patch.

## Cite Vyre

Use [`CITATION.cff`](../CITATION.cff).

## What support does not cover

- Private consumer product paths (for example Keyhog-specific rule packs)
- Untested operation domains outside recorded evidence
- Silent recovery after a requested GPU backend fails
- Commercial SLAs (not part of the Vyre 0.7.2 open-source support contract)
