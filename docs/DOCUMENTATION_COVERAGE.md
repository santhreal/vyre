# Documentation coverage

Applies to Vyre 0.7.2.

Documentation coverage is measured, not assumed. A file existing on disk does
not prove that its claims match the current code or release train.

## Measured surfaces

| Surface | Authority | Gate | What the result proves |
| --- | --- | --- | --- |
| Navigation and lifecycle | [`DOCS.toml`](DOCS.toml) | `python3 scripts/docs_manifest.py --check` | Every Markdown page has one lifecycle row, every active page is reachable once, inactive pages are excluded, and generated pages name one source. |
| Crate ownership | Cargo metadata and [`CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml) | `python3 scripts/crate_ownership.py --check` | Every workspace package has one owner and allowed production dependency set. |
| Crate testing guides | Cargo targets plus [`testing/TESTING.toml`](testing/TESTING.toml) | `python3 scripts/testing_guides.py --check` | Every workspace package has current commands, hardware requirements, evidence outputs, skip rules, and failure semantics. |
| Public API snapshots | Publishable workspace manifests | `bash scripts/check_public_api_snapshot.sh` | Snapshot files exactly match the publishable package set. |
| Markdown links | Active Markdown pages | `bash scripts/check_docs_links.sh` | Active Markdown link targets exist and are publishable. |
| Path-like references | Active Markdown pages | `python3 scripts/check_docs_references.py` | Explicit repository paths used as inputs exist and are publishable. |

These gates measure different facts. A clean link gate does not prove a support
claim, and a current API snapshot does not prove that an example executes.

## Documentation authorities

| Question | Authority |
| --- | --- |
| What is the executable lifecycle? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Which crate owns a responsibility or dependency edge? | [`CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml) |
| Where does an operation live? | The foundation operation registry and its generated schema |
| How do you add an operation or backend? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What is the wire format? | [`docs/wire-format.md`](wire-format.md) |
| What is the memory model? | [`docs/memory-model.md`](memory-model.md) |
| Which error does a Vyre surface return? | [`docs/error-codes.md`](error-codes.md) and the owning crate's rustdoc |
| How do you run a crate's tests? | The generated guide under [`docs/testing/`](testing/) |
| How do you run release benchmarks? | [`docs/PERF.md`](PERF.md), [`vyre-bench/README.md`](../vyre-bench/README.md), and the benchmark target registry |
| When can a release tag ship? | [`docs/RELEASE.md`](RELEASE.md) and the generated release evidence |
| How does a generic downstream analyzer integrate? | [`docs/consumer-integration.md`](consumer-integration.md) |
| Which named external integration is documented? | [`docs/consumer-showcase.md`](consumer-showcase.md) |

No local contract depends on an unpublished consumer tree. A named external
integration owns its own product documentation, support status, benchmarks, and
severity policy.

## Public item contract

When you add a public item:

1. Write a summary that states what the item does.
2. Add a compiling example that exercises observable behavior.
3. Document every error condition for a fallible function.
4. Name the owning tier and canonical sibling for a registered operation.
5. Prove a Tier 3 composition chain reaches registered lower-tier operations.
6. Regenerate the public API snapshot and corpus documentation evidence.

`missing_docs` is necessary but not sufficient. The documentation matrix must
also connect the claim to current manifests, executable examples, and release
evidence.
