# Documentation Governance

## Authorities

| Contract | Authority |
| --- | --- |
| Executable compiler and artifact lifecycle | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Crate ownership and allowed production dependencies | [`CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml) |
| Optimization layers, owners, and benchmark targets | [`optimization/README.md`](optimization/README.md) |
| Documentation lifecycle, audience, owner, authority, generation, and navigation | [`DOCS.toml`](DOCS.toml) |
| Release procedure | [`RELEASE.md`](RELEASE.md) |

Source and behavior outrank prose when an authority has drifted. Correct the
authority and its generated projections in the same change.

## Lifecycle classes

- `current`: active normative guidance.
- `generated`: a projection of the source named in `DOCS.toml`.
- `superseded`: retained decision history replaced by a current authority.
- `archived`: historical evidence only.

Only `current` and `generated` pages appear in `SUMMARY.md`. Superseded and
archived pages cannot define current behavior.

## Page contract

Every `DOCS.toml` page declares:

- an audience: `user`, `extension`, `contributor`, or `release`;
- one owner from the manifest owner registry;
- a document kind and reader-task navigation section;
- one authority source;
- `manual` or `generated` ownership;
- a generator when the page is generated.

Generated pages cannot own their input facts. Manual pages do not duplicate
source-derived inventories. Public and extension pages describe externally
observable behavior and supported extension seams. Execution queues, local
planning paths, agent/worktree procedure, and numbered migration phases remain
in contributor governance or historical evidence.

Only Git-tracked Markdown is publishable. Ignored and untracked review files do
not enter the lifecycle manifest or mdBook.

Active navigation is ordered by reader task: authority, architecture,
lifecycle and extension contracts, optimization, user workflows, reference,
testing and conformance, then performance and release.

## Generated outputs

`SUMMARY.md` and `INDEX.md` are generated from `DOCS.toml` by
`scripts/docs_manifest.py`. `INDEX.md` reports Cargo-derived package and shipped
target counts plus every page's audience, owner, authority, kind, and generator.
Edit the manifest or named source, then regenerate the outputs.
