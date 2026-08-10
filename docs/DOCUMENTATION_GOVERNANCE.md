# Documentation Governance

## Authorities

| Contract | Authority |
| --- | --- |
| Executable compiler and artifact lifecycle | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Crate ownership and allowed production dependencies | [`CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml) |
| Optimization layers, owners, and benchmark targets | [`optimization/README.md`](optimization/README.md) |
| Documentation lifecycle and navigation | [`DOCS.toml`](DOCS.toml) |
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

Only Git-tracked Markdown is publishable. Ignored and untracked review files do
not enter the lifecycle manifest or mdBook.

## Generated outputs

`SUMMARY.md` and `INDEX.md` are generated from `DOCS.toml` by
`scripts/docs_manifest.py`. `INDEX.md` reports Cargo-derived package and shipped
target counts. Edit the manifest or named source, then regenerate the outputs.
