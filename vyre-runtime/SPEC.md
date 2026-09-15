# vyre-runtime

Layer `runtime`. Owner `runtime`.

## Owns

Execute the artifact's selected persistence: sessions, recovery, residency,
scheduling, caches, telemetry, readback, and IO. Does not decide whether to be
persistent.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-runtime).

## Must never contain

Compilation, and any decision about whether to be persistent. Persistence is
selected during compile, inside the artifact; this crate executes that
selection. A runtime that decides to fuse has taken a decision away from the
search that could have measured it.

## What crosses its edges

Out of this crate, into:

- `vyre-libs` over the `product-libraries` seam, private: composition trees the
  megakernel planner plans against. Built when: always.
- `vyre-driver` over the `backend-contract` seam, public: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-megakernel` over the `megakernel-compiler` seam, public: whole-graph
  compilation and immutable artifact contracts. Built when: always.

Into this crate, from:

- `vyre` over the `runtime` seam, private.
- `vyre-bench` over the `runtime` seam, private.
- `vyre-conform` over the `runtime` seam, private.

## Direction that may not reverse

`vyre-libs`, `vyre-driver`, `vyre-foundation`, and `vyre-megakernel` must
never depend on `vyre-runtime`. The edge is one way: a
cycle back into this crate makes the two crates one crate that cannot be built,
reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 5 features beyond `default`: `libs-compositions`,
  `megakernel-batch`, `remote-cache`, `subgroup-ops`, `uring-cmd-nvme`. Each
  builds alone.
- The public surface is recorded in `docs/public-api/vyre-runtime.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
