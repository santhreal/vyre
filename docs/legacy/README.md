# Legacy documents

Nothing in this directory describes how vyre works today. Do not follow the
instructions in these files, and do not treat their claims as current.

These three documents were written on 2026-05-01 against a crate layout that no
longer exists. They are kept because they record the reasoning behind the
separation work that followed, not because their conclusions still hold. The
backlog items they list are either done or superseded.

For current documentation, start at [`../INDEX.md`](../INDEX.md). For the
architecture as built, read [`../ARCHITECTURE.md`](../ARCHITECTURE.md). For work
still planned, read
[`../../audits/RELEASE_GATE.md`](../../audits/RELEASE_GATE.md), which is the
published release gate and execution backlog. The repository-root roadmap and
backlog files are maintainer working documents excluded from publication, so they
are named here and not linked; nothing on this page requires them.

## What is here

One of the three is published and you can open it:

- [`PERF_ROADMAP_2026-05-01.md`](PERF_ROADMAP_2026-05-01.md), a performance
  roadmap snapshot, superseded by the published release gate above.

Two are retained in the maintainer's working copy and excluded from the
repository, because `.gitignore` keeps files named `*BACKLOG*` and `*AUDIT*` out
of public history. They are named so the record is not hidden from you, and not
linked so that no link here dies the moment you clone:

- `CC_OWNED_BACKLOG_2026-05-01.md`, a backlog snapshot, superseded by the
  published release gate above.
- `SEPARATION_AUDIT_2026-05-01.md`, the audit that motivated the crate split
  described in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
