# registry

One directory declares every runnable and every placement fact in this
workspace. A gate reads a declaration; no gate restates one.

One declared thing is one file, named after it. Adding a lane, a script or a
crate means adding a file. No file enumerates the set: a consumer scans the
directory, so a workflow, a script or a member with no file is a finding rather
than a silence.

## Directories

`ci/` holds one file per workflow job. A row names the workflow file, the job,
the required status context, the triggers, and the ordered steps the job runs.
A step is a gate id, a gate-sweep subset, a declared script, or a command that
is none of those. A job whose steps disagree with the workflow on disk is a
finding in both directions.

`scripts/` holds one file per script, on disk or deleted. `state` is one of:

- `delegates`: the file exists and does nothing but invoke the gate in `gate`.
- `implements`: a gate or a generator runs the file to do the work, and `gate`
  names it.
- `operator`: a human runs the file; it is not part of any gate.
- `holds_logic`: the file carries checks of its own that belong in a gate.
  `destination` names where they go.
- `retired`: the file is deleted, and `gate` or `reason` records where its
  checks went.

`crates/` holds one file per workspace member: its concern, its pages, its test
placement, and the members it depends on. `depends_on` is checked against the
manifest and against the direction the boundary allows.

## Editing

Edit the file for the thing that changed. A row is data a gate reads, so a
value that no longer describes the tree fails the gate that reads it.
