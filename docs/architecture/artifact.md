# The artifact is the output type

An `Artifact` is a validated whole-graph plan with its ABI and liveness,
device-neutral, identified by a digest. It is the module, not the machine
code. Every production compile emits one.

Two compiles of the same validated request produce the same digest. That is
what makes an artifact cache sound and what makes two routes comparable: if
two paths disagree, the digest says whether they were even compiling the
same thing.

## Persistence is a schedule, not a second output

One resident kernel that never returns to the host is a schedule the
compiler may select. It is recorded inside the artifact. It is not a second
output type and there is no route that produces it directly.

That matters because persistence wins only sometimes. It wins when the
graph is long enough that launch and materialization dominate, when
occupancy survives fusion, when the device supports the required
synchronization, and when keeping weights and scratch resident pays for
itself. It loses on a two-node graph, on a fusion that collapses occupancy
through register or shared-memory pressure, and on a device with no
device-wide barrier, where the correct plan is a sequence of dispatches.

Static and persistent routes consume the same artifact class and must
produce the same bytes for the same inputs. There is no side door that
emits a one-off kernel and skips the planner.

`vyre-runtime` executes the persistence the artifact selected. It does not
decide whether to be persistent. A runtime that decided that would be
taking the decision away from the search that could have measured it.

## Identity is neutral; payloads are not

The same semantic plan digests the same everywhere. A `TargetPayload`
carries dialect bytes plus the geometry that won on one device profile, so
it does not.

A payload names its `TargetProfile`: a stable identity, a positive
generation, per-axis workgroup limits, an invocation limit, a dynamic
shared-memory limit, and a subgroup width that is zero or a power of two. A
zero identity, a zero generation, a zero limit or a non-power-of-two
subgroup width is rejected when the profile is built.

Admission then checks every entry point against that profile: a workgroup
extent past a per-axis limit, an invocation count past the invocation limit,
or a dynamic shared-memory requirement past the shared-memory limit is
rejected as a malformed payload. The corrective action is to emit geometry
the authenticated profile admits, not to relax the profile.

A payload also carries the digest of the neutral artifact it was compiled
from, and its own body digest is recomputed and compared on admission. A
payload that has been edited, or that names a different artifact, does not
admit.

## What consumes the artifact

Device admission, materialization, submission, queues, residency and
recovery are consumers. None of them alters artifact identity. See
[run an artifact on a device](../guide/backends.md) for the route and
[add a backend](../extending/backend.md) for the contract a driver
implements.
