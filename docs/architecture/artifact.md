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

Schema 9 authenticates the versioned backend-neutral phase schedule, transform
preconditions, inverse/source provenance, exact phase geometry and bounded
resources. Decode rejects malformed schedules before physical lowering.

Schema 8 authenticates the selected execution topology and its bounded search
accounting. Decode rejects zero queue or resident-partition cardinalities,
inconsistent candidate counts, work beyond the recorded budget, and measured
plans without positive launch and device-time evidence.

Schema 7 authenticates each entry input and output as a Program buffer name
paired with its canonical graph value. It also records retained-predecessor
lineage. Materializers resolve active buffers through those named records and
inactive split-segment declarations through exact target `(group, slot)`
metadata. A missing, ambiguous or unrelated identity rejects the payload.

## One plan owns physical storage

Schedule selection produces one versioned allocation and layout plan, recorded
in the artifact. It maps every canonical value to a region: device slot, address
space, owner, offset, bytes, alignment and padding. Each placement inside a
region states the value's byte offset, packed size, lifetime, alias class, live
stage range, whether an ordering effect touches it, its element layout, and the
storage operations permitted for it: reuse of another value's dead storage,
in-place update, rematerialization, spill and prefetch.

The plan reports the per-device liveness peak and the aggregate across devices
before candidates are ranked, and candidate ranking prices peak memory from that
figure. A compile whose artifact states a peak the ranking did not price is
refused.

Constant-lifetime values are addressed in the constant space and every other
value in the device space; the artifact never allocates constant storage. Two
values may share bytes only when their alias classes match or their live ranges
are disjoint.

Lowering verifies its own bindings against the plan. A bound value the plan
places nowhere, a group binding a value outside its live range, and a
constant-space placement bound writable are rejected before emission.

`vyre-runtime` allocates one buffer per artifact-owned region, in recorded
order, and binds every placement to it. It does not pack, resize, merge,
reorder, or discover reuse.

A backend that reports the device bytes it holds reconciles that figure against
the planned peak before a measurement is spent. Fewer bytes than the plan
requires is refused as `MKC041_UNRECONCILED_RESIDENT_BYTES`. A backend with no
memory query reports zero, which leaves the planned figure unreconciled.

## What consumes the artifact

Device admission, materialization, submission, queues, residency and
recovery are consumers. None of them alters artifact identity. See
[run an artifact on a device](../guide/backends.md) for the route and
[add a backend](../extending/backend.md) for the contract a driver
implements.
