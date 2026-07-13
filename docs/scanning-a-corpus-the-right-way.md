# Scanning a corpus the right way

The fast-path scan APIs, resident sessions, async batches, timed attribution,
and count-then-collect, are the ones consumers should reach for, and they are
the least obvious from the type list. This guide is the intended route from "I
have a corpus and a pattern set" to "the GPU is doing the least possible work per
scan." Every signature below is the current public surface of
`vyre_libs::scan::GpuLiteralSet` and its resident/async twins.

The one-line rule: **build the matcher once, keep the immutable tables resident,
overlap batches, and let the device count, never re-upload tables, never page
matches on the host.**

---

## 0. Which API do I want?

| Situation | Use | Why |
|---|---|---|
| One corpus, scanned once, exits | `scan_all` | Count-then-collect: complete match set, no cap tuning, common case is one dispatch |
| A hot loop re-scanning batch after batch | `prepare_resident_scan` → `ResidentLiteralScan::scan_into` | The DFA + prefilter tables upload **once**; each scan re-stages only the haystack |
| Per-region presence **and** positions in one launch | `prepare_resident_fused_scan` → `ResidentFusedRegionScan::scan_into` | One 14-binding dispatch produces both outputs; tables still upload once |
| Two or more batches you can pipeline | `scan_into_async` → `PendingMatches::await_into` | Batch _n+1_'s upload overlaps batch _n_'s device execution |
| You need the kernel-vs-staging split | any `*_timed` twin → `TimedDispatchResult` | `wall_ns` / `device_ns` / `enqueue_ns` / `wait_ns`, cheap enough to leave on |

The single worst thing a consumer can do is call the borrowed one-shot `scan`
in a loop: it re-uploads every immutable table on every call. If you scan more
than once, you want a resident session.

---

## 1. Build the matcher once

```rust
use vyre_libs::scan::GpuLiteralSet;

// Infallible for a valid pattern set; `try_compile` returns the structured
// LiteralSetCompileError if you want to handle an oversized/empty set.
let matcher = GpuLiteralSet::compile(patterns);
```

Compilation builds the Aho–Corasick DFA and the candidate prefilter masks. It is
a one-time cost, never rebuild it per batch. If a corpus arrives mixed-case and
you would otherwise lowercase it on the host, compile with
`GpuLiteralSet::compile_case_insensitive` instead and scan the raw bytes: the
fold happens in the transition table, not with a second haystack copy.

---

## 2. Count-then-collect for a one-shot scan

`scan_all` dispatches at a default capacity; the match kernel's atomic counter
reports the **true** total even past that capacity, so on saturation the output
is resized to exactly that count and re-dispatched once. The common case is a
single dispatch; a saturated chunk is exactly two. It never silently truncates 
you get the complete set or a structured `BackendError`.

```rust
// The complete match set, no cap to tune, no host paging loop.
let matches = matcher.scan_all(backend, haystack)?;

// Or decode into a buffer you reuse across corpora:
let mut matches = Vec::new();
matcher.scan_all_into(backend, haystack, &mut matches)?;
```

Reach for the fixed-cap `scan` / `scan_into` only when you have a hard upper
bound and _want_ the fail-closed overflow error as a signal.

---

## 3. Keep the tables resident for a hot loop

A resident session uploads the seven immutable tables once at `prepare` time.
Each scan then re-stages only the haystack, a 4-byte counter reset, and the
haystack length (the table re-upload is gone).

```rust
// Size the session for the largest haystack and match count you expect.
let session = matcher.prepare_resident_scan(
    backend,
    haystack_capacity_bytes, // e.g. corpus_len + slack
    max_matches,
)?;

let mut matches = Vec::new();
let mut scratch = Vec::new(); // reused packed-haystack staging
for batch in corpus_batches {
    session.scan_into(backend, batch, &mut matches, &mut scratch)?;
    consume(&matches);
}
session.free(backend)?;
```

`scan_into` fails closed if a batch exceeds the resident capacity, or if the
device match count exceeds `max_matches`: never a silent truncated decode. Size
`max_matches` from your corpus, or use `scan_all` when the count is unbounded.

### Presence + positions in one launch

If you scan a coalesced corpus of many files (regions) and need both the
per-region literal-presence bitmap **and** the positioned matches, the fused
resident session produces both in one dispatch:

```rust
let session = matcher.prepare_resident_fused_scan(
    backend,
    haystack_capacity_bytes,
    max_regions,
    max_matches,
)?;

let mut presence = Vec::new(); // per-region presence bitmap words
let mut matches = Vec::new();  // (pattern_id, start, end) triples
let mut scratch = Vec::new();
session.scan_into(
    backend,
    haystack,
    region_starts, // region i spans region_starts[i]..region_starts[i+1]
    /* region_base */ 0,
    &mut presence,
    &mut matches,
    &mut scratch,
)?;
session.free(backend)?;
```

`region_starts` is the array of region start offsets; the first must be `0`. The
kernel reads the live region count from the array length, so one session sized
for the full corpus also serves smaller sub-batches under the cap.

### A corpus larger than one window

When the corpus is bigger than a single resident window will hold (a >4 GiB scan,
or a device budget smaller than the corpus), do **not** hand-split it into batches
and stitch the results yourself. `scan_paged_fused` pages it for you, into
byte-budgeted windows at file boundaries, and returns one unified result in a
single global region numbering with u64 positions, identical to a single-shot scan
of the concatenated corpus:

```rust
use vyre_libs::scan::{scan_paged_fused, PagedScanResult};

let files: &[&[u8]] = /* one slice per region/file */;
let result: PagedScanResult = scan_paged_fused(
    &matcher,
    backend,
    files,
    window_budget_bytes, // host RSS is bounded by this, not by the corpus
    max_matches,
)?;
// result.presence, per global region; result.matches. GlobalMatch { region_id, start:u64, end:u64, .. }
```

Boundary matches are handled correctly (a literal spanning two windows is found in
the window its start falls in, no miss, no double count), so you get the same
answer as a single contiguous scan while host memory stays bounded by one window.
For throughput, `scan_paged_fused_async` pipelines the windows (window *k+1*'s
staging overlaps window *k*'s device execution) with a bit-identical result.

---

## 4. Overlap batches with the async twins

Every single-dispatch entry point has a `PendingDispatch` twin. Submit the next
batch before awaiting the previous one, and its host staging / upload overlaps
the first batch's device execution.

```rust
// Both in flight before either is awaited.
let pending_a = matcher.scan_into_async(backend, batch_a, max_matches)?;
let pending_b = matcher.scan_into_async(backend, batch_b, max_matches)?;

let mut matches_a = Vec::new();
let mut matches_b = Vec::new();
pending_a.await_into(&mut matches_a)?; // or await_matches() -> Vec<Match>
pending_b.await_into(&mut matches_b)?;
```

The async twin keeps the cap's fail-closed contract: an overflow past
`max_matches` is an error at await, never a silent partial. `scan_all` is
deliberately sync-only, its 2-dispatch count-then-resize loop cannot be a
well-typed fire-and-forget handle, so an async caller submits `scan_into_async`
at a fixed cap and, on its fail-closed overflow, falls back to a synchronous
`scan_all`.

The fused path has an async twin too:
`scan_presence_and_positions_by_region_async(...)` returns a `PendingFusedRegion`
whose `await_into(&mut matches)` decodes **both** outputs (it returns the
presence words and fills the match triples).

---

## 5. Leave timing on

Every dispatch path has a `*_timed` twin returning a `TimedDispatchResult`
(`wall_ns`, `device_ns`, `enqueue_ns`, `wait_ns`). It is cheap enough to leave on
in production, and it is how you tell a kernel regression from a staging
regression.

```rust
let timed = session.scan_into_timed(backend, batch, &mut matches, &mut scratch)?;
// device_ns is a loud `None` on a backend without a device timer, never a
// fabricated zero. The staging cost is wall_ns not spent in device_ns.
if let Some(device_ns) = timed.device_ns {
    let staging_ns = timed.wall_ns.saturating_sub(device_ns);
    record(device_ns, staging_ns);
}
```

The end-to-end claim vyre must win is "faster than the best CPU path, staging
included", so the split above is the number that matters, not the kernel time
alone. The `scan.literal_set.vs_cpu_aho_corasick` and
`scan.literal_set.decode_heavy` bench cases measure exactly this split on
consumer-shaped corpora.

---

## Putting it together

The fast path, in one shape:

1. **Compile once**: `GpuLiteralSet::compile` (or `_case_insensitive`).
2. **Prepare resident**: `prepare_resident_scan` / `prepare_resident_fused_scan`
   so tables upload once.
3. **Async batches**: `scan_into_async`, two in flight, to overlap staging with
   compute.
4. **Timed attribution**: the `*_timed` twins, left on, to keep the
   kernel-vs-staging split honest.
5. **Count-then-collect**: `scan_all` when the match count is unbounded, so no
   consumer ever pages matches on the host.

Anything that re-uploads a table per scan, folds the whole corpus on the host, or
pages matches by hand is leaving the win on the floor.
