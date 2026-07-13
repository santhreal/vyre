//! W2-4: paged corpus planning, turn a corpus larger than one resident window
//! into a deterministic sequence of resident-window dispatches with STABLE GLOBAL
//! region ids, so a >4 GiB (or over-budget) scan is a driver concern, not a
//! consumer-side batch-splitting exercise.
//!
//! The host-paging model (not the kernel's u32 shard offset). The fused scan's
//! `region_base` argument is a u32 GLOBAL BYTE offset, so it can only shard a
//! corpus that fits u32 (≤ 4 GiB) in one region-id space. To scan a corpus LARGER
//! than that, this driver runs each window as an INDEPENDENT local scan
//! (`region_base = 0`, window-relative `region_starts`) and globalizes on the HOST
//! to u64 positions and to global region ids (= original file indices). Every
//! window's byte budget stays ≤ u32, but the corpus as a whole is unbounded, and
//! host RSS is bounded by one window, not the corpus.
//!
//! [`plan_corpus_windows`] is the pure host-side planner (total coverage, no gap or
//! overlap, monotone offsets, stable global ids, fail-closed on an unscannable
//! file); [`scan_paged_fused`] drives one resident fused session across the plan
//! (tables uploaded once) and globalizes each window's presence rows and positioned
//! matches; [`scan_paged_fused_async`] pipelines the windows (staging of window
//! `k+1` overlaps device execution of window `k`) via the borrowed async fused
//! dispatch, sharing the exact same globalization so its result is identical; and
//! [`scan_paths_paged`] is the disk-backed driver, it takes file PATHS and reads
//! only one window's files into memory at a time (host RSS bounded by the window,
//! not the corpus), sharing the same globalization so its result equals the
//! in-memory scan. The planner is proven independently because its correctness is a
//! pure function of the lengths and the budget; the drivers' globalization is proven
//! equal to a single-shot scan (sync), to each other (async == sync), and across the
//! memory/disk boundary (disk == in-memory).

use std::collections::VecDeque;
use std::io::Read;
use std::ops::Range;
use std::path::Path;

use vyre::VyreBackend;
use vyre_foundation::match_result::Match;

use crate::scan::literal_set::{GpuLiteralSet, PendingFusedRegion};

/// One resident-window worth of a paged corpus: a maximal contiguous run of files
/// whose combined bytes fit the window budget (or a single over-budget file on its
/// own), plus the global region id of the run's first file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CorpusWindow {
    /// Global file indices covered by this window, `[start, end)`. `start` is also
    /// the window's `global_region_base` (region id == original file index).
    pub(crate) file_range: Range<usize>,
    /// Global byte offset of this window's first byte within the whole corpus 
    /// added to a window-local match `start`/`end` to globalize it.
    pub(crate) byte_offset: u64,
    /// Total bytes in this window (== the window's haystack length). Never exceeds
    /// the budget unless the window holds a single over-budget file (then it is
    /// exactly that file's length, and [`plan_corpus_windows`] flags it).
    pub(crate) byte_len: usize,
    /// The `region_base` argument for the fused scan of this window: the global
    /// region id of `file_range.start`, so per-region presence rows and region
    /// attribution stay in the original file numbering.
    pub(crate) global_region_base: u32,
}

impl CorpusWindow {
    /// Window-local start offset of global file `file_index` (which must lie in
    /// `file_range`), given the corpus's per-file lengths. The paging driver uses
    /// this to build the window's local `region_starts` (first == 0).
    pub(crate) fn local_region_starts(&self, file_lengths: &[usize]) -> Vec<u32> {
        let mut starts = Vec::with_capacity(self.file_range.len());
        let mut offset = 0u32;
        for &len in &file_lengths[self.file_range.clone()] {
            starts.push(offset);
            // Bounded by construction: the window's total bytes fit u32.
            offset = offset.saturating_add(len as u32);
        }
        starts
    }
}

/// Why a corpus could not be planned into resident windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PagedCorpusError {
    /// The window budget was zero, so no file could ever be placed.
    ZeroBudget,
    /// A single file is larger than the u32 haystack ABI can address, so it cannot
    /// be scanned in one dispatch even alone.
    FileExceedsHaystackAbi { file_index: usize, len: usize },
    /// The corpus has more files (regions) than the u32 region-id ABI allows.
    TooManyRegions { count: usize },
}

impl std::fmt::Display for PagedCorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBudget => write!(
                f,
                "paged_corpus: window budget is 0 bytes, so no file fits. Fix: pass a window_budget_bytes at least as large as the largest file (and within the resident haystack capacity)."
            ),
            Self::FileExceedsHaystackAbi { file_index, len } => write!(
                f,
                "paged_corpus: file {file_index} is {len} bytes, larger than the u32 haystack ABI can address in one dispatch. Fix: pre-split that file into sub-{ceiling}-byte regions before paging.",
                ceiling = u32::MAX
            ),
            Self::TooManyRegions { count } => write!(
                f,
                "paged_corpus: corpus has {count} files but the region-id ABI is u32. Fix: coalesce fewer files per corpus or shard the corpus."
            ),
        }
    }
}

impl std::error::Error for PagedCorpusError {}

/// Plan a corpus (given each file's byte length, in coalesced order) into a
/// sequence of resident windows of at most `window_budget_bytes` each.
///
/// Each window is a MAXIMAL contiguous run of files fitting the budget. A single
/// file larger than the budget gets its own window (a literal match must see
/// contiguous bytes, so a file is never split across windows), its `byte_len`
/// then exceeds the budget, which the caller must accommodate by sizing the
/// resident haystack capacity to `max(byte_len)`.
///
/// # Errors
/// - [`PagedCorpusError::ZeroBudget`] if `window_budget_bytes == 0`.
/// - [`PagedCorpusError::FileExceedsHaystackAbi`] if any file exceeds `u32::MAX`
///   bytes (unscannable in one dispatch).
/// - [`PagedCorpusError::TooManyRegions`] if the file count exceeds `u32::MAX`.
pub(crate) fn plan_corpus_windows(
    file_lengths: &[usize],
    window_budget_bytes: usize,
) -> Result<Vec<CorpusWindow>, PagedCorpusError> {
    if window_budget_bytes == 0 {
        return Err(PagedCorpusError::ZeroBudget);
    }
    if file_lengths.len() > u32::MAX as usize {
        return Err(PagedCorpusError::TooManyRegions {
            count: file_lengths.len(),
        });
    }
    // Fail closed on any unscannable file BEFORE planning (Law 10, never silently
    // drop or truncate a file that cannot be dispatched).
    for (file_index, &len) in file_lengths.iter().enumerate() {
        if len > u32::MAX as usize {
            return Err(PagedCorpusError::FileExceedsHaystackAbi { file_index, len });
        }
    }

    let mut windows = Vec::new();
    let mut file_start = 0usize;
    let mut byte_offset = 0u64;

    while file_start < file_lengths.len() {
        let mut file_end = file_start;
        let mut window_bytes = 0usize;
        // Grow the window while the NEXT file still fits the budget. Always take at
        // least one file (even an over-budget one (so progress is guaranteed)).
        while file_end < file_lengths.len() {
            let next = file_lengths[file_end];
            let grown = window_bytes.saturating_add(next);
            if file_end > file_start && grown > window_budget_bytes {
                break;
            }
            window_bytes = grown;
            file_end += 1;
        }

        windows.push(CorpusWindow {
            file_range: file_start..file_end,
            byte_offset,
            byte_len: window_bytes,
            // file_start <= u32::MAX (checked via file count above).
            global_region_base: file_start as u32,
        });

        byte_offset = byte_offset.saturating_add(window_bytes as u64);
        file_start = file_end;
    }

    Ok(windows)
}

/// A positioned match in GLOBAL corpus coordinates: u64 byte positions (so a
/// corpus larger than the u32 haystack ABI is representable) and the global region
/// id (== the original file index the match starts in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalMatch {
    /// Pattern id, in the compiled matcher's numbering.
    pub pattern_id: u32,
    /// Global region id (the original file index the match's START falls in).
    pub region_id: u32,
    /// Global byte start of the match (u64, unbounded by the per-window u32 ABI).
    pub start: u64,
    /// Global byte end of the match (exclusive).
    pub end: u64,
}

/// The unified result of a paged fused scan, in a single global numbering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedScanResult {
    /// `region_count * presence_words` u32s; region `r` occupies
    /// `presence[r*presence_words .. (r+1)*presence_words]`, bit `pid` set iff
    /// pattern `pid` occurs in region `r`.
    pub presence: Vec<u32>,
    /// Number of global regions (== `files.len()`).
    pub region_count: u32,
    /// Presence bitmap words per region.
    pub presence_words: u32,
    /// Every positioned match in global coordinates, sorted by
    /// `(region_id, start, end, pattern_id)`.
    pub matches: Vec<GlobalMatch>,
}

/// Aggregated timing attribution for a paged scan (W3-3 "attribution everywhere"
/// extended to the W2-4 paging path). Every field is a HONEST aggregate over the
/// per-window dispatches (never a fabricated value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedScanTiming {
    /// Number of window dispatches the corpus was paged into.
    pub windows: u32,
    /// Total OWN bytes scanned (excludes per-window `L-1` overlap tails, so this is
    /// the true corpus size and a valid throughput denominator).
    pub bytes_scanned: u64,
    /// Sum of every window's wall-clock dispatch time (host-observed).
    pub wall_ns: u64,
    /// Sum of every window's device (kernel) time, or `None` if ANY window ran on a
    /// backend without a device timer, the absence is surfaced loudly, never
    /// fabricated as 0 (Law 10). `Some` only when every window reported device time.
    pub device_ns: Option<u64>,
}

/// Per-shard timing for one device in a sharded scan, the `per-shard-active-ns`
/// signal the W3-5 `load_balance_policy` rebalances the next batch on. An operator
/// (or an auto-balancer) reads these to see which device carried the most byte-work
/// and how long it took, then feeds new `weights` into
/// [`scan_sharded_fused_weighted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardTiming {
    /// Index of this shard in the backend set.
    pub shard: u32,
    /// Windows assigned to this shard.
    pub windows: u32,
    /// Own bytes (overlap excluded) scanned on this shard (its byte-work share).
    pub bytes_scanned: u64,
    /// Summed wall-clock dispatch time across this shard's windows.
    pub wall_ns: u64,
    /// Summed device (kernel) time across this shard's windows, or `None` if any of
    /// them lacked a device timer (loud absence, never a fabricated 0. Law 10). A
    /// shard that received no windows reports `Some(0)`.
    pub device_ns: Option<u64>,
}

/// The per-shard timing breakdown of a sharded scan: one [`ShardTiming`] per backend
/// in the set, in device-set order. The load imbalance across these entries is the
/// evidence for reweighting a subsequent batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedScanTiming {
    /// One entry per backend, in the order the backends were passed.
    pub shards: Vec<ShardTiming>,
}

/// The host-side staging for one window: the haystack (own bytes + the `L-1`-byte
/// overlap tail), the `region_starts` (own regions + a dummy overlap region), and
/// the metadata needed to globalize the window's results. Shared by the sync and
/// async drivers so their boundary handling cannot drift (ONE PLACE).
struct WindowStaging {
    haystack: Vec<u8>,
    region_starts: Vec<u32>,
    own_len: usize,
    own_region_count: usize,
    byte_offset: u64,
    global_region_base: u32,
}

/// Overlap = longest pattern minus one: exactly enough tail context for any match
/// that STARTS in this window to complete, never more.
fn paging_overlap(matcher: &GpuLiteralSet) -> usize {
    matcher
        .pattern_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .saturating_sub(1) as usize
}

/// Assemble one window's haystack (own files ++ up to `overlap` bytes of the
/// following files) and its `region_starts` (own file starts ++ a dummy region at
/// `own_len` that absorbs any match landing in the overlap tail).
fn stage_window(
    files: &[&[u8]],
    file_lengths: &[usize],
    window: &CorpusWindow,
    overlap: usize,
) -> WindowStaging {
    let own_len = window.byte_len;
    let mut haystack = Vec::with_capacity(own_len + overlap);
    for file in &files[window.file_range.clone()] {
        haystack.extend_from_slice(file);
    }
    let mut gathered = 0usize;
    for file in &files[window.file_range.end..] {
        if gathered >= overlap {
            break;
        }
        let take = (overlap - gathered).min(file.len());
        haystack.extend_from_slice(&file[..take]);
        gathered += take;
    }
    let (region_starts, own_region_count) =
        window_region_starts(window, file_lengths, gathered > 0);
    WindowStaging {
        haystack,
        region_starts,
        own_len,
        own_region_count,
        byte_offset: window.byte_offset,
        global_region_base: window.global_region_base,
    }
}

/// The window's `region_starts` (own file starts ++ a dummy region at `own_len`
/// when there is an overlap tail) and its own region count. Shared by the
/// in-memory and disk staging paths so the boundary layout cannot drift.
fn window_region_starts(
    window: &CorpusWindow,
    file_lengths: &[usize],
    has_overlap: bool,
) -> (Vec<u32>, usize) {
    let mut region_starts = window.local_region_starts(file_lengths);
    let own_region_count = region_starts.len();
    if has_overlap {
        region_starts.push(window.byte_len as u32);
    }
    (region_starts, own_region_count)
}

/// Fold one window's raw fused-scan outputs into the global result: keep the own
/// regions' presence rows (drop the dummy overlap row), and globalize the matches
/// (drop those starting in the overlap tail, the next window owns them; add the
/// window byte offset for u64 global positions; attribute the region by start).
/// Presence words-per-region for one window's raw fused output, validated to divide
/// evenly by the window's region count. The single owner of that invariant, shared by
/// the sequential [`globalize_window`] and the parallel per-shard globalization (ONE
/// PLACE) so a presence-shape drift is caught identically on both paths.
fn window_presence_words(
    win_presence: &[u32],
    scan_region_count: usize,
) -> Result<usize, vyre::BackendError> {
    if scan_region_count == 0 || win_presence.len() % scan_region_count != 0 {
        return Err(vyre::BackendError::new(format!(
            "scan_paged_fused: window presence length {} is not a multiple of its region count {scan_region_count}. Fix: internal invariant broke, report with the corpus shape.",
            win_presence.len()
        )));
    }
    Ok(win_presence.len() / scan_region_count)
}

/// Map one window's LOCAL matches into GLOBAL `(region_id, byte-offset)` space and
/// append them to `out`: drop any match starting in the overlap tail (the next window
/// owns it), add the window byte offset for u64 global positions, and attribute the
/// region by start. The single owner of the local→global match transform, shared by
/// the sequential and parallel sharded globalization (ONE PLACE).
fn map_window_matches(staging: &WindowStaging, win_matches: &[Match], out: &mut Vec<GlobalMatch>) {
    let own_starts = &staging.region_starts[..staging.own_region_count];
    for hit in win_matches {
        if (hit.start as usize) >= staging.own_len {
            continue; // starts in the overlap tail, the next window owns it
        }
        let local_region = own_starts
            .partition_point(|&start| start <= hit.start)
            .saturating_sub(1);
        out.push(GlobalMatch {
            pattern_id: hit.pattern_id,
            region_id: staging.global_region_base + local_region as u32,
            start: staging.byte_offset + u64::from(hit.start),
            end: staging.byte_offset + u64::from(hit.end),
        });
    }
}

fn globalize_window(
    staging: &WindowStaging,
    win_presence: &[u32],
    win_matches: &[Match],
    presence_words: &mut usize,
    presence: &mut Vec<u32>,
    matches: &mut Vec<GlobalMatch>,
) -> Result<(), vyre::BackendError> {
    let words = window_presence_words(win_presence, staging.region_starts.len())?;
    if *presence_words == 0 {
        *presence_words = words;
    } else if words != *presence_words {
        return Err(vyre::BackendError::new(format!(
            "scan_paged_fused: presence word count changed across windows ({} -> {words}). Fix: internal invariant broke, report with the corpus shape.",
            *presence_words
        )));
    }
    presence.extend_from_slice(&win_presence[..staging.own_region_count * words]);
    map_window_matches(staging, win_matches, matches);
    Ok(())
}

fn finish_result(
    presence: Vec<u32>,
    region_count: u32,
    presence_words: usize,
    mut matches: Vec<GlobalMatch>,
) -> PagedScanResult {
    matches.sort_unstable_by_key(|hit| (hit.region_id, hit.start, hit.end, hit.pattern_id));
    PagedScanResult {
        presence,
        region_count,
        presence_words: presence_words as u32,
        matches,
    }
}

/// Common prologue: the global region count, per-file lengths, and the window plan.
/// Returns `None` (an empty result) when the corpus is empty.
fn plan_paged(
    files: &[&[u8]],
    window_budget_bytes: usize,
) -> Result<Option<(u32, Vec<usize>, Vec<CorpusWindow>)>, vyre::BackendError> {
    let region_count = u32::try_from(files.len()).map_err(|_| {
        vyre::BackendError::new(
            "scan_paged_fused: file count exceeds the u32 region-id ABI. Fix: coalesce fewer files per corpus or shard the corpus.".to_string(),
        )
    })?;
    let file_lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
    let windows = plan_corpus_windows(&file_lengths, window_budget_bytes)
        .map_err(|error| vyre::BackendError::new(error.to_string()))?;
    if windows.is_empty() {
        return Ok(None);
    }
    Ok(Some((region_count, file_lengths, windows)))
}

/// Scan a `files` corpus that may exceed one resident window, paging it into
/// windows of at most `window_budget_bytes`, and return the unified per-region
/// presence bitmap + positioned matches in one global numbering.
///
/// The result is IDENTICAL to a single-shot fused scan of the concatenated corpus:
/// each window carries `L-1` bytes of the next window as overlap (so a match
/// spanning a window boundary is found in exactly the window its start falls in),
/// the overlap bytes are attributed to a discardable dummy region (so they never
/// over-fire an own region's presence), and matches starting in the overlap are
/// dropped (found instead as the next window's own content), no double count, no
/// boundary miss (Law 10). This is the SYNCHRONOUS driver: it reuses ONE resident
/// fused session (tables uploaded once), scanning each window in turn.
///
/// # Errors
/// Returns [`vyre::BackendError`] on any window's dispatch/readback failure, if the
/// plan is invalid (see [`plan_corpus_windows`], surfaced as a backend error), or
/// if the file count exceeds the u32 region ABI.
pub fn scan_paged_fused(
    matcher: &GpuLiteralSet,
    backend: &dyn VyreBackend,
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    let Some((region_count, file_lengths, windows)) = plan_paged(files, window_budget_bytes)?
    else {
        return Ok(finish_result(Vec::new(), 0, 0, Vec::new()));
    };
    let overlap = paging_overlap(matcher);

    // ONE resident session sized for the largest window: own bytes + overlap, and
    // own regions + one dummy overlap region.
    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_len)
        .max()
        .unwrap_or(0);
    let max_window_regions = windows
        .iter()
        .map(|window| window.file_range.len())
        .max()
        .unwrap_or(0) as u32;
    let session = matcher.prepare_resident_fused_scan(
        backend,
        max_window_bytes + overlap + 64,
        max_window_regions + 1,
        max_matches,
    )?;

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    let mut win_presence: Vec<u32> = Vec::new();
    let mut win_matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();

    for window in &windows {
        let staging = stage_window(files, &file_lengths, window, overlap);
        session.scan_into(
            backend,
            &staging.haystack,
            &staging.region_starts,
            0,
            &mut win_presence,
            &mut win_matches,
            &mut scratch,
        )?;
        if let Err(error) = globalize_window(
            &staging,
            &win_presence,
            &win_matches,
            &mut presence_words,
            &mut presence,
            &mut matches,
        ) {
            let _ = session.free(backend);
            return Err(error);
        }
    }

    session.free(backend)?;
    Ok(finish_result(
        presence,
        region_count,
        presence_words,
        matches,
    ))
}

/// Timed twin of [`scan_paged_fused`]: identical result plus an aggregated
/// [`PagedScanTiming`] over the per-window dispatches (W3-3 attribution on the
/// paging path). It differs from the untimed driver in exactly one call 
/// `scan_into_timed` instead of `scan_into`: and reuses the same shared
/// `stage_window` / `globalize_window` / `finish_result` helpers, so the paged
/// result is byte-identical to the untimed driver (ONE PLACE). The device-time
/// aggregate is `Some` only when EVERY window reported device time; a single
/// timer-less window collapses it to a loud `None`, never a fabricated 0 (Law 10).
///
/// # Errors
/// Same as [`scan_paged_fused`].
pub fn scan_paged_fused_timed(
    matcher: &GpuLiteralSet,
    backend: &dyn VyreBackend,
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<(PagedScanResult, PagedScanTiming), vyre::BackendError> {
    let Some((region_count, file_lengths, windows)) = plan_paged(files, window_budget_bytes)?
    else {
        let timing = PagedScanTiming {
            windows: 0,
            bytes_scanned: 0,
            wall_ns: 0,
            device_ns: Some(0),
        };
        return Ok((finish_result(Vec::new(), 0, 0, Vec::new()), timing));
    };
    let overlap = paging_overlap(matcher);

    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_len)
        .max()
        .unwrap_or(0);
    let max_window_regions = windows
        .iter()
        .map(|window| window.file_range.len())
        .max()
        .unwrap_or(0) as u32;
    let session = matcher.prepare_resident_fused_scan(
        backend,
        max_window_bytes + overlap + 64,
        max_window_regions + 1,
        max_matches,
    )?;

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    let mut win_presence: Vec<u32> = Vec::new();
    let mut win_matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();

    // Honest aggregation: sum wall over every window; sum device only while every
    // window has reported one, a single `None` window makes the whole aggregate
    // `None` (no fabricated 0 for a missing timer).
    let mut wall_ns: u64 = 0;
    let mut device_ns_acc: Option<u64> = Some(0);
    let mut bytes_scanned: u64 = 0;

    for window in &windows {
        let staging = stage_window(files, &file_lengths, window, overlap);
        let timed = match session.scan_into_timed(
            backend,
            &staging.haystack,
            &staging.region_starts,
            0,
            &mut win_presence,
            &mut win_matches,
            &mut scratch,
        ) {
            Ok(timed) => timed,
            Err(error) => {
                let _ = session.free(backend);
                return Err(error);
            }
        };
        wall_ns = wall_ns.saturating_add(timed.wall_ns);
        device_ns_acc = match (device_ns_acc, timed.device_ns) {
            (Some(acc), Some(window_device)) => Some(acc.saturating_add(window_device)),
            _ => None,
        };
        bytes_scanned = bytes_scanned.saturating_add(window.byte_len as u64);
        if let Err(error) = globalize_window(
            &staging,
            &win_presence,
            &win_matches,
            &mut presence_words,
            &mut presence,
            &mut matches,
        ) {
            let _ = session.free(backend);
            return Err(error);
        }
    }

    session.free(backend)?;
    let timing = PagedScanTiming {
        windows: windows.len() as u32,
        bytes_scanned,
        wall_ns,
        device_ns: device_ns_acc,
    };
    Ok((
        finish_result(presence, region_count, presence_words, matches),
        timing,
    ))
}

/// Scan a `files` corpus split into byte-range window shards distributed across a
/// SET of backends, the W3-5 multi-GPU sharding architecture
/// (`MULTI_GPU_SHARDING_AGGREGATION_PLAN.toml`, `regex-haystack-byte-range-shards`).
/// Window `k` is assigned to `backends[k % backends.len()]` (round-robin), and each
/// backend holds its OWN resident fused session, so on a real multi-GPU host the
/// shards run concurrently on distinct peer devices.
///
/// The partition (byte-range windows), the halo (`L-1` overlap = the plan's
/// `overlap-boundary-by-max-pattern-lookbehind`), and the aggregation (host
/// globalize + stable sort by `(region, start, end, pattern_id)` = the plan's
/// `allgather-...-stable-sort-by-input-offset-rule-id-and-finding-span`) are the
/// SAME shared helpers as [`scan_paged_fused`] (ONE PLACE). So the sharded result
/// is byte-identical to a single-shot scan regardless of shard count or device
/// assignment (the `parity_policy` the plan mandates).
///
/// On a single-device host (a `backends` set of one device, or the same device
/// repeated) the shards run SEQUENTIALLY, which still exercises the full
/// partition / round-robin-assignment / per-shard-session / aggregation
/// architecture; only cross-device *parallelism* is unmodeled until a second
/// physical device is present.
///
/// # Errors
/// [`vyre::BackendError`] if `backends` is empty (fail closed, a shard scan needs
/// at least one device), or on any window's dispatch/readback/free failure.
pub fn scan_sharded_fused(
    matcher: &GpuLiteralSet,
    backends: &[&dyn VyreBackend],
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    scan_sharded_core(
        matcher,
        backends,
        files,
        window_budget_bytes,
        max_matches,
        None,
    )
    .map(|(result, _timing)| result)
}

/// Per-shard-timed twin of [`scan_sharded_fused`]: identical result plus a
/// [`ShardedScanTiming`] breaking wall/device time and byte-work down PER DEVICE 
/// the `per-shard-active-ns` signal the W3-5 `load_balance_policy` uses to reweight
/// the next batch. A run under equal round-robin whose shard timings are skewed is
/// the evidence that the devices differ in throughput; feed proportional `weights`
/// into [`scan_sharded_fused_weighted`] on the following batch to rebalance.
///
/// # Errors
/// Same as [`scan_sharded_fused`].
pub fn scan_sharded_fused_timed(
    matcher: &GpuLiteralSet,
    backends: &[&dyn VyreBackend],
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<(PagedScanResult, ShardedScanTiming), vyre::BackendError> {
    scan_sharded_core(
        matcher,
        backends,
        files,
        window_budget_bytes,
        max_matches,
        None,
    )
}

/// Throughput-WEIGHTED twin of [`scan_sharded_fused`]: assigns window shards so
/// each device receives byte-work proportional to its `weights[i]` (the plan's
/// `load_balance_policy` / `device-throughput-weight` partition), a faster device
/// gets more windows. Assignment is a deterministic greedy least-loaded-by-weight
/// pass over the windows, so it is reproducible and independent of device timing.
/// Because the aggregation is order-independent (globalize + stable sort), the
/// result is byte-identical to [`scan_sharded_fused`] and to a single-shot scan for
/// ANY weights (only the work DISTRIBUTION changes, never the answer (parity_policy)).
///
/// # Errors
/// [`vyre::BackendError`] if `backends` is empty, if `weights.len() != backends.len()`
/// (each device needs exactly one weight, fail closed rather than guess), or on any
/// window's dispatch/readback/free failure.
pub fn scan_sharded_fused_weighted(
    matcher: &GpuLiteralSet,
    backends: &[&dyn VyreBackend],
    weights: &[u32],
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    if weights.len() != backends.len() {
        return Err(vyre::BackendError::new(format!(
            "scan_sharded_fused_weighted: {} weights for {} backends. Fix: pass exactly one throughput weight per device backend.",
            weights.len(),
            backends.len()
        )));
    }
    scan_sharded_core(
        matcher,
        backends,
        files,
        window_budget_bytes,
        max_matches,
        Some(weights),
    )
    .map(|(result, _timing)| result)
}

/// Assign each window (by its own byte length) to a shard index. With `weights ==
/// None` the assignment is plain round-robin (`index % shard_count`). With weights,
/// it is a deterministic greedy pass that places each window on the shard whose
/// projected `assigned_bytes / weight` would be smallest (ties → lower index), so
/// cumulative byte-work tracks the weight vector. A zero weight is treated as `1`
/// so a device is never starved by a bad weight and never divides by zero. Pure and
/// host-side (the load-balancing logic is unit-tested without a GPU).
fn shard_assignment(
    window_byte_lens: &[usize],
    shard_count: usize,
    weights: Option<&[u32]>,
) -> Vec<usize> {
    match weights {
        None => (0..window_byte_lens.len())
            .map(|index| index % shard_count.max(1))
            .collect(),
        Some(weights) => {
            let mut assigned_bytes = vec![0u128; shard_count];
            let mut assignment = Vec::with_capacity(window_byte_lens.len());
            for &bytes in window_byte_lens {
                let mut best_shard = 0usize;
                let mut best_ratio = u128::MAX;
                for shard in 0..shard_count {
                    let weight = u128::from(weights.get(shard).copied().unwrap_or(1).max(1));
                    let projected = (assigned_bytes[shard] + bytes as u128) / weight;
                    if projected < best_ratio {
                        best_ratio = projected;
                        best_shard = shard;
                    }
                }
                assigned_bytes[best_shard] += bytes as u128;
                assignment.push(best_shard);
            }
            assignment
        }
    }
}

fn scan_sharded_core(
    matcher: &GpuLiteralSet,
    backends: &[&dyn VyreBackend],
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
    weights: Option<&[u32]>,
) -> Result<(PagedScanResult, ShardedScanTiming), vyre::BackendError> {
    if backends.is_empty() {
        return Err(vyre::BackendError::new(
            "scan_sharded_fused: the backend set is empty. Fix: pass at least one device backend to shard the corpus across.".to_string(),
        ));
    }
    // One timing slot per backend, in device-set order; each starts at zero windows
    // with a present (Some(0)) device aggregate (an idle shard is timed, not absent).
    let mut shard_timings: Vec<ShardTiming> = (0..backends.len())
        .map(|shard| ShardTiming {
            shard: shard as u32,
            windows: 0,
            bytes_scanned: 0,
            wall_ns: 0,
            device_ns: Some(0),
        })
        .collect();
    let Some((region_count, file_lengths, windows)) = plan_paged(files, window_budget_bytes)?
    else {
        return Ok((
            finish_result(Vec::new(), 0, 0, Vec::new()),
            ShardedScanTiming {
                shards: shard_timings,
            },
        ));
    };
    let overlap = paging_overlap(matcher);
    let window_byte_lens: Vec<usize> = windows.iter().map(|window| window.byte_len).collect();
    let assignment = shard_assignment(&window_byte_lens, backends.len(), weights);

    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_len)
        .max()
        .unwrap_or(0);
    let max_window_regions = windows
        .iter()
        .map(|window| window.file_range.len())
        .max()
        .unwrap_or(0) as u32;

    // Group each shard's window indices (ascending), so a shard replays its own windows
    // in global order. Shard `s` owns every window `i` with `assignment[i] == s`.
    let mut shard_window_indices: Vec<Vec<usize>> = vec![Vec::new(); backends.len()];
    for (index, &shard) in assignment.iter().enumerate() {
        shard_window_indices[shard].push(index);
    }

    // One window's own-region presence block plus its globalized matches, tagged with
    // the GLOBAL window index. Presence is order-sensitive (the per-window own-region
    // rows are concatenated in window order), so the parent re-sorts by `window_index`
    // before concatenating; matches are order-free (finish_result sorts canonically).
    struct ShardWindowOutput {
        window_index: usize,
        words: usize,
        presence_block: Vec<u32>,
        matches: Vec<GlobalMatch>,
    }
    struct ShardThreadResult {
        shard: usize,
        outputs: Vec<ShardWindowOutput>,
        timing: ShardTiming,
    }

    let file_lengths_ref = &file_lengths;
    let windows_ref = &windows;

    // Spawn one thread per device shard: each prepares its OWN resident session, scans
    // its assigned windows CONCURRENTLY with the other devices, globalizes into owned
    // per-window blocks, and frees its session. N devices thus execute in parallel
    // rather than through the old sequential shard loop (W3-5 cross-device PARALLEL
    // dispatch). On a single physical device the shards still run concurrently (distinct
    // sessions + streams); genuine multi-GPU speedup needs distinct peer devices, but
    // the parallel dispatch + deterministic aggregation is exercised and proven correct
    // on one device by driving several backend handles at once.
    let thread_results: Result<Vec<ShardThreadResult>, vyre::BackendError> = std::thread::scope(
        |scope| {
            let mut handles = Vec::with_capacity(backends.len());
            for shard in 0..backends.len() {
                let backend = backends[shard];
                let indices = std::mem::take(&mut shard_window_indices[shard]);
                handles.push(scope.spawn(
                    move || -> Result<ShardThreadResult, vyre::BackendError> {
                        let mut timing = ShardTiming {
                            shard: shard as u32,
                            windows: 0,
                            bytes_scanned: 0,
                            wall_ns: 0,
                            device_ns: Some(0),
                        };
                        // An idle shard (no assigned windows) is timed Some(0) and does
                        // no device work (no session is built for it).
                        if indices.is_empty() {
                            return Ok(ShardThreadResult {
                                shard,
                                outputs: Vec::new(),
                                timing,
                            });
                        }
                        let session = matcher.prepare_resident_fused_scan(
                            backend,
                            max_window_bytes + overlap + 64,
                            max_window_regions + 1,
                            max_matches,
                        )?;
                        // Per-thread scratch reused across THIS shard's windows (the
                        // no-realloc property, held per thread rather than shared).
                        let mut win_presence: Vec<u32> = Vec::new();
                        let mut win_matches: Vec<Match> = Vec::new();
                        let mut scratch: Vec<u8> = Vec::new();
                        let mut outputs = Vec::with_capacity(indices.len());
                        let mut scan_error: Option<vyre::BackendError> = None;
                        for index in indices {
                            let window = &windows_ref[index];
                            let staging = stage_window(files, file_lengths_ref, window, overlap);
                            let timed = match session.scan_into_timed(
                                backend,
                                &staging.haystack,
                                &staging.region_starts,
                                0,
                                &mut win_presence,
                                &mut win_matches,
                                &mut scratch,
                            ) {
                                Ok(timed) => timed,
                                Err(error) => {
                                    scan_error = Some(error);
                                    break;
                                }
                            };
                            // Per-shard timing (the per-shard-active-ns the
                            // load_balance_policy rebalances on). device_ns stays Some
                            // only while every window reported one (loud None otherwise,
                            // never a fabricated 0).
                            timing.windows += 1;
                            timing.bytes_scanned =
                                timing.bytes_scanned.saturating_add(window.byte_len as u64);
                            timing.wall_ns = timing.wall_ns.saturating_add(timed.wall_ns);
                            timing.device_ns = match (timing.device_ns, timed.device_ns) {
                                (Some(acc), Some(window_device)) => {
                                    Some(acc.saturating_add(window_device))
                                }
                                _ => None,
                            };
                            let words = match window_presence_words(
                                &win_presence,
                                staging.region_starts.len(),
                            ) {
                                Ok(words) => words,
                                Err(error) => {
                                    scan_error = Some(error);
                                    break;
                                }
                            };
                            let presence_block =
                                win_presence[..staging.own_region_count * words].to_vec();
                            let mut window_matches = Vec::new();
                            map_window_matches(&staging, &win_matches, &mut window_matches);
                            outputs.push(ShardWindowOutput {
                                window_index: index,
                                words,
                                presence_block,
                                matches: window_matches,
                            });
                        }
                        // Always free this shard's session before surfacing a scan error
                        // (ONE free path per thread, Law 10 (no resident session leaks)).
                        let free_error = session.free(backend).err();
                        if let Some(error) = scan_error {
                            return Err(error);
                        }
                        if let Some(error) = free_error {
                            return Err(error);
                        }
                        Ok(ShardThreadResult {
                            shard,
                            outputs,
                            timing,
                        })
                    },
                ));
            }
            // Join every shard thread. `scope` guarantees all threads complete before it
            // returns, so an early return on the first error never leaks a running thread
            // (each frees its own session internally). A thread panic fails closed rather
            // than returning a partial cross-device result (Law 10).
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle.join() {
                    Ok(Ok(result)) => results.push(result),
                    Ok(Err(error)) => return Err(error),
                    Err(_) => {
                        return Err(vyre::BackendError::new(
                            "scan_sharded_fused: a per-device shard dispatch thread panicked. Fix: a device backend or its resident session raised an unrecoverable panic, inspect the backend logs; cross-device sharded dispatch fails closed rather than returning a partial result.".to_string(),
                        ))
                    }
                }
            }
            Ok(results)
        },
    );
    let results = thread_results?;

    // Deterministic aggregation: place each shard's timing in device-set order, then
    // concatenate every window's own-region presence block in GLOBAL WINDOW ORDER (so
    // the presence layout is byte-identical to a sequential single-device scan) and
    // gather all matches (finish_result sorts them canonically, so their order is free).
    let mut all_outputs: Vec<ShardWindowOutput> = Vec::new();
    for result in results {
        shard_timings[result.shard] = result.timing;
        all_outputs.extend(result.outputs);
    }
    all_outputs.sort_by_key(|output| output.window_index);

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    for output in all_outputs {
        if presence_words == 0 {
            presence_words = output.words;
        } else if output.words != presence_words {
            return Err(vyre::BackendError::new(format!(
                "scan_sharded_fused: presence word count changed across windows ({presence_words} -> {}). Fix: internal invariant broke, report with the corpus shape.",
                output.words
            )));
        }
        presence.extend_from_slice(&output.presence_block);
        matches.extend(output.matches);
    }

    Ok((
        finish_result(presence, region_count, presence_words, matches),
        ShardedScanTiming {
            shards: shard_timings,
        },
    ))
}

/// One pattern-database shard: a sub-matcher compiled over a SUBSET of the global
/// rule set, plus the global pattern id each of its LOCAL pattern ids maps to
/// (`global_pattern_ids[local_id]`). This is the unit of the W3-5
/// `pattern-database-replicated-shards` workload: "stripe the large rule database
/// by rule family across the device set", as opposed to the byte-range
/// haystack sharding of [`scan_sharded_fused`].
pub struct PatternShard<'a> {
    /// The sub-matcher, compiled over this shard's rule subset.
    pub matcher: &'a GpuLiteralSet,
    /// `global_pattern_ids[i]` is the global rule id of this sub-matcher's local
    /// pattern `i`; its length must be at least the sub-matcher's pattern count.
    pub global_pattern_ids: &'a [u32],
}

/// Scan `haystack` with a pattern database STRIPED across shards: each shard holds a
/// disjoint subset of the rules, runs on its assigned device
/// (`backends[shard_index % backends.len()]`), and its local matches are remapped
/// into the GLOBAL rule numbering and merged into one canonical report (stable-sorted
/// by `(pattern_id, start, end)`).
///
/// Because literal matching is independent per rule, the union of disjoint rule
/// subsets over the same haystack is exactly the full rule set's match set, so the
/// striped result is identical to scanning one un-sharded matcher built over every
/// rule (the plan's `replicated-and-striped-...-produce-identical-detection-output`
/// parity policy). On a real multi-GPU host the shards run on distinct peer devices;
/// on a single-device host they run sequentially, which still exercises the full
/// stripe / remap / merge architecture.
///
/// # Errors
/// [`vyre::BackendError`] if `backends` is empty (fail closed), if any shard emits a
/// local pattern id with no entry in its `global_pattern_ids` (a malformed shard map
///: fail closed rather than drop or mis-attribute the finding, Law 10), or on any
/// shard's scan failure.
pub fn scan_pattern_sharded(
    shards: &[PatternShard<'_>],
    backends: &[&dyn VyreBackend],
    haystack: &[u8],
) -> Result<Vec<Match>, vyre::BackendError> {
    if backends.is_empty() {
        return Err(vyre::BackendError::new(
            "scan_pattern_sharded: the backend set is empty. Fix: pass at least one device backend to stripe the rule database across.".to_string(),
        ));
    }
    let mut merged: Vec<Match> = Vec::new();
    for (index, shard) in shards.iter().enumerate() {
        let backend = backends[index % backends.len()];
        let local_matches = shard.matcher.scan_all(backend, haystack)?;
        for mut hit in local_matches {
            let Some(&global_id) = shard.global_pattern_ids.get(hit.pattern_id as usize) else {
                return Err(vyre::BackendError::new(format!(
                    "scan_pattern_sharded: shard {index} produced local pattern id {} but its global_pattern_ids map has only {} entries. Fix: give each shard a global id for every rule in its sub-matcher.",
                    hit.pattern_id,
                    shard.global_pattern_ids.len()
                )));
            };
            // In-place remap of the local rule id to the global one (Match is
            // #[non_exhaustive]; mutating the pub field avoids reconstructing it).
            hit.pattern_id = global_id;
            merged.push(hit);
        }
    }
    // Canonical report order, the same (pattern_id, start, end) Ord a single-shot
    // full-matcher scan produces, so the striped result is directly comparable.
    merged.sort_unstable();
    Ok(merged)
}

/// Number of window dispatches kept in flight by [`scan_paged_fused_async`]: two is
/// enough to overlap window `k+1`'s host staging + upload with window `k`'s device
/// execution without unbounded memory.
const PAGED_PIPELINE_DEPTH: usize = 2;

/// Asynchronous twin of [`scan_paged_fused`]: pipelines the windows so each
/// window's host staging + upload overlaps the previous window's device execution,
/// keeping [`PAGED_PIPELINE_DEPTH`] dispatches in flight. It uses the BORROWED async
/// fused dispatch (the tables re-upload per window, amortized over a large window 
/// rather than staying resident), which is the trade the overlap buys. The
/// boundary handling (`L-1` overlap, dummy overlap region, start-based dedup) is the
/// SAME shared globalization as the sync driver, so the result is identical.
///
/// # Errors
/// Same as [`scan_paged_fused`].
pub fn scan_paged_fused_async(
    matcher: &GpuLiteralSet,
    backend: &dyn VyreBackend,
    files: &[&[u8]],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    let Some((region_count, file_lengths, windows)) = plan_paged(files, window_budget_bytes)?
    else {
        return Ok(finish_result(Vec::new(), 0, 0, Vec::new()));
    };
    let overlap = paging_overlap(matcher);

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    let mut win_matches: Vec<Match> = Vec::new();
    let mut inflight: VecDeque<(WindowStaging, PendingFusedRegion)> = VecDeque::new();

    let retire = |staging: WindowStaging,
                  pending: PendingFusedRegion,
                  presence_words: &mut usize,
                  presence: &mut Vec<u32>,
                  matches: &mut Vec<GlobalMatch>,
                  win_matches: &mut Vec<Match>|
     -> Result<(), vyre::BackendError> {
        // The retained `staging` keeps its haystack alive; the pending decodes both
        // the presence words (returned) and the match triples (into win_matches).
        let win_presence = pending.await_into(win_matches)?;
        globalize_window(
            &staging,
            &win_presence,
            win_matches,
            presence_words,
            presence,
            matches,
        )
    };

    for window in &windows {
        let staging = stage_window(files, &file_lengths, window, overlap);
        let pending = matcher.scan_presence_and_positions_by_region_async(
            backend,
            &staging.haystack,
            &staging.region_starts,
            0,
            max_matches,
        )?;
        inflight.push_back((staging, pending));
        if inflight.len() >= PAGED_PIPELINE_DEPTH {
            let (staging, pending) = inflight.pop_front().expect("depth reached");
            retire(
                staging,
                pending,
                &mut presence_words,
                &mut presence,
                &mut matches,
                &mut win_matches,
            )?;
        }
    }
    while let Some((staging, pending)) = inflight.pop_front() {
        retire(
            staging,
            pending,
            &mut presence_words,
            &mut presence,
            &mut matches,
            &mut win_matches,
        )?;
    }

    Ok(finish_result(
        presence,
        region_count,
        presence_words,
        matches,
    ))
}

/// Read a window's own files (plus up to `overlap` bytes of the following files)
/// from disk into `haystack`, returning the overlap bytes gathered. Only this
/// window's bytes are ever resident (host RSS bounded by the window, not the
/// corpus); it reads only the needed `overlap` prefix of each following file.
fn fill_window_from_paths(
    paths: &[&Path],
    window: &CorpusWindow,
    overlap: usize,
    haystack: &mut Vec<u8>,
) -> std::io::Result<usize> {
    haystack.clear();
    for path in &paths[window.file_range.clone()] {
        std::fs::File::open(path)?.read_to_end(haystack)?;
    }
    let mut gathered = 0usize;
    for path in &paths[window.file_range.end..] {
        if gathered >= overlap {
            break;
        }
        let want = (overlap - gathered) as u64;
        let before = haystack.len();
        std::fs::File::open(path)?
            .take(want)
            .read_to_end(haystack)?;
        gathered += haystack.len() - before;
    }
    Ok(gathered)
}

/// Disk-backed [`scan_paged_fused`]: scan a corpus given as file `paths` (each a
/// region, in order) that may exceed one resident window, reading only ONE window's
/// files into memory at a time. The result is identical to [`scan_paged_fused`]
/// over the same bytes held in memory (it shares the exact overlap/dummy/dedup
/// globalization), the difference is that host RSS stays bounded by one window,
/// so a corpus too large to hold at once is still a single call.
///
/// # Errors
/// Returns [`vyre::BackendError`] on a stat/read failure of any path, a dispatch
/// failure, an invalid plan, a file larger than the u32 haystack ABI, or a file
/// count over the u32 region ABI.
pub fn scan_paths_paged(
    matcher: &GpuLiteralSet,
    backend: &dyn VyreBackend,
    paths: &[&Path],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    let region_count = u32::try_from(paths.len()).map_err(|_| {
        vyre::BackendError::new(
            "scan_paths_paged: file count exceeds the u32 region-id ABI. Fix: fewer files per corpus or shard the corpus.".to_string(),
        )
    })?;
    // Stat each path for its size (fail closed on an unreadable or oversized file).
    let mut file_lengths: Vec<usize> = Vec::with_capacity(paths.len());
    for path in paths {
        let len = std::fs::metadata(path)
            .map_err(|error| {
                vyre::BackendError::new(format!(
                    "scan_paths_paged: cannot stat {}: {error}. Fix: ensure every path is a readable regular file.",
                    path.display()
                ))
            })?
            .len();
        let len = usize::try_from(len).map_err(|_| {
            vyre::BackendError::new(format!(
                "scan_paths_paged: {} is {len} bytes, larger than host usize. Fix: pre-split it.",
                path.display()
            ))
        })?;
        file_lengths.push(len);
    }

    let windows = plan_corpus_windows(&file_lengths, window_budget_bytes)
        .map_err(|error| vyre::BackendError::new(error.to_string()))?;
    if windows.is_empty() {
        return Ok(finish_result(Vec::new(), 0, 0, Vec::new()));
    }
    let overlap = paging_overlap(matcher);

    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_len)
        .max()
        .unwrap_or(0);
    let max_window_regions = windows
        .iter()
        .map(|window| window.file_range.len())
        .max()
        .unwrap_or(0) as u32;
    let session = matcher.prepare_resident_fused_scan(
        backend,
        max_window_bytes + overlap + 64,
        max_window_regions + 1,
        max_matches,
    )?;

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    let mut win_presence: Vec<u32> = Vec::new();
    let mut win_matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();

    for window in &windows {
        let mut haystack: Vec<u8> = Vec::new();
        let gathered = match fill_window_from_paths(paths, window, overlap, &mut haystack) {
            Ok(gathered) => gathered,
            Err(error) => {
                let _ = session.free(backend);
                return Err(vyre::BackendError::new(format!(
                    "scan_paths_paged: reading window files failed: {error}. Fix: ensure every path is a readable regular file."
                )));
            }
        };
        let (region_starts, own_region_count) =
            window_region_starts(window, &file_lengths, gathered > 0);
        let staging = WindowStaging {
            haystack,
            region_starts,
            own_len: window.byte_len,
            own_region_count,
            byte_offset: window.byte_offset,
            global_region_base: window.global_region_base,
        };

        session.scan_into(
            backend,
            &staging.haystack,
            &staging.region_starts,
            0,
            &mut win_presence,
            &mut win_matches,
            &mut scratch,
        )?;
        if let Err(error) = globalize_window(
            &staging,
            &win_presence,
            &win_matches,
            &mut presence_words,
            &mut presence,
            &mut matches,
        ) {
            let _ = session.free(backend);
            return Err(error);
        }
    }

    session.free(backend)?;
    Ok(finish_result(
        presence,
        region_count,
        presence_words,
        matches,
    ))
}

/// Prefetch depth for [`scan_paths_paged_prefetched`]: a `sync_channel(1)` lets the
/// reader be at most one window ahead of the scanner, so at most two windows' bytes
/// are resident at once (disk I/O overlaps device compute without unbounded RSS).
const PAGED_PREFETCH_DEPTH: usize = 1;

/// Prefetched disk-backed paged scan: a background thread reads window `k+1`'s files
/// from disk while the GPU scans window `k`, so disk I/O overlaps device compute.
/// Host RSS stays bounded, a `sync_channel(PAGED_PREFETCH_DEPTH)` backpressures the
/// reader, and the result is identical to [`scan_paths_paged`] (same plan and
/// globalization; the only change is that reads overlap compute).
///
/// # Errors
/// Same as [`scan_paths_paged`], plus a reader-thread I/O error surfaced as a
/// backend error.
pub fn scan_paths_paged_prefetched(
    matcher: &GpuLiteralSet,
    backend: &dyn VyreBackend,
    paths: &[&Path],
    window_budget_bytes: usize,
    max_matches: u32,
) -> Result<PagedScanResult, vyre::BackendError> {
    let region_count = u32::try_from(paths.len()).map_err(|_| {
        vyre::BackendError::new(
            "scan_paths_paged_prefetched: file count exceeds the u32 region-id ABI. Fix: fewer files per corpus or shard the corpus.".to_string(),
        )
    })?;
    let mut file_lengths: Vec<usize> = Vec::with_capacity(paths.len());
    for path in paths {
        let len = std::fs::metadata(path)
            .map_err(|error| {
                vyre::BackendError::new(format!(
                    "scan_paths_paged_prefetched: cannot stat {}: {error}. Fix: ensure every path is a readable regular file.",
                    path.display()
                ))
            })?
            .len();
        file_lengths.push(usize::try_from(len).map_err(|_| {
            vyre::BackendError::new(format!(
                "scan_paths_paged_prefetched: {} is {len} bytes, larger than host usize. Fix: pre-split it.",
                path.display()
            ))
        })?);
    }

    let windows = plan_corpus_windows(&file_lengths, window_budget_bytes)
        .map_err(|error| vyre::BackendError::new(error.to_string()))?;
    if windows.is_empty() {
        return Ok(finish_result(Vec::new(), 0, 0, Vec::new()));
    }
    let overlap = paging_overlap(matcher);

    // Owned copies for the reader thread (paths and the window plan are Send).
    let owned_paths: Vec<std::path::PathBuf> =
        paths.iter().map(|path| path.to_path_buf()).collect();
    let reader_windows = windows.clone();

    // The reader sends `(window_index, haystack, overlap_gathered)` in order; the
    // bounded channel keeps it at most PAGED_PREFETCH_DEPTH windows ahead.
    #[allow(clippy::type_complexity)]
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::io::Result<(usize, Vec<u8>, usize)>>(
        PAGED_PREFETCH_DEPTH,
    );
    let reader = std::thread::spawn(move || {
        let path_refs: Vec<&Path> = owned_paths.iter().map(|path| path.as_path()).collect();
        for (index, window) in reader_windows.iter().enumerate() {
            let mut haystack = Vec::new();
            let message = fill_window_from_paths(&path_refs, window, overlap, &mut haystack)
                .map(|gathered| (index, haystack, gathered));
            let forwarded_error = message.is_err();
            // If the receiver hung up, or we just forwarded an error, stop reading.
            if tx.send(message).is_err() || forwarded_error {
                break;
            }
        }
    });

    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_len)
        .max()
        .unwrap_or(0);
    let max_window_regions = windows
        .iter()
        .map(|window| window.file_range.len())
        .max()
        .unwrap_or(0) as u32;
    let session = matcher.prepare_resident_fused_scan(
        backend,
        max_window_bytes + overlap + 64,
        max_window_regions + 1,
        max_matches,
    )?;

    let mut presence: Vec<u32> = Vec::new();
    let mut presence_words: usize = 0;
    let mut matches: Vec<GlobalMatch> = Vec::new();
    let mut win_presence: Vec<u32> = Vec::new();
    let mut win_matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    let mut outcome: Result<(), vyre::BackendError> = Ok(());

    for message in rx.iter() {
        let (index, haystack, gathered) = match message {
            Ok(payload) => payload,
            Err(error) => {
                outcome = Err(vyre::BackendError::new(format!(
                    "scan_paths_paged_prefetched: reading window files failed: {error}. Fix: ensure every path is a readable regular file."
                )));
                break;
            }
        };
        let window = &windows[index];
        let (region_starts, own_region_count) =
            window_region_starts(window, &file_lengths, gathered > 0);
        let staging = WindowStaging {
            haystack,
            region_starts,
            own_len: window.byte_len,
            own_region_count,
            byte_offset: window.byte_offset,
            global_region_base: window.global_region_base,
        };
        if let Err(error) = session.scan_into(
            backend,
            &staging.haystack,
            &staging.region_starts,
            0,
            &mut win_presence,
            &mut win_matches,
            &mut scratch,
        ) {
            outcome = Err(error);
            break;
        }
        if let Err(error) = globalize_window(
            &staging,
            &win_presence,
            &win_matches,
            &mut presence_words,
            &mut presence,
            &mut matches,
        ) {
            outcome = Err(error);
            break;
        }
    }

    // Dropping the receiver unblocks a reader still parked on `send`; join so the
    // thread never outlives the call.
    drop(rx);
    let _ = reader.join();
    let _ = session.free(backend);
    outcome?;
    Ok(finish_result(
        presence,
        region_count,
        presence_words,
        matches,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every file appears in exactly one window, windows tile the file index space
    /// with no gap or overlap, byte offsets are contiguous and monotone, and each
    /// window's global_region_base equals its first global file index.
    fn assert_plan_is_a_partition(file_lengths: &[usize], windows: &[CorpusWindow]) {
        let mut expected_file = 0usize;
        let mut expected_offset = 0u64;
        for window in windows {
            assert_eq!(
                window.file_range.start, expected_file,
                "windows must tile file indices with no gap/overlap"
            );
            assert_eq!(
                window.global_region_base as usize, window.file_range.start,
                "global_region_base must equal the window's first global file index"
            );
            assert_eq!(
                window.byte_offset, expected_offset,
                "byte offsets must be contiguous and monotone"
            );
            let bytes: usize = file_lengths[window.file_range.clone()].iter().sum();
            assert_eq!(
                window.byte_len, bytes,
                "byte_len must sum the window's files"
            );
            expected_file = window.file_range.end;
            expected_offset += bytes as u64;
        }
        assert_eq!(
            expected_file,
            file_lengths.len(),
            "the windows must cover every file"
        );
        let total: u64 = file_lengths.iter().map(|&l| l as u64).sum();
        assert_eq!(expected_offset, total, "the windows must cover every byte");
    }

    #[test]
    fn empty_corpus_plans_to_no_windows() {
        assert_eq!(plan_corpus_windows(&[], 1024), Ok(Vec::new()));
    }

    #[test]
    fn zero_budget_fails_closed() {
        assert_eq!(
            plan_corpus_windows(&[10], 0),
            Err(PagedCorpusError::ZeroBudget)
        );
    }

    #[test]
    fn all_files_in_one_window_when_under_budget() {
        let lengths = [100, 200, 300];
        let windows = plan_corpus_windows(&lengths, 1024).expect("plan");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].file_range, 0..3);
        assert_eq!(windows[0].byte_len, 600);
        assert_eq!(windows[0].global_region_base, 0);
        assert_plan_is_a_partition(&lengths, &windows);
    }

    #[test]
    fn splits_at_the_budget_boundary_with_stable_global_ids() {
        // Budget 250: [100,100] fills to 200 (next 100 -> 300 > 250, break),
        // then [100,100] -> 200, then [100] -> 100. Three windows.
        let lengths = [100, 100, 100, 100, 100];
        let windows = plan_corpus_windows(&lengths, 250).expect("plan");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].file_range, 0..2);
        assert_eq!(windows[0].global_region_base, 0);
        assert_eq!(windows[0].byte_offset, 0);
        assert_eq!(windows[1].file_range, 2..4);
        assert_eq!(windows[1].global_region_base, 2);
        assert_eq!(windows[1].byte_offset, 200);
        assert_eq!(windows[2].file_range, 4..5);
        assert_eq!(windows[2].global_region_base, 4);
        assert_eq!(windows[2].byte_offset, 400);
        assert_plan_is_a_partition(&lengths, &windows);
    }

    #[test]
    fn exact_budget_fit_packs_the_whole_window() {
        let lengths = [128, 128];
        let windows = plan_corpus_windows(&lengths, 256).expect("plan");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].byte_len, 256);
        assert_plan_is_a_partition(&lengths, &windows);
    }

    #[test]
    fn oversized_file_gets_its_own_window_and_progress_is_guaranteed() {
        // File 1 (500) exceeds the 250 budget: it lands alone in its own window,
        // and planning still terminates covering everything.
        let lengths = [100, 500, 100];
        let windows = plan_corpus_windows(&lengths, 250).expect("plan");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].file_range, 0..1); // 100
        assert_eq!(windows[1].file_range, 1..2); // 500, over budget, alone
        assert_eq!(windows[1].byte_len, 500);
        assert_eq!(windows[2].file_range, 2..3); // 100
        assert_plan_is_a_partition(&lengths, &windows);
    }

    #[test]
    fn a_run_of_tiny_files_coalesces_up_to_the_budget() {
        let lengths = [10; 100]; // 100 files of 10 bytes
        let windows = plan_corpus_windows(&lengths, 55).expect("plan"); // 5 files/window
        assert!(windows.iter().all(|w| w.byte_len <= 55));
        for window in &windows[..windows.len() - 1] {
            assert_eq!(window.file_range.len(), 5, "each full window packs 5 files");
        }
        assert_plan_is_a_partition(&lengths, &windows);
    }

    #[test]
    fn local_region_starts_are_window_relative_first_zero() {
        let lengths = [100, 100, 100, 100, 100];
        let windows = plan_corpus_windows(&lengths, 250).expect("plan");
        // Window 1 covers global files 2..4; its local region starts are [0, 100].
        let starts = windows[1].local_region_starts(&lengths);
        assert_eq!(starts, vec![0, 100]);
        assert_eq!(starts[0], 0, "the first local region start must be 0");
    }

    #[test]
    fn file_over_u32_abi_fails_closed() {
        let lengths = [10, (u32::MAX as usize) + 1];
        assert_eq!(
            plan_corpus_windows(&lengths, usize::MAX),
            Err(PagedCorpusError::FileExceedsHaystackAbi {
                file_index: 1,
                len: (u32::MAX as usize) + 1,
            })
        );
    }

    #[test]
    fn every_error_variant_names_its_owner_and_fix_path() {
        // W8-2 discipline applied to this module's refusals: owner prefix + fix.
        let errors = [
            PagedCorpusError::ZeroBudget,
            PagedCorpusError::FileExceedsHaystackAbi {
                file_index: 0,
                len: 1,
            },
            PagedCorpusError::TooManyRegions { count: 1 },
        ];
        for error in &errors {
            // Exhaustive match: a new variant fails to compile until listed.
            match error {
                PagedCorpusError::ZeroBudget
                | PagedCorpusError::FileExceedsHaystackAbi { .. }
                | PagedCorpusError::TooManyRegions { .. } => {}
            }
            let rendered = error.to_string();
            assert!(
                rendered.starts_with("paged_corpus:"),
                "refusal lacks the owner prefix: {rendered}"
            );
            assert!(
                rendered.contains("Fix:"),
                "refusal lacks a fix path: {rendered}"
            );
        }
    }

    /// The load-bearing correctness proof: a paged fused scan of a multi-window
    /// corpus, including a `secret` literal that STRADDLES a window boundary 
    /// produces EXACTLY the same per-region presence and (globalized) positioned
    /// matches as a single-shot fused scan of the concatenated corpus (Law 10, no
    /// boundary miss, no overlap over-fire, no double count). Runs on the real GPU;
    /// skips cleanly with none.
    #[test]
    fn paged_fused_scan_equals_single_shot_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping paged fused GPU parity test");
                return;
            }
        };

        // "secret" (id 1) is the longest pattern (6) -> overlap 5.
        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),    // 9 : region 0, contains XYZ
            b"cccsec".to_vec(),       // 6 : region 1, ends with "sec"
            b"retdddsecret".to_vec(), // 12: region 2, starts "ret" (completes a cross-window "secret") + its own "secret"
            b"eeeABfff".to_vec(),     // 8 : region 3, contains AB
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 16usize; // forces window0=files[0..2] (15B), window1=files[2..3], window2=files[3..4]
        let max_matches = 4_096u32;

        // Multi-window sanity: the boundary between region 1 and region 2 is a
        // WINDOW boundary, so the cross-file "secret" also spans windows.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        let plan = plan_corpus_windows(&lengths, budget).expect("plan");
        assert!(
            plan.len() >= 2,
            "the test corpus must span multiple windows; got {}",
            plan.len()
        );

        let paged = scan_paged_fused(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
            .expect("paged fused scan");

        // Single-shot ground truth over the concatenated corpus.
        let mut whole = Vec::new();
        let mut region_starts = Vec::new();
        for file in &files {
            region_starts.push(whole.len() as u32);
            whole.extend_from_slice(file);
        }
        let mut single_matches: Vec<Match> = Vec::new();
        let single_presence = matcher
            .scan_presence_and_positions_by_region(
                backend.as_ref(),
                &whole,
                &region_starts,
                0,
                max_matches,
                &mut single_matches,
            )
            .expect("single-shot fused scan");

        // Presence identical, in the same global region numbering.
        assert_eq!(
            paged.presence, single_presence,
            "paged per-region presence must equal the single-shot scan"
        );
        assert_eq!(paged.region_count as usize, files.len());
        assert!(
            paged.presence.iter().any(|&word| word != 0),
            "the corpus must set presence bits (non-vacuous)"
        );

        // Matches identical after globalizing the single-shot result the same way.
        let mut expected: Vec<GlobalMatch> = single_matches
            .iter()
            .map(|hit| {
                let region = region_starts
                    .partition_point(|&start| start <= hit.start)
                    .saturating_sub(1);
                GlobalMatch {
                    pattern_id: hit.pattern_id,
                    region_id: region as u32,
                    start: u64::from(hit.start),
                    end: u64::from(hit.end),
                }
            })
            .collect();
        expected.sort_unstable_by_key(|hit| (hit.region_id, hit.start, hit.end, hit.pattern_id));
        assert_eq!(
            paged.matches, expected,
            "paged globalized matches must equal the single-shot scan"
        );

        // The cross-window "secret" (pattern id 1) starting in region 1 must be
        // present (proving the overlap caught the boundary-spanning match).
        assert!(
            paged
                .matches
                .iter()
                .any(|hit| hit.pattern_id == 1 && hit.region_id == 1),
            "the window-boundary-spanning `secret` (region 1) must be found"
        );
    }

    /// The timed paged driver returns a result byte-identical to the untimed
    /// driver, plus an honest per-window timing aggregate: `windows` matches the
    /// plan, `bytes_scanned` equals the corpus size (overlap excluded), `wall_ns`
    /// is non-zero, and on the real GPU `device_ns` is `Some` (a device timer is
    /// present). Runs on the real GPU; skips cleanly with none.
    #[test]
    fn timed_paged_fused_scan_equals_untimed_and_reports_timing_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping timed paged GPU test");
                return;
            }
        };

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),
            b"cccsec".to_vec(),
            b"retdddsecret".to_vec(),
            b"eeeABfff".to_vec(),
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 16usize;
        let max_matches = 4_096u32;

        let untimed = scan_paged_fused(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
            .expect("untimed paged fused scan");
        let (timed_result, timing) =
            scan_paged_fused_timed(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
                .expect("timed paged fused scan");

        // The result must be byte-identical to the untimed driver (timing is free).
        assert_eq!(
            timed_result, untimed,
            "the timed paged scan result must equal the untimed paged scan result"
        );

        // The timing aggregate must be honest and concrete.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        let plan = plan_corpus_windows(&lengths, budget).expect("plan");
        assert_eq!(
            timing.windows as usize,
            plan.len(),
            "timing.windows must equal the number of planned windows"
        );
        assert!(timing.windows >= 2, "the corpus must span multiple windows");
        let corpus_bytes: u64 = files.iter().map(|file| file.len() as u64).sum();
        assert_eq!(
            timing.bytes_scanned, corpus_bytes,
            "bytes_scanned must equal the corpus size (overlap excluded)"
        );
        assert!(timing.wall_ns > 0, "wall_ns must be a real measurement");
        assert!(
            timing.device_ns.is_some(),
            "the wgpu backend has a device timer; device_ns must be Some, not a fabricated absence"
        );
        assert!(
            timing.device_ns.unwrap() > 0,
            "device_ns must be a real non-zero kernel time on the GPU"
        );
    }

    /// The empty-corpus timed path reports a zeroed-but-present device aggregate
    /// (`Some(0)`, no windows), never `None` (there was nothing to fail to time)
    ///: and an empty result. No GPU needed (plan is empty, no dispatch).
    #[test]
    fn timed_paged_fused_scan_empty_corpus_reports_zero_timing() {
        use vyre_driver_reference::CpuRefBackend;

        let matcher = GpuLiteralSet::compile(&[b"secret".as_slice()]);
        // A CpuRef backend is never touched here: an empty plan short-circuits
        // before any dispatch, so this exercises the zero-window timing branch.
        let (result, timing) = scan_paged_fused_timed(&matcher, &CpuRefBackend, &[], 64, 16)
            .expect("empty timed scan");
        assert_eq!(result.region_count, 0);
        assert!(result.matches.is_empty());
        assert_eq!(
            timing,
            PagedScanTiming {
                windows: 0,
                bytes_scanned: 0,
                wall_ns: 0,
                device_ns: Some(0)
            },
            "an empty corpus times as zero windows with a present (Some(0)) device aggregate"
        );
    }

    /// The multi-GPU sharding driver produces a result byte-identical to the
    /// single-device paged scan regardless of shard/device-set size, the plan's
    /// `parity_policy`. Exercised with a 1-device set AND a 3-device set (the same
    /// GPU repeated, so the round-robin window→shard assignment + per-shard resident
    /// sessions + aggregation all run) on the real GPU; skips cleanly with none.
    #[test]
    fn sharded_fused_scan_equals_single_device_across_device_set_sizes_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping sharded fused GPU parity test");
                return;
            }
        };
        let device: &dyn VyreBackend = backend.as_ref();

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),
            b"cccsec".to_vec(),
            b"retdddsecret".to_vec(),
            b"eeeABfff".to_vec(),
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 16usize; // forces >=3 windows (secret straddles a window boundary)
        let max_matches = 4_096u32;

        // Single-device ground truth (the existing paged driver).
        let single = scan_paged_fused(&matcher, device, &file_refs, budget, max_matches)
            .expect("single-device paged scan");

        // A multi-window plan is required for round-robin to actually distribute.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        assert!(
            plan_corpus_windows(&lengths, budget).expect("plan").len() >= 2,
            "the corpus must span multiple windows so sharding distributes work"
        );

        // 1-device set: degenerates to the single-device scan.
        let one = scan_sharded_fused(&matcher, &[device], &file_refs, budget, max_matches)
            .expect("sharded scan over 1 device");
        assert_eq!(
            one, single,
            "sharded scan over a 1-device set must equal the single-device paged scan"
        );

        // 3-device set (same GPU repeated): the round-robin assigns window k to
        // device k%3, each with its own resident session; the aggregation must still
        // produce the identical global result.
        let three = scan_sharded_fused(
            &matcher,
            &[device, device, device],
            &file_refs,
            budget,
            max_matches,
        )
        .expect("sharded scan over 3 devices");
        assert_eq!(
            three, single,
            "sharded scan over a 3-device set must equal the single-device paged scan (parity_policy)"
        );

        // Non-vacuous: the boundary-spanning `secret` (pattern 1) starting in region
        // 1 is present, proving the halo/overlap survived sharding.
        assert!(
            three
                .matches
                .iter()
                .any(|hit| hit.pattern_id == 1 && hit.region_id == 1),
            "the window-boundary-spanning `secret` must survive multi-shard aggregation"
        );

        // Throughput-WEIGHTED assignment (3:1 across a 2-device set) redistributes
        // the byte-work but must produce the identical result (parity_policy).
        let weighted = scan_sharded_fused_weighted(
            &matcher,
            &[device, device],
            &[3, 1],
            &file_refs,
            budget,
            max_matches,
        )
        .expect("weighted sharded scan");
        assert_eq!(
            weighted, single,
            "throughput-weighted sharding must equal the single-device paged scan"
        );

        // Fail closed on an empty device set and on a weights/backends length mismatch.
        assert!(
            scan_sharded_fused(&matcher, &[], &file_refs, budget, max_matches).is_err(),
            "an empty backend set must fail closed, not silently scan nothing"
        );
        assert!(
            scan_sharded_fused_weighted(
                &matcher,
                &[device, device],
                &[1],
                &file_refs,
                budget,
                max_matches
            )
            .is_err(),
            "a weights/backends length mismatch must fail closed"
        );
    }

    /// W3-5 cross-device PARALLEL dispatch under real concurrency stress: a
    /// many-window corpus sharded across a FOUR-handle device set drives four
    /// resident sessions dispatching CONCURRENTLY (one OS thread per shard). The
    /// aggregated result must be byte-identical to a single-device paged scan
    /// (parity is order-independent despite the nondeterministic thread interleave),
    /// AND the honest per-shard timing must show the work genuinely SPREAD across all
    /// four shards, not serialized onto one. On a single physical GPU the four
    /// sessions still run concurrently (distinct sessions + streams), proving the
    /// parallel dispatch + deterministic aggregation correct; true multi-GPU speedup
    /// needs distinct peer devices. Runs on the real GPU; skips cleanly with none.
    #[test]
    fn parallel_sharded_dispatch_across_four_concurrent_handles_equals_single_shot_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping parallel sharded stress test");
                return;
            }
        };
        let device: &dyn VyreBackend = backend.as_ref();

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);

        // 32 deterministic files sprinkled with matches, plus a `secret` that STRADDLES
        // the file-10/file-11 boundary (file 10 ends "sec", file 11 starts "ret"), so
        // the overlap halo must survive across a shard boundary.
        let mut files: Vec<Vec<u8>> = Vec::new();
        for i in 0..32usize {
            if i == 10 {
                files.push(b"begin10sec".to_vec()); // ends with "sec"
            } else if i == 11 {
                files.push(b"retTAILb11".to_vec()); // starts "ret" -> completes cross-file "secret"
            } else {
                let mut file = Vec::new();
                file.extend_from_slice(b"pad");
                if i % 3 == 0 {
                    file.extend_from_slice(b"XYZ");
                }
                if i % 4 == 0 {
                    file.extend_from_slice(b"secret");
                }
                if i % 5 == 0 {
                    file.extend_from_slice(b"AB");
                }
                file.extend_from_slice(format!("z{i:02}").as_bytes());
                files.push(file);
            }
        }
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 24usize; // small budget over ~320B -> many windows
        let max_matches = 4_096u32;

        // The corpus must span MANY windows so four shards each get several, genuine
        // concurrent multi-session dispatch, not a degenerate one-window case.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        let window_count = plan_corpus_windows(&lengths, budget).expect("plan").len();
        assert!(
            window_count >= 8,
            "the stress corpus must span many windows so 4 shards run concurrently; got {window_count}"
        );

        // Single-device ground truth.
        let single = scan_paged_fused(&matcher, device, &file_refs, budget, max_matches)
            .expect("single-device paged scan");

        // FOUR concurrent handles to the same GPU -> four threads dispatch in parallel.
        let devices: [&dyn VyreBackend; 4] = [device, device, device, device];
        let (parallel, timing) =
            scan_sharded_fused_timed(&matcher, &devices, &file_refs, budget, max_matches)
                .expect("parallel 4-shard sharded scan");

        // Byte-identical to the single-device scan despite the concurrent interleave.
        assert_eq!(
            parallel, single,
            "parallel 4-shard dispatch must equal the single-device paged scan (parity is interleave-independent)"
        );

        // Non-vacuous: real presence bits and the boundary-straddling `secret`.
        assert!(
            parallel.presence.iter().any(|&word| word != 0),
            "the corpus must set presence bits (non-vacuous)"
        );
        assert!(
            parallel.matches.iter().any(|hit| hit.pattern_id == 1),
            "the corpus contains `secret` matches that must survive parallel aggregation"
        );

        // Work was genuinely SPREAD across all four concurrent shards, and the honest
        // per-shard timing sums back to the whole corpus (round-robin over >=8 windows
        // guarantees every shard of 4 gets >=1 window).
        assert_eq!(
            timing.shards.len(),
            4,
            "one timing slot per device in the 4-handle set"
        );
        for (shard, entry) in timing.shards.iter().enumerate() {
            assert_eq!(
                entry.shard as usize, shard,
                "shard timing is in device-set order"
            );
            assert!(
                entry.windows > 0,
                "shard {shard} did no work, parallel dispatch failed to distribute across all 4 devices"
            );
        }
        let total_windows: u32 = timing.shards.iter().map(|entry| entry.windows).sum();
        assert_eq!(
            total_windows as usize, window_count,
            "every planned window must be dispatched by exactly one shard"
        );
        let total_bytes: u64 = timing.shards.iter().map(|entry| entry.bytes_scanned).sum();
        let corpus_bytes: u64 = files.iter().map(|file| file.len() as u64).sum();
        assert_eq!(
            total_bytes, corpus_bytes,
            "the per-shard bytes_scanned must sum to the corpus size (overlap excluded)"
        );
    }

    /// The shard-assignment balancer is pure host logic (no GPU): round-robin with
    /// no weights, and byte-work proportional to weights otherwise, deterministically.
    #[test]
    fn shard_assignment_round_robins_and_honors_throughput_weights() {
        // No weights -> plain round-robin.
        let lens = [10usize, 10, 10, 10];
        assert_eq!(shard_assignment(&lens, 3, None), vec![0, 1, 2, 0]);

        // Equal weights over equal windows -> evenly distributed (one each, then wrap).
        assert_eq!(
            shard_assignment(&lens, 3, Some(&[1, 1, 1])),
            vec![0, 1, 2, 0]
        );

        // 3:1 weight over 4 equal windows -> shard 0 takes 3, shard 1 takes 1.
        let assignment = shard_assignment(&lens, 2, Some(&[3, 1]));
        let shard0 = assignment.iter().filter(|&&s| s == 0).count();
        let shard1 = assignment.iter().filter(|&&s| s == 1).count();
        assert_eq!(
            (shard0, shard1),
            (3, 1),
            "a 3:1 throughput weight must give shard 0 three of four equal windows"
        );

        // A zero weight is treated as 1 (never starved, never divides by zero): the
        // zero-weight shard still receives work rather than being skipped forever.
        let assignment = shard_assignment(&lens, 2, Some(&[0, 1]));
        assert!(
            assignment.contains(&0) && assignment.contains(&1),
            "a zero-weight shard must still receive work (treated as weight 1)"
        );
    }

    /// Property gate for the sharding load balancer: for ANY window sizes, shard count,
    /// and (optional) weights, the assignment is a valid TOTAL PARTITION, one shard per
    /// window, every index in range, and every window's byte-work conserved onto exactly
    /// one shard (no window dropped or double-counted). The unweighted case is exact
    /// round-robin. This is the invariant the parallel sharded scan relies on to be
    /// byte-identical to a single-device scan regardless of how work is distributed.
    mod shard_assignment_props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(4_000))]

            #[test]
            fn shard_assignment_is_a_valid_total_partition(
                window_lens in proptest::collection::vec(1usize..=1_000_000, 1..64),
                shard_count in 1usize..=8,
                use_weights in any::<bool>(),
                weight_seed in any::<u64>(),
            ) {
                // Deterministic small per-shard weights (0..=4) from the seed; a zero
                // weight is a valid input (the balancer treats it as 1, never starves).
                let weights: Vec<u32> = (0..shard_count)
                    .map(|i| ((weight_seed >> ((i % 8) * 8)) as u32) % 5)
                    .collect();
                let assignment = if use_weights {
                    shard_assignment(&window_lens, shard_count, Some(&weights))
                } else {
                    shard_assignment(&window_lens, shard_count, None)
                };

                // Total function: exactly one shard per window, all in range.
                prop_assert_eq!(assignment.len(), window_lens.len());
                for &shard in &assignment {
                    prop_assert!(
                        shard < shard_count,
                        "assignment {} out of range for {} shards",
                        shard,
                        shard_count
                    );
                }

                // Byte-work conservation: every window's bytes land on exactly one shard,
                // so the per-shard sums reconstruct the total (nothing dropped/doubled).
                let mut per_shard = vec![0u64; shard_count];
                for (index, &shard) in assignment.iter().enumerate() {
                    per_shard[shard] += window_lens[index] as u64;
                }
                let total: u64 = window_lens.iter().map(|&l| l as u64).sum();
                prop_assert_eq!(
                    per_shard.iter().sum::<u64>(),
                    total,
                    "byte-work must be conserved across shards"
                );

                // Unweighted assignment is exact round-robin.
                if !use_weights {
                    for (index, &shard) in assignment.iter().enumerate() {
                        prop_assert_eq!(
                            shard,
                            index % shard_count,
                            "unweighted assignment must be round-robin"
                        );
                    }
                }
            }
        }
    }

    /// The per-shard-timed sharded scan returns a result identical to the untimed
    /// sharded scan plus an honest per-device timing breakdown, the
    /// `per-shard-active-ns` signal the load_balance_policy rebalances on. Runs on
    /// the real GPU; skips cleanly with none.
    #[test]
    fn sharded_timed_scan_reports_honest_per_shard_timing_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping sharded timed GPU test");
                return;
            }
        };
        let device: &dyn VyreBackend = backend.as_ref();

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),    // 9
            b"cccsec".to_vec(),       // 6
            b"retdddsecret".to_vec(), // 12
            b"eeeABfff".to_vec(),     // 8
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let corpus_bytes: u64 = files.iter().map(|file| file.len() as u64).sum();
        let budget = 16usize;
        let max_matches = 4_096u32;

        let set: [&dyn VyreBackend; 2] = [device, device];
        let untimed =
            scan_sharded_fused(&matcher, &set, &file_refs, budget, max_matches).expect("untimed");
        let (timed_result, timing) =
            scan_sharded_fused_timed(&matcher, &set, &file_refs, budget, max_matches)
                .expect("timed sharded scan");

        // Result is byte-identical to the untimed sharded scan (timing is free).
        assert_eq!(
            timed_result, untimed,
            "timed sharded result must equal the untimed sharded result"
        );

        // One timing entry per device, in order.
        assert_eq!(timing.shards.len(), 2);
        assert_eq!(timing.shards[0].shard, 0);
        assert_eq!(timing.shards[1].shard, 1);

        // The plan has 3 windows; round-robin over 2 devices → shard 0 gets 2, shard
        // 1 gets 1, and together they cover every window and every corpus byte.
        let total_windows: u32 = timing.shards.iter().map(|shard| shard.windows).sum();
        let plan_windows = plan_corpus_windows(
            &files.iter().map(|file| file.len()).collect::<Vec<_>>(),
            budget,
        )
        .expect("plan")
        .len() as u32;
        assert_eq!(
            total_windows, plan_windows,
            "per-shard window counts must sum to the total window count"
        );
        assert_eq!(
            timing.shards[0].windows, 2,
            "round-robin gives shard 0 windows 0 and 2"
        );
        assert_eq!(
            timing.shards[1].windows, 1,
            "round-robin gives shard 1 window 1"
        );
        let total_bytes: u64 = timing.shards.iter().map(|shard| shard.bytes_scanned).sum();
        assert_eq!(
            total_bytes, corpus_bytes,
            "per-shard byte-work must sum to the corpus size (overlap excluded)"
        );

        // Each shard that ran windows reports real wall time and a present device
        // time on the GPU (the per-shard-active-ns the rebalancer needs).
        for shard in &timing.shards {
            if shard.windows > 0 {
                assert!(shard.wall_ns > 0, "an active shard must report wall time");
                assert!(
                    shard.device_ns.is_some_and(|ns| ns > 0),
                    "an active shard on the GPU must report non-zero device time"
                );
            }
        }
    }

    /// A pattern database STRIPED across shards (disjoint rule subsets) produces the
    /// exact same global match set as scanning one matcher built over every rule 
    /// the `pattern-database-replicated-shards` parity policy. Exercised with a
    /// 1-device and a 2-device set, plus the fail-closed malformed-shard-map path.
    /// Runs on the real GPU; skips cleanly with none.
    #[test]
    fn pattern_sharded_scan_equals_full_rule_database_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping pattern-sharded GPU parity test");
                return;
            }
        };
        let device: &dyn VyreBackend = backend.as_ref();

        // Full rule database: global ids 0..4.
        let full_patterns: &[&[u8]] = &[b"AKIA", b"secret", b"token", b"AB"];
        let full = GpuLiteralSet::compile(full_patterns);
        let haystack: &[u8] =
            b"AKIA here, a secret token, and AB plus another secret and a token trailing";

        // Ground truth: scan the whole rule database at once (global numbering).
        let mut expected = full.scan_all(device, haystack).expect("full scan");
        expected.sort_unstable();
        assert!(!expected.is_empty(), "the corpus must match some rules");

        // Stripe the rules into two disjoint shards, each with its own sub-matcher
        // and a local→global id map.
        let shard0_matcher = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"token".as_slice()]);
        let shard1_matcher = GpuLiteralSet::compile(&[b"secret".as_slice(), b"AB".as_slice()]);
        let shards = [
            PatternShard {
                matcher: &shard0_matcher,
                global_pattern_ids: &[0, 2], // AKIA→0, token→2
            },
            PatternShard {
                matcher: &shard1_matcher,
                global_pattern_ids: &[1, 3], // secret→1, AB→3
            },
        ];

        // 1-device set: all shards on the one device.
        let striped_one =
            scan_pattern_sharded(&shards, &[device], haystack).expect("striped scan (1 device)");
        assert_eq!(
            striped_one, expected,
            "striped rule-database scan must equal the full-database scan"
        );

        // 2-device set: shard 0 → device 0, shard 1 → device 1 (same GPU here).
        let striped_two = scan_pattern_sharded(&shards, &[device, device], haystack)
            .expect("striped scan (2 devices)");
        assert_eq!(
            striped_two, expected,
            "striped scan across a 2-device set must equal the full-database scan"
        );

        // Fail closed on an empty device set.
        assert!(
            scan_pattern_sharded(&shards, &[], haystack).is_err(),
            "an empty backend set must fail closed"
        );

        // Fail closed on a malformed shard map: a sub-matcher with two rules but a
        // global-id map of length one (a `token` match (local id 1) has no global id).
        let bad_matcher = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"token".as_slice()]);
        let bad = [PatternShard {
            matcher: &bad_matcher,
            global_pattern_ids: &[0], // missing the mapping for local id 1 (token)
        }];
        let error = scan_pattern_sharded(&bad, &[device], b"a token here")
            .expect_err("a malformed shard map must fail closed, not drop the finding");
        assert!(
            error.to_string().contains("global_pattern_ids"),
            "the error must name the missing global id mapping: {error}"
        );
    }

    /// The disk read assembles a window's own bytes plus exactly `overlap` bytes of
    /// the following file (the boundary context the paged scan needs (with no GPU)).
    #[test]
    fn fill_window_from_paths_reads_own_bytes_plus_overlap_prefix() {
        let dir = std::env::temp_dir().join(format!("vyre_paged_fill_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let files: [&[u8]; 3] = [b"AAAA", b"BBBB", b"CCCC"];
        let mut paths = Vec::new();
        for (index, bytes) in files.iter().enumerate() {
            let path = dir.join(format!("{index}.bin"));
            std::fs::write(&path, bytes).expect("write file");
            paths.push(path);
        }
        let path_refs: Vec<&Path> = paths.iter().map(|path| path.as_path()).collect();

        // budget 8 -> window 0 = files[0..2] (8 bytes), window 1 = file[2].
        let lengths = [4usize, 4, 4];
        let windows = plan_corpus_windows(&lengths, 8).expect("plan");
        assert_eq!(windows[0].file_range, 0..2);

        let mut haystack = Vec::new();
        let gathered =
            fill_window_from_paths(&path_refs, &windows[0], 3, &mut haystack).expect("fill");

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            &haystack[..8],
            b"AAAABBBB",
            "own bytes are the window's files"
        );
        assert_eq!(
            &haystack[8..],
            b"CCC",
            "then exactly 3 overlap bytes of the next file"
        );
        assert_eq!(gathered, 3);
    }

    /// The disk-backed paged scan must produce EXACTLY the same result as the
    /// in-memory paged scan over the same bytes, bounded host RSS changes no result
    /// bit. Writes the corpus to a temp dir and scans it by path on the real GPU.
    #[test]
    fn scan_paths_paged_equals_in_memory_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping disk-paged GPU parity test");
                return;
            }
        };

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),
            b"cccsec".to_vec(),
            b"retdddsecret".to_vec(),
            b"eeeABfff".to_vec(),
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 16usize;
        let max_matches = 4_096u32;

        let dir = std::env::temp_dir().join(format!("vyre_paged_paths_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let mut paths = Vec::new();
        for (index, bytes) in files.iter().enumerate() {
            let path = dir.join(format!("{index:02}.bin"));
            std::fs::write(&path, bytes).expect("write file");
            paths.push(path);
        }
        let path_refs: Vec<&Path> = paths.iter().map(|path| path.as_path()).collect();

        let in_memory =
            scan_paged_fused(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
                .expect("in-memory paged scan");
        let from_disk =
            scan_paths_paged(&matcher, backend.as_ref(), &path_refs, budget, max_matches)
                .expect("disk paged scan");

        let _ = std::fs::remove_dir_all(&dir);

        // Multi-window so the disk path reads more than one window.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        assert!(plan_corpus_windows(&lengths, budget).expect("plan").len() >= 2);
        assert_eq!(
            from_disk, in_memory,
            "the disk-backed paged scan must equal the in-memory paged scan"
        );
        assert!(!from_disk.matches.is_empty(), "non-vacuous");
    }

    /// The prefetched disk scan (background reader thread overlapping disk I/O with
    /// device compute) must produce EXACTLY the same result as the synchronous disk
    /// scan, overlapping I/O with compute changes no result bit (Law 10). Writes a
    /// multi-window corpus to a temp dir and scans it both ways on the real GPU.
    #[test]
    fn scan_paths_paged_prefetched_equals_sync_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping prefetched disk-paged GPU test");
                return;
            }
        };

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),
            b"cccsec".to_vec(),
            b"retdddsecret".to_vec(),
            b"eeeABfff".to_vec(),
        ];
        let budget = 16usize;
        let max_matches = 4_096u32;

        let dir = std::env::temp_dir().join(format!("vyre_paged_prefetch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let mut paths = Vec::new();
        for (index, bytes) in files.iter().enumerate() {
            let path = dir.join(format!("{index:02}.bin"));
            std::fs::write(&path, bytes).expect("write file");
            paths.push(path);
        }
        let path_refs: Vec<&Path> = paths.iter().map(|path| path.as_path()).collect();

        let sync = scan_paths_paged(&matcher, backend.as_ref(), &path_refs, budget, max_matches)
            .expect("synchronous disk paged scan");
        let prefetched = scan_paths_paged_prefetched(
            &matcher,
            backend.as_ref(),
            &path_refs,
            budget,
            max_matches,
        )
        .expect("prefetched disk paged scan");

        let _ = std::fs::remove_dir_all(&dir);

        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        assert!(plan_corpus_windows(&lengths, budget).expect("plan").len() >= 2);
        assert_eq!(
            prefetched, sync,
            "the prefetched disk paged scan must equal the synchronous disk paged scan"
        );
        assert!(!prefetched.matches.is_empty(), "non-vacuous");
    }

    /// The async pipelined paged scan must produce EXACTLY the same result as the
    /// synchronous one, overlapping window staging with device execution changes no
    /// result bit (Law 10) (on the same multi-window, boundary-straddling corpus).
    /// Runs on the real GPU; skips cleanly with none.
    #[test]
    fn async_paged_fused_scan_equals_sync_on_gpu() {
        use vyre_driver_wgpu::WgpuBackend;

        let backend = match WgpuBackend::shared() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("no wgpu backend ({error}); skipping async paged GPU parity test");
                return;
            }
        };

        let patterns: &[&[u8]] = &[b"XYZ", b"secret", b"AB"];
        let matcher = GpuLiteralSet::compile(patterns);
        let files: Vec<Vec<u8>> = vec![
            b"aaaXYZbbb".to_vec(),
            b"cccsec".to_vec(),
            b"retdddsecret".to_vec(),
            b"eeeABfff".to_vec(),
        ];
        let file_refs: Vec<&[u8]> = files.iter().map(|file| file.as_slice()).collect();
        let budget = 16usize;
        let max_matches = 4_096u32;

        // Multi-window so the pipeline actually overlaps >=2 in-flight dispatches.
        let lengths: Vec<usize> = files.iter().map(|file| file.len()).collect();
        assert!(
            plan_corpus_windows(&lengths, budget).expect("plan").len() >= 2,
            "the test corpus must span multiple windows"
        );

        let sync = scan_paged_fused(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
            .expect("sync paged scan");
        let overlapped =
            scan_paged_fused_async(&matcher, backend.as_ref(), &file_refs, budget, max_matches)
                .expect("async paged scan");

        assert_eq!(
            overlapped, sync,
            "the async pipelined paged scan must equal the synchronous paged scan bit-for-bit"
        );
        assert!(
            !overlapped.matches.is_empty(),
            "the corpus must produce matches (non-vacuous)"
        );
        assert!(
            overlapped
                .matches
                .iter()
                .any(|hit| hit.pattern_id == 1 && hit.region_id == 1),
            "the window-boundary-spanning `secret` (region 1) must be found in the async path too"
        );
    }
}
