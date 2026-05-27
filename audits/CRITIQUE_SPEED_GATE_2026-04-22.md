# CRITIQUE_SPEED_GATE_2026-04-22

Scope: audit the actual `surgec scan` execution path against the RELEASE gate requirement that the smallest winning cell across rules × corpora × competitors be at least `1000x`.

Method:
- Local code audit of the current `surgec` and `vyre` hot path.
- Commit-window audit for `e8a7339667` through `458fd412f0`.
- Competitor numbers normalized to the extent their publishers make them public.
- Back-of-envelope estimates use these explicit assumptions when code does not measure them today:
  - NVMe sequential read: `7 GB/s` conservative, `12 GB/s` optimistic.
  - Host memcpy: `40 GB/s` sustained.
  - gzip inflate: `500 MB/s/core`.
  - Batch comparator for dispatch math: `4096 files/dispatch`.

Bottom line: the smallest speedup we can prove today is **`0.0x`** because there is no end-to-end competitor harness in-tree, no published `surgec` numerator for any competitor cell, and the actual scan path still pays eager host ingest, per-file/per-rule dispatch, CPU-serial decode, and cold-process pipeline creation costs that prevent the 1000x gate from even being measured honestly.

## Findings

1. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:174-212` | `scan_gpu_with_context()` reads every file with `fs::read` before any GPU dispatch. For a `100 GB` corpus, host read alone costs `14.3 s` at `7 GB/s` or `8.3 s` at `12 GB/s`. Vyre's own target for zero-CPU ingest is `25 GB/s`, which would be `4.0 s` for the same corpus, so the current ingest path burns `4.3-10.3 s` before matching starts. | Replace `fs::read` ingestion with the `AsyncUringStream`/`NvmeGpuIngestDriver` path and make the zero-copy path the default scanner data plane.

2. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:232-266` | `collect_files()` duplicates the same eager host-materialization policy for the legacy collection path. This means the codebase contains two whole-file read funnels, both of which enforce `100 GB -> 100 GB host RAM traffic` before dispatch. | Delete the duplicate eager collector path and route all file acquisition through one streaming ingest layer.

3. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:555,903-909` | `pack_bytes_as_u32_words()` expands every source byte to a `u32`, so a `100 GB` corpus becomes a `400 GB` packed haystack. This is a `4.0x` memory amplification before the first literal prepass dispatch. | Keep haystacks byte-packed on device and adapt substring/AC kernels to consume packed bytes directly.

4. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:840-845` | Single-literal prepass dispatch copies the entire packed haystack with `packed_haystack.to_vec()`. On a `100 GB` corpus that is a second `400 GB` host copy. Even at `40 GB/s` memcpy, that one copy is another `10.0 s` wall time. | Change `gpu_hits_for_literal()` to borrow or stream device buffers instead of cloning owned `Vec<u8>` inputs.

5. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:671-679` | Multi-literal AC prepass does the same `packed_haystack.to_vec()` copy. On the same `100 GB` corpus, every multi-literal signal pays another `400 GB` copy before upload. | Keep the prepass haystack resident and fan out multiple signals over one persistent device buffer.

6. CRITICAL | `libs/performance/matching/vyre/vyre-driver-wgpu/src/pipeline_persistent.rs:245-253,393-421` | Even `dispatch_borrowed()` does not stay borrowed on the wgpu backend. `legacy_contents()` allocates a padded `Vec<u8>` and `queue.write_buffer()` uploads it for every input binding. For a `400 GB` packed haystack, the "borrowed" path still creates another `400 GB` staging copy, or roughly `10.0 s` at `40 GB/s` memcpy. | Introduce true zero-copy/buffer-alias input handles so host slices are not recopied into fresh staging vectors per dispatch.

7. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:882-894` | `decode_u32_buffer()` turns every GPU match buffer back into a host `Vec<u32>`. A dense literal-result buffer sized to haystack length means `4 bytes * N positions`; for a `100 GB` corpus, worst-case readback is another `400 GB`. | Compact matches on device and return sparse hit lists instead of dense per-byte `u32` result vectors.

8. CRITICAL | `libs/performance/matching/vyre/vyre-runtime/src/uring/stream.rs:1-8,129-147,195-225` | The zero-copy ingest path exists and explicitly supports DMA into GPU-visible memory and BAR1 peer memory, but `surgec` never calls it. The bypass is total, not partial: the active scanner path is `fs::read`-based, not `io_uring`-based. | Wire `surgec scan` to `AsyncUringStream` and make eager host reads an opt-out debugging path, not the production path.

9. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:174-212`; `libs/performance/matching/vyre/vyre-runtime/src/uring/stream.rs:195-225`; `libs/performance/matching/vyre/docs/BENCHMARKS.md:51` | Relative to the in-tree `25 GB/s` zero-streaming target, current eager host ingest caps the scan at `7-12 GB/s` on the file-read leg. That is only `28%-48%` of the target before packing, upload, and result traffic are counted. | Treat `25 GB/s` as the required floor for the scanner path itself, not just the runtime bench, and fail CI until the scanner uses the same path.

10. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:309-347` | The hot loop is `for layer in layers` then `for plan in dispatch_plans`, which means dispatch count scales as `files * layers * rules`. For `1,000,000` files and `100` rules, raw-only minimum clause dispatches are `100,000,000`. | Batch files into multi-file megakernel work queues so dispatch count scales with batch count, not file count.

11. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:29-35,80-94,116-160`; `libs/tools/surgec/src/scan/collector.rs:269-287,309-347` | Default decode recursion can create up to `4` total scan layers (`raw + 3 decoded depths` is the realistic hot-path multiplier under the current recursion ceiling). That raises the `1,000,000 files * 100 rules` minimum from `100,000,000` clause dispatches to roughly `400,000,000`. | Collapse decode + match into one batched pipeline so decoded layers are queued into the same device batch instead of becoming independent per-file dispatch loops.

12. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:557-583,626-733` | Clause dispatches are not the full count. Before each clause dispatch, `build_clause_inputs()` performs a per-signal hit discovery pass. If `100` rules average `10` signals each, raw-only hit-discovery dispatches are on the order of `1,000,000,000` for a `1M`-file corpus, before decode-layer multiplication. | Precompute all signal matches for a batch in one megakernel or encode all literals for a clause family into one device program.

13. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:320-328,526-603` | `build_clause_inputs()` runs inside the per-layer/per-plan loop, so counts/offsets/lengths buffers are rebuilt from scratch for every file-rule pair. At `100,000,000` raw-only clause preparations, even a modest `4 KB` metadata/input package produces `~381 GB` of extra host-side transient traffic. | Build reusable batched offset tables once per batch and reuse them across clauses where the source file batch is unchanged.

14. CRITICAL | `libs/performance/matching/vyre/docs/BENCHMARKS.md:52` | Vyre's own dispatch-overhead gate is only `>= 200K dispatches/sec`. At that rate, `100,000,000` raw-only clause dispatches take `500 s` (`8.3 min`) even if dispatch itself were the only cost. `400,000,000` dispatches take `2,000 s` (`33.3 min`). The 1000x claim dies on dispatch count alone. | The scanner needs a batched megakernel path that reduces dispatch count by `10^5-10^6x`, not micro-optimizations inside the current loop.

15. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:309-347`; `libs/performance/matching/vyre/docs/BENCHMARKS.md:52` | Against a `4096-file` batch model, `1,000,000` files need only `245` batch dispatches. Current raw-only `100,000,000` clause dispatches are `408,163x` more dispatches than a batched megakernel. The `4`-layer case is `1,632,653x` worse. | Replace per-file clause submission with batched work queues and a single kernel launch per file-batch.

16. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:663-669,838` | Hit-discovery programs embed `haystack_len = file_bytes.len()` into the compiled program. Distinct file sizes therefore create distinct pipeline fingerprints. On a corpus with `1,000,000` distinct file sizes, even one literal-prepass shape can induce `1,000,000` unique pipelines. | Make haystack length a runtime parameter/buffer input, not part of the compiled program identity.

17. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:663-669,838`; `libs/performance/matching/vyre/vyre-driver-wgpu/src/pipeline.rs:195-214,303-309` | With `100` rules and one size-specialized prepass program each, `1,000,000` distinct file sizes imply up to `100,000,000` cold pipeline creations. Even if each cold pipeline cost only the in-tree `2 ms` compile-to-dispatch target, that is `200,000 s` or `55.6 h` of compile time. | Remove file-size specialization from pipeline keys and compile one reusable prepass pipeline per operation family.

18. CRITICAL | `libs/performance/matching/vyre/vyre-driver-wgpu/src/pipeline.rs:154-174` | `compile_with_config()` calls `runtime::init_device()` and builds a fresh pool on the standalone path. Any cold `surgec scan` process pays device bootstrap and pipeline setup again. There is no process-spanning warm start for the scanner binary. | Move scanner pipeline setup into a persistent daemon or a serialized compiled-pipeline cache that survives process boundaries.

19. CRITICAL | `libs/performance/matching/vyre/vyre-driver-wgpu/src/pipeline.rs:195-214,241-309` | A disk-cache hit only avoids WGSL regeneration. The code still recreates bind group layouts, pipeline layout, shader module, and compute pipeline on every cold process. This means the cache is warm for source text, not for executable pipelines. | Persist compiled pipeline artifacts or keep a long-lived scan service process that amortizes compilation over many scans.

20. CRITICAL | `libs/performance/matching/vyre/vyre-driver-wgpu/src/pipeline_disk_cache.rs:32-68,216-227` | The disk cache stores only `*.wgsl` and `*.toml` metadata under `~/.cache/vyre/pipeline`; it does not store compiled pipelines. The warm-start semantics required by `surgec scan` are therefore absent across CLI invocations. | Introduce a versioned compiled-pipeline cache, or explicitly scope the 1000x gate to a resident daemon that retains compiled pipelines.

21. CRITICAL | `libs/performance/matching/vyre/vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs:5-16,32-67` | `compile_compute_pipeline_with_layout()` is documented and implemented as uncached. Shader module creation and compute pipeline creation therefore remain in the cold path even after a disk-cache hit. | Add a persistent compiled-pipeline store keyed by adapter fingerprint + program ABI, not just WGSL source.

22. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:116-160` | Decode is fully CPU-serial: it walks encodings one by one, then candidates one by one, then recurses depth-first. There is no parallel file-level or region-level decode scheduling here. | Push decode extraction and transform stages into a parallel worker pool or onto the GPU so decode cannot serialize the scanner.

23. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:80-94,125-138,416-439` | Default decode budget is `64 MB` total decoded bytes per file. At `500 MB/s/core` gzip inflate, a worst-case file spends `128 ms` in serial inflate alone. Across `1,000,000` files that is `128,000 s` or `35.6 h` of one-core decode time. | Parallelize decode across files and regions; cap per-file serial work tighter until on-device decode lands.

24. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:416-439` | Gzip extraction uses `flate2::read::GzDecoder` with `read_to_end`, which means full serialized inflate into a host `Vec<u8>`. A zip bomb that expands to the `16 MB` gzip cap still forces a linear CPU inflate before the scanner can continue. | Replace `read_to_end` with chunked parallel decode or device-side inflate that streams directly into scan buffers.

25. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:129-159` | Every decode candidate recursively re-enters `decode_recursive()` before appending the current layer, which serializes nested encoding pipelines. A `base64 -> gzip -> hex` payload cannot overlap stages; it is forced into a depth-first CPU chain. | Convert nested decode into a work queue so independent decode candidates and layers execute concurrently.

26. HIGH | `libs/tools/surgec/src/scan/collector.rs:637-648,771-814` | Regex-backed signals bypass the GPU and run `regex::bytes::Regex::find_iter()` on CPU for every file and every regex pattern. Any ruleset with regex-heavy signals loses the GPU advantage entirely on that branch. | Add a batched GPU regex engine or isolate regex-heavy rules out of the 1000x gate until they have a competitive path.

27. HIGH | `libs/tools/surgec/src/scan/collector.rs:740-768` | `compiled_dfa_for_literals()` guards the global DFA cache with `Mutex<HashMap<...>>`. On a many-file scan, every cache hit and miss for literal sets contends on the same mutex. | Replace the global mutex with a sharded lock-free cache or compile DFA tables at build time for shipped rule sets.

28. HIGH | `libs/tools/surgec/src/scan/collector.rs:610-616` | `group_patterns_by_string_id()` clones every `CompiledPattern`. For `100` rules with `10` signals each across `1,000,000` files, even `1,000` clone operations/file is `1,000,000,000` pattern clones before matching. | Pre-index patterns once at compile/load time and pass shared references in the hot path.

29. HIGH | `libs/tools/surgec/src/scan/collector.rs:552-599` | `build_clause_inputs()` allocates fresh `counts`, `offsets`, `lengths`, `metadata`, and packed byte vectors for every file-rule pair. With `100,000,000` raw-only clause preparations, even `~12 KB` of transient host allocation per clause implies `~1.1 TB` of total allocator churn. | Reuse slab-backed buffers per batch and keep metadata resident instead of rebuilding vectors per clause.

30. HIGH | `libs/tools/surgec/src/scan/collector.rs:595` | `offsets: Arc::from(offsets.clone())` doubles the offsets buffer at the point of return. With `MAX_CACHED_POSITIONS` slots across many signals, this creates an avoidable second full copy in the hottest per-clause path. | Build the `Arc<[u32]>` directly from the original offsets allocation and stop cloning it.

31. HIGH | `libs/tools/surgec/src/scan/dispatch.rs:96-113` | `dispatch_rules()` calls `decode_offsets_input(inputs)?` once per clause-document dispatch, which decodes the full offsets buffer back into a fresh `Vec<u32>`. That is another full host copy of the slot-offset table for every clause dispatch. | Keep offsets in typed host memory once, or resolve slots on device and emit byte offsets directly.

32. HIGH | `libs/tools/surgec/src/scan/dispatch.rs:149-226` | `dispatch_rule()` still loops per clause and does host-side slot decode, byte-offset translation, confidence scoring, and provenance construction after each GPU dispatch. The GPU is not fed in batches; the CPU remains in the control loop for every clause. | Move result reduction, offset translation, and confidence scaffolding into a batched post-pass instead of coupling them to each dispatch.

33. HIGH | `libs/tools/surgec/src/main.rs:313-323`; `libs/tools/surgec/src/output/tfidf.rs:20-59` | The recent TF-IDF pass adds a second full `WalkDir` file count and two string-heavy hash-map passes over findings. For `1,000,000` files, the extra `WalkDir` is another `1,000,000` directory entries touched after the scan has already paid to walk them once. | Gate TF-IDF behind an offline post-processor and keep the benchmarked scan path free of corpus-ranking passes.

34. HIGH | `libs/tools/surgec/src/output/tfidf.rs:21-55` | TF-IDF ranking clones path strings and rule strings repeatedly into `HashMap<String, HashSet<String>>` and `HashMap<String, HashMap<String, usize>>`. At `10,000,000` findings with average `80-byte` path+rule payload, this is on the order of gigabytes of avoidable heap traffic. | Store interned IDs, not owned `String`s, in ranking maps.

35. HIGH | `libs/tools/surgec/src/main.rs:325-333`; `libs/tools/surgec/src/scan/auto_suppress.rs:28-56` | Auto-suppression adds another O(findings) pass that clones `(rule_name, file_path)` into a `HashMap<(String, String), usize>`. On `10,000,000` findings, that is another tens-to-hundreds of MB of string churn after the scan completes. | Make suppression proposal generation an explicit offline command, not part of the hot scan lane.

36. CRITICAL | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/wgpu_dispatch.rs:48-52,79-107` | The recent landing wave added a host-side megakernel wrapper that reads the work queue back from GPU into `Vec<u8>`, rebuilds ring bytes on the host, allocates `io_queue_bytes`, and then redispatches. That is the opposite of a persistent device-side queue: it inserts readback, serialization, and allocation into the supposed batching path. | Keep the work queue and ring resident on device and submit megakernel dispatches without host round-trips.

37. HIGH | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/wgpu_dispatch.rs:97` | `let io_queue_bytes = vec![0u8; 64 * 8 * 4];` allocates a fresh `2048-byte` queue per megakernel dispatch. The allocation is small, but it is still a per-dispatch heap allocation in a path that is supposed to demonstrate dispatch amortization. | Reuse a preallocated queue buffer or make it part of the persistent megakernel state.

38. CRITICAL | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:212-217,388-396` | The gate requires `benches/competition/`, `surgec/benches/vs_competition.rs`, and `docs/BENCHMARK.md`. None of those artifacts exist today. Without them, the smallest winning factor cannot be measured, published, or regenerated. | Implement the benchmark harness before making any 1000x claim.

39. CRITICAL | `libs/performance/matching/vyre/docs/BENCHMARKS.md:47,51`; `https://www.intel.com/content/www/us/en/collections/libraries/hyperscan/performance-analysis-hyperscan-hsbench.html` | The only clean throughput baseline in the target set is Hyperscan. Intel publishes `5861 megabits/sec` for one single-core sample on HTTP traffic. That is about `732.6 MB/s` or `0.733 GB/s`. A `1000x` win over that cell would require `~733 GB/s`, but Vyre's own zero-streaming goal is only `25 GB/s`, a `29.3x` gap. | Narrow the gate to a workload class where the claimed advantage is physically plausible, or raise the internal data-plane target by more than an order of magnitude.

40. CRITICAL | `libs/performance/matching/vyre/docs/BENCHMARKS.md:47,51`; `https://www.intel.com/content/www/us/en/collections/libraries/hyperscan/performance-analysis-hyperscan-hsbench.html` | Intel's other published Hyperscan sample is `3355 megabits/sec` on Gutenberg text, about `419.4 MB/s` or `0.419 GB/s`. Even that easier cell would demand `~419 GB/s` to clear `1000x`, still `16.8x` above Vyre's own `25 GB/s` ingest target. | Stop talking about a generic 1000x "vs Hyperscan" gate until the gate is normalized to a physically consistent benchmark shape.

41. HIGH | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:214-217`; `https://semgrep.dev/blog/2023/semgrep-speed/` | Semgrep publishes wall-clock repo-scan numbers, not bytes/sec: OSS average CI scans "just under `10 s`", average full scans around `20 s`, and Pro full scans under `300 s`. A `1000x` gate against those cells means proving `20 ms` on the same full-repo workload or `300 ms` on the Pro workload. No in-tree harness measures that shape today. | Define corpus size, ruleset, and wall-clock normalization for Semgrep explicitly and build a reproducible benchmark for it.

42. HIGH | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:214-217`; `https://github.blog/changelog/2026-03-24-faster-incremental-analysis-with-codeql-in-pull-requests/`; `https://docs.github.com/en/code-security/code-scanning/creating-an-advanced-setup-for-code-scanning/recommended-hardware-resources-for-running-codeql` | CodeQL publishes hardware guidance and scan-duration buckets, not lines/sec. GitHub's current public wording groups non-incremental scans into `<3 min`, `3-7 min`, and `>7 min`, and recommends up to `8 cores / 64 GB RAM` for `>1M LOC`. A `1000x` claim against the `>7 min` bucket implies `<420 ms` on a large-repo semantic analysis workload. We do not measure that workload at all. | Either build a like-for-like semantic-analysis benchmark or remove CodeQL from the 1000x gate until the workload is made comparable.

43. HIGH | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:214-217`; `https://snyk.io/blog/sast-tools-speed-comparison-snyk-code-sonarqube-lgtm/` | Snyk's public comparison is against LGTM and SonarQube on `48` JS repos using a `16-core / 64 GB` Xeon. Their own published averages are `22 s` for Snyk Code and `312 s` for LGTM. A `1000x` win over the `22 s` cell would require `22 ms`. No such surgec number exists in-tree. | Build the same repo set or an equivalently disclosed corpus, then publish a like-for-like wall-clock comparison.

44. HIGH | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:214-217`; `https://docs.projectdiscovery.io/opensource/nuclei/mass-scanning-cli`; `https://projectdiscovery.io/blog/how-elastic-scaled-proactive-detection-with-projectdiscovery-cloud` | Nuclei's public numbers are HTTP/network-oriented: default global rate limit `150 req/s`, and one public case study reports `14,500` assets in under `5 min` (`48.3 assets/s`). Those are network workloads, not local corpus scans. A `1000x` gate against `150 req/s` means `150,000 req/s`, which is not even the same dimension as `surgec`'s current code-scanning path. | Remove Nuclei from the scanner-throughput gate unless the benchmark is redefined around equivalent network scanning tasks.

45. CRITICAL | `libs/performance/matching/vyre/audits/RELEASE_GATE.md:216-217,395-396`; `libs/performance/matching/vyre/benches/RESULTS.md` | There is no published `surgec` numerator against any competitor/corpus cell. Existing results cover microbenches like fingerprinting and megakernel encode costs, not `surgec vs competitor` end-to-end scans. Quantitatively, the smallest speedup we can **prove** today is therefore **`0.0x`** on every competitor cell because no admissible measurement exists. | Do not claim readiness for the 1000x gate until the benchmark harness exists and emits reproducible competitor and surgec timings for the same corpus and rules.

## Confidence Interval

The smallest speedup we can prove today is **`0.0x`**.

Reason:
- The RELEASE gate requires the minimum winning cell across corpora × rules × competitors.
- There is no `benches/competition/`, no `surgec/benches/vs_competition.rs`, and no regenerated `docs/BENCHMARK.md`.
- Existing in-tree measurements are microbenchmarks, not end-to-end `surgec` scans against any competitor on a shared corpus.
- Without a measured numerator, the claimable speedup is zero, not "unknown", because the gate is a proof obligation and no proof has been produced.

## Primary-Source References

- Intel Hyperscan hsbench sample throughput:
  - https://www.intel.com/content/www/us/en/collections/libraries/hyperscan/performance-analysis-hyperscan-hsbench.html
- Semgrep published scan times:
  - https://semgrep.dev/blog/2023/semgrep-speed/
- CodeQL published scan-bucket and hardware guidance:
  - https://github.blog/changelog/2026-03-24-faster-incremental-analysis-with-codeql-in-pull-requests/
  - https://docs.github.com/en/code-security/code-scanning/creating-an-advanced-setup-for-code-scanning/recommended-hardware-resources-for-running-codeql
- Snyk Code published speed comparison:
  - https://snyk.io/blog/sast-tools-speed-comparison-snyk-code-sonarqube-lgtm/
- Nuclei public throughput knobs and case-study numbers:
  - https://docs.projectdiscovery.io/opensource/nuclei/mass-scanning-cli
  - https://projectdiscovery.io/blog/how-elastic-scaled-proactive-detection-with-projectdiscovery-cloud
