# C parser Linux subsystem proof

This artifact backs `c-parser-linux-subsystem`.

Evidence sources:

Required generated evidence:

- `release/evidence/parser/c-parser-linux-subsystem.json`
- `release/evidence/parser/linux-subsystem-corpus-manifest.json`
- `release/evidence/parser/c-parser-diagnostics-summary.json`
- `release/evidence/parser/c-parser-throughput.json`

Release contract:

- The corpus root must canonicalize to a Linux subsystem checkout path.
- The report and manifest must record `linux_root`, `linux_subsystem`, `linux_subsystem_depth`, `linux_kbuild_file`, and `linux_kbuild_file_in_corpus`.
- The report and manifest must record a stable `corpus_fingerprint` for the exact file list and source byte sizes.
- `linux_subsystem` must be one of `kernel`, `fs`, `mm`, `net`, `drivers`, or `lib`.
- `linux_kbuild_file` must point to a discovered `Makefile`, `Kbuild`, or `Kconfig` sentinel inside the selected corpus root, not merely the Linux repository root.
- The corpus must contain at least `250` C files and `4194304` source bytes.
- The corpus report, manifest, and throughput proof must record `source_collection_mode = recursive_all_c_files` and nonzero `visited_dir_count`, proving the runner recursively collected every `.c` file under the selected Linux subsystem root rather than sampling a subset.
- `linux-subsystem-corpus-manifest.json` must match `c-parser-linux-subsystem.json` on file counts, source bytes, Linux subsystem provenance, recursive source collection provenance, include dirs, macros, file entry count, and `corpus_fingerprint`.
- `c-parser-diagnostics-summary.json` must match `c-parser-linux-subsystem.json` on failed file count and failure entry count; a clean parse report cannot be paired with stale diagnostics.
- `c-parser-throughput.json` must cover the same full corpus floor: at least `250` parsed files, `4194304` source bytes, nonzero wall/time-rate fields, and matching Linux subsystem provenance, include dirs, macros, and `corpus_fingerprint` from the parse report.
- Include directories and macro definitions must be non-empty effective compiler inputs, not only raw CLI flags; the corpus runner infers Linux include roots (`include`, `uapi`, `generated`, `arch/x86`, and `tools/include`) and kernel macros when they are not supplied explicitly, then records the effective values in the report, manifest, diagnostics-adjacent throughput proof, and cross-artifact comparisons.
- Parsed files must equal total files and failed files must be zero.
- Aggregate AST, VAST, and semantic graph section byte counts must all be nonzero.
- Every manifest file entry must report `parsed = true`, nonzero `source_bytes`, nonzero `object_bytes`, nonzero `ast_bytes`, nonzero `vast_bytes`, nonzero `semantic_graph_bytes`, and nonzero `wall_ns`.
