# Command-line interfaces

Applies to Vyre 0.7.2.

This document is generated from `docs/CLI.toml` and the executable `--help`
output of every Cargo binary. Run `python3 scripts/cli_docs.py --check` to
rebuild every binary, execute every help route, and reject drift.

| Package | Binary | Audience | Commands | README |
| --- | --- | --- | --- | --- |
| `vyre-grammar-gen` | `vyre-grammar-gen` | public | `dump-lexer`, `dump-lr`, `emit` | [`vyre-grammar-gen/README.md`](../vyre-grammar-gen/README.md) |
| `vyre-driver-wgpu` | `vyre-wgpu` | public | `demo` | [`vyre-driver-wgpu/README.md`](../vyre-driver-wgpu/README.md) |
| `vyre-conform` | `vyre-conform` | internal | `dispatch`, `merge`, `plan`, `prove` | [`conform/vyre-conform/README.md`](../conform/vyre-conform/README.md) |
| `xtask` | `audit_rule_contracts` | internal | none | [`xtask/README.md`](../xtask/README.md) |
| `xtask` | `lint_shape_tests` | internal | none | [`xtask/README.md`](../xtask/README.md) |
| `xtask` | `public_api_check` | internal | none | [`xtask/README.md`](../xtask/README.md) |
| `xtask` | `scaffold_rule` | internal | none | [`xtask/README.md`](../xtask/README.md) |
| `xtask` | `vyre_new_op` | internal | `new-op` | [`xtask/README.md`](../xtask/README.md) |
| `xtask` | `xtask` | internal | `abstraction-gate`, `backend-matrix`, `bench-crossback`, `bench-release`, `catalog`, `check-cat-a`, `check-tier-deps`, `command-matrix`, `compile`, `conformance-matrix`, `dep-drift`, `docs-check`, `feature-matrix`, `gate1`, `heuristic-audit`, `hot-path-scan`, `hygiene-matrix`, `launch-state`, `lego-audit`, `lego-quick`, `lint-shape-tests`, `list-ops`, `metadata-matrix`, `op-matrix`, `operation-schema`, `optimization-corpus`, `optimization-matrix`, `package-readiness`, `parser-coherence`, `platform-boundary`, `print-composition`, `quick-check`, `recursion-gate`, `release-benchmarks`, `release-completion-audit`, `release-conformance`, `release-evidence`, `release-gate`, `release-workload-matrix`, `shrink`, `source-similar`, `test-matrix`, `trace-f32`, `verify-rewrite-proofs`, `version-matrix`, `vyre-release-gate`, `weir-matrix`, `whats-similar` | [`xtask/README.md`](../xtask/README.md) |
| `vyre-bench` | `vyre-bench` | internal | `compare`, `dashboard`, `evolve-server`, `explain`, `list`, `release-matrix`, `run`, `snapshot-diff`, `validate-benchmark-bundle`, `validate-comparison`, `validate-report` | [`vyre-bench/README.md`](../vyre-bench/README.md) |
| `vyre-debug` | `vyre-dbg` | public | `artifact-report`, `bisect-rewrites`, `carrier-summary`, `diff-descriptors`, `diff-emit`, `dump-descriptor`, `dump-wgsl`, `emit-replay`, `failure-trace`, `find-dangling`, `find-uncarriered`, `pipeline-cache-clear` | [`vyre-debug/README.md`](../vyre-debug/README.md) |
| `vyre-lints` | `vyre-lints` | public | none | [`vyre-lints/README.md`](../vyre-lints/README.md) |

## `vyre-grammar-gen`

Package: `vyre-grammar-gen`. Audience: public.

Hardware: No accelerator is required.

Environment: No environment variables alter CLI behavior.

Configuration: Input and output paths are explicit command-line arguments.

Failure behavior: Invalid JSON, malformed LR data, unsupported formats, and filesystem errors return a non-zero status with context.

Exit codes: 0 on success, 1 on generation or I/O failure, 2 on invalid arguments.

### Top-level help

```text
Compile C11 lexer grammar into a GPU-ready table.

Usage: vyre-grammar-gen <COMMAND>

Commands:
  emit        Emit the C11 lexer table to disk
  dump-lexer  Print a hex dump of the lexer DFA blob to stdout
  dump-lr     Print a hex dump of a caller-supplied LR table blob to stdout
  help        Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### `dump-lexer` help

```text
Print a hex dump of the lexer DFA blob to stdout

Usage: vyre-grammar-gen dump-lexer [OPTIONS]

Options:
      --smoke-lexer  Same as emit: use the tiny synthetic DFA instead of full C11 table
  -h, --help         Print help
```

### `dump-lr` help

```text
Print a hex dump of a caller-supplied LR table blob to stdout

Usage: vyre-grammar-gen dump-lr --lr-json <LR_JSON>

Options:
      --lr-json <LR_JSON>  JSON-encoded `LrTable` to serialize and dump
  -h, --help               Print help
```

### `emit` help

```text
Emit the C11 lexer table to disk

Usage: vyre-grammar-gen emit [OPTIONS]

Options:
      --out-dir <OUT_DIR>
          Output directory
          
          [default: ./rules/c11]

      --smoke-lexer
          Use a tiny synthetic lexer DFA for CLI smoke tests

      --format <FORMAT>
          `bin` (default) or `json` sidecar metadata next to `.bin` files

          Possible values:
          - bin:  Only `.bin` files
          - json: `.bin` plus `.json` sidecars (metadata, not a second wire format)
          
          [default: bin]

      --lr-json <LR_JSON>
          Optional JSON-encoded `LrTable` to serialize as `c11_lr_tables.bin`

  -h, --help
          Print help (see a summary with '-h')
```

## `vyre-wgpu`

Package: `vyre-driver-wgpu`. Audience: public.

Hardware: The demo requires a Vulkan, Metal, DX12, or WebGPU compute device and never falls back to CPU.

Environment: Backend diagnostics honor the driver crate's VYRE_WGPU_TIMESTAMPS, VYRE_CACHE_DIR, VYRE_DUMP_KDESC, and VYRE_CAPTURE_FAILED_DESCRIPTOR controls.

Configuration: The demo has no config file. Device and dispatch configuration come from the WGPU backend defaults.

Failure behavior: Invalid arguments, missing devices, lowering failures, dispatch failures, and malformed output return status 1 with an actionable error.

Exit codes: 0 on help, version, or exact demo success; 1 on argument, device, dispatch, or output failure.

### Top-level help

```text
vyre 0.7.2
Run a minimal Vyre IR program on the local WGPU device.

Usage: vyre [--version] <COMMAND>

Commands:
  demo  dispatch one u32 write and verify the exact result 42

Options:
  -h, --help     print this help
  -V, --version  print the Vyre version

Exit codes:
  0  help, version, or GPU demo completed
  1  invalid arguments, device acquisition, dispatch, or output validation failed
```

### `demo` help

```text
Dispatch one generated Vyre IR program on the local WGPU device.

Usage: vyre demo

Hardware:
  A Vulkan, Metal, DX12, or WebGPU compute device is required.
  The command never falls back to CPU.

Output:
  vyre demo gpu_u32=42
```

## `vyre-conform`

Package: `vyre-conform`. Audience: internal.

Hardware: Dispatch and proof commands require every requested backend device. Planning and merge are device independent.

Environment: VYRE_BACKEND selects a backend. VYRE_CONFORM_PROOF_WORKERS and VYRE_CONFORM_PROOF_* tune proof execution and timing.

Configuration: Operation, backend, shard, certificate, and output selections are command-line arguments.

Failure behavior: Unknown commands, unavailable requested backends, malformed artifacts, conformance mismatches, and I/O failures return non-zero.

Exit codes: 0 on success or help, 1 on execution failure, 2 on invalid arguments.

### Top-level help

```text
usage: vyre-conform dispatch --backend <backend-id|auto> --ops <all|<op_id>>
       vyre-conform plan [--out <plan.json>] [--backend <all|backend-id>] [--ops <all|op_id>] [--shard <index>/<count>]
       vyre-conform merge --out <merged.json> <prove-shard.json>...
       vyre-conform prove [--out <cert.json>] [--certificates <dir>] [--backend <all|backend-id>] [--ops <all|op_id>] [--shard <index>/<count>]  # default: .internals/certs//prove.json
```

## `audit_rule_contracts`

Package: `xtask`. Audience: internal.

Hardware: No accelerator is required.

Environment: No environment variables alter CLI behavior.

Configuration: The command reads the enclosing Santh rules/launch and tests/launch_rule_truth trees.

Failure behavior: A missing rules tree or incomplete rule contract returns status 1.

Exit codes: 0 on complete contracts or help, 1 on contract failure, 2 on invalid arguments.

### Top-level help

```text
Audit launch-rule contracts and truth-test directories.

Usage: audit_rule_contracts

Exit codes:
  0  every rule contract and truth-test directory exists
  1  the rule tree is unavailable or a contract is incomplete
  2  command-line arguments are invalid
```

## `lint_shape_tests`

Package: `xtask`. Audience: internal.

Hardware: No accelerator is required.

Environment: No environment variables alter CLI behavior.

Configuration: --threshold sets the maximum shape-only test percentage.

Failure behavior: Incomplete scans, malformed sources, or a threshold violation return status 1.

Exit codes: 0 within threshold or on help, 1 on scan or threshold failure, 2 on invalid arguments.

### Top-level help

```text
Audit tests for assertions that prove only result shape.

Usage: lint_shape_tests [--threshold <percentage>]

Options:
  --threshold <percentage>  maximum permitted shape-test percentage (default: 0)
  -h, --help                print this help

Output:
  TEST_AUDIT_<date>.md in the enclosing Santh workspace

Exit codes:
  0  scan completed within the threshold
  1  scan failed, was incomplete, or exceeded the threshold
  2  command-line arguments are invalid
```

## `public_api_check`

Package: `xtask`. Audience: internal.

Hardware: No accelerator is required.

Environment: VYRE_CARGO_RUNNER selects the Cargo wrapper.

Configuration: --update refreshes snapshots and --allow-breaking explicitly permits removals.

Failure behavior: Tool launch, API generation, UTF-8, inventory, compatibility, and snapshot mismatches return status 1.

Exit codes: 0 on matching or approved updated snapshots, 1 on API failure, 2 on invalid arguments.

### Top-level help

```text
Check publishable facade APIs against committed snapshots.

Usage: public_api_check [--update [--allow-breaking]]

Options:
  --update          refresh snapshots after compatibility checks
  --allow-breaking  permit removed API items while refreshing
  -h, --help        print this help

Environment:
  VYRE_CARGO_RUNNER  Cargo wrapper to invoke (default: cargo_full)

Exit codes:
  0  snapshots match or were refreshed
  1  API generation, compatibility, or snapshot checks failed
  2  command-line arguments are invalid
```

## `scaffold_rule`

Package: `xtask`. Audience: internal.

Hardware: No accelerator is required.

Environment: No environment variables alter CLI behavior.

Configuration: The slug selects directories in the enclosing Santh rule and truth-test trees.

Failure behavior: Missing slugs and filesystem creation failures return non-zero without claiming success.

Exit codes: 0 on scaffold creation or help, 1 on filesystem failure, 2 on invalid arguments.

### Top-level help

```text
Scaffold one launch-rule contract and truth-test suite.

Usage: scaffold_rule <slug>

Arguments:
  <slug>  launch-rule directory name

Exit codes:
  0  scaffold created
  1  input or filesystem failure
  2  command-line arguments are invalid
```

## `vyre_new_op`

Package: `xtask`. Audience: internal.

Hardware: No accelerator is required.

Environment: VYRE_SPEC_MAINTAINER=1 permits reserved internal. and test. operation identifiers.

Configuration: Operation id, archetype, display name, summary, and category are explicit arguments.

Failure behavior: Invalid identifiers, archetypes, categories, collisions, and write failures return non-zero.

Exit codes: 0 on scaffold creation or help, 1 on validation or write failure, 2 on invalid arguments.

### Top-level help

```text
Usage:
  vyre new-op <id> --archetype <archetype> [--display-name <text>] [--summary <text>] [--category <A|C>]

Examples:
  cargo_full run -p vyre --bin vyre_new_op -- new-op primitive.arithmetic.test_op --archetype binary-arithmetic

Reserved prefixes 'internal.' and 'test.' require VYRE_SPEC_MAINTAINER=1.
```

## `xtask`

Package: `xtask`. Audience: internal.

Hardware: Requirements are command specific. Backend, conformance, and benchmark commands require their declared devices.

Environment: Commands read documented release, backend, benchmark, and Cargo wrapper environment controls.

Configuration: Each subcommand documents its manifest, config, input, and output arguments in top-level help and the generated inventory.

Failure behavior: Unknown commands return non-zero. Each command fails closed when required inputs, devices, or evidence are unavailable.

Exit codes: 0 on help or command success, 1 on command failure or unknown subcommand, 2 where a subcommand rejects arguments.

### Top-level help

```text
vyre xtask runner

USAGE:
cargo_full run --bin xtask -- <subcommand> [options]

SUBCOMMANDS:
quick-check --op NAME               Run minimal <5s verification path for a single op
abstraction-gate                     Enforce registered building-block boundaries
bench-crossback [program]           Cross-backend perf table
backend-matrix [--output PATH]      Probe linked CUDA/WGPU backend release policy
bench-release [--backend all]        Run the legacy cross-backend release benchmark coordinator
shrink <file.vir> <oracle.sh>       Delta-debug a crashing vyre wire formulation down to a minimal reproducer
check-cat-a                         Run every Cat-A pre-merge gate
check-tier-deps                     Reject upward tier path dependencies (T4→T1 only)
command-matrix [--output PATH] [--check] Generate/check xtask command owner/proof matrix
compile <program.vir> --to TARGET   Emit target artifact(s) (wgsl/spirv/secondary_text/native_module/hlsl)
conformance-matrix [--check] [--output PATH] Enumerate/check release op/backend conformance coverage
dep-drift                           Fail if any repo manifest pins a workspace-managed dependency to a different version
docs-check                           Validate manifest-backed documentation lifecycle and generated navigation
feature-matrix [--output PATH]      Generate Vyre/Weir crate feature evidence matrix
print-composition <op_id>           Walk an op's Region tree and print its decomposition chain
trace-f32 <op_id>                   Run an op's test_inputs through vyre-reference and dump expected_output literal
gate1                               Enforce Gate 1 complexity budget (CI floor)
launch-state [--output PATH]       Generate public launch completion state evidence
list-ops [--write PATH|--check]     Render or check the schema-derived operation inventory
metadata-matrix [--output PATH]     Generate Vyre/Weir crate metadata evidence
operation-schema [--output PATH] [--check] [--validate PATH]  Generate or verify the canonical live operation contract schema
op-matrix [--output PATH]           Generate operation/backend coverage evidence
optimization-matrix [--output PATH] Generate release optimization integration evidence
package-readiness [--output PATH]  Generate pre-publish package order evidence
optimization-corpus [--output PATH]  Generate release optimization corpus manifest
parser-coherence [--output PATH]   Generate distributed C parser ownership evidence
platform-boundary                  Fail on consumer names in platform crate docs/comments
version-matrix [--output PATH]      Generate Vyre/Weir manifest version matrix
weir-matrix [--output PATH]         Generate Weir analysis API evidence matrix
catalog [--out DIR] [--check]       Emit one markdown table per subsystem under docs/catalog; --check gates drift
release-gate                        Pre-publish sanity checks (catalog + gate1 + Cargo.lock clean)
release-workload-matrix [--output PATH]  Generate cheap release workload family evidence
release-benchmarks [--backend cuda] Generate long-running release benchmark artifacts
release-conformance [--backend all] Generate real backend conformance artifacts
release-completion-audit [--output PATH]  Generate final prompt-to-artifact audit evidence
release-evidence                    Generate cheap structural release evidence artifacts
vyre-release-gate [--prepublish] [--manifest PATH]  Enforce final or prepublication evidence closure
recursion-gate [--strict]           Enforce recursion thesis (every Tier-2.5 primitive has a vyre-self consumer)
heuristic-audit [--strict]          Surface hand-rolled heuristics that should be self-consumer calls
verify-rewrite-proofs               Verify optimizer rewrite proof fixtures
hygiene-matrix [--output PATH]      Scan Vyre/Weir source hygiene release blockers
lego-audit [--report-only|--with-repo|--write-baseline] [--duplicate-report-json PATH] Deeper LEGO-block enforcement and composition baseline management
lego-quick [--all] [--source-similar] Fast pre-commit gate plus optional source-dedup scan
whats-similar (--op-id <id>|--all) [--duplicate-report-json PATH] Pre-write/all-pairs duplicate query by IR shape
source-similar [--root PATH] [--check] [--include-untracked] [--duplicate-report-json PATH] Repo-wide Rust source duplicate scanner
hot-path-scan [--strict]            Scan files in HOT_PATHS.toml for clone/alloc/lock patterns
test-matrix [--output PATH]         Generate Vyre/Weir test architecture evidence
lint-shape-tests [--strict]         Scan test modules for shape-only assertions

--help                              Print this message
```

## `vyre-bench`

Package: `vyre-bench`. Audience: internal.

Hardware: Run commands require the explicitly selected backend device. Report validation and comparison are device independent.

Environment: RAYON_NUM_THREADS configures CPU baselines. VYRE_ALLOW_FEW_SAMPLES=1 permits local smoke runs below the release sample floor.

Configuration: Suite, case, backend, sample, budget, report, and output settings are command-line arguments.

Failure behavior: Unavailable backends, invalid suites or reports, benchmark mismatches, timeouts, and budget violations return non-zero.

Exit codes: 0 on success or help, 1 on benchmark or validation failure, 2 on invalid arguments.

### Top-level help

```text
Canonical performance and evolution harness for Vyre

Usage: vyre-bench <COMMAND>

Commands:
  run                        
  compare                    
  validate-report            
  validate-comparison        
  validate-benchmark-bundle  
  snapshot-diff              
  list                       
  explain                    
  dashboard                  
  release-matrix             
  evolve-server              
  help                       Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

## `vyre-dbg`

Package: `vyre-debug`. Audience: public.

Hardware: Descriptor analysis is device independent. Emission and replay requirements depend on the selected artifact and backend.

Environment: HOME selects the default pipeline cache location used by pipeline-cache-clear.

Configuration: Programs, descriptors, traces, and output modes are explicit subcommand arguments.

Failure behavior: Missing or malformed artifacts, emission failures, and cache filesystem failures return non-zero.

Exit codes: 0 on success or help, 1 on artifact or operation failure, 2 on invalid arguments.

### Top-level help

```text
Usage: vyre-dbg <COMMAND>

Commands:
  artifact-report       
  dump-descriptor       
  dump-wgsl             
  find-dangling         
  find-uncarriered      
  carrier-summary       
  bisect-rewrites       
  diff-descriptors      
  failure-trace         
  emit-replay           
  diff-emit             
  pipeline-cache-clear  
  help                  Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

### `artifact-report` help

```text
Usage: vyre-dbg artifact-report --envelope <ENVELOPE>

Options:
      --envelope <ENVELOPE>  
  -h, --help                 Print help
```

### `bisect-rewrites` help

```text
Usage: vyre-dbg bisect-rewrites [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
  -h, --help                     Print help
```

### `carrier-summary` help

```text
Usage: vyre-dbg carrier-summary [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
      --json                     
  -h, --help                     Print help
```

### `diff-descriptors` help

```text
Usage: vyre-dbg diff-descriptors --prog-a <PROG_A> --prog-b <PROG_B>

Options:
      --prog-a <PROG_A>  
      --prog-b <PROG_B>  
  -h, --help             Print help
```

### `diff-emit` help

```text
Usage: vyre-dbg diff-emit --kdesc-a <KDESC_A> --kdesc-b <KDESC_B>

Options:
      --kdesc-a <KDESC_A>  
      --kdesc-b <KDESC_B>  
  -h, --help               Print help
```

### `dump-descriptor` help

```text
Usage: vyre-dbg dump-descriptor [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
  -h, --help                     Print help
```

### `dump-wgsl` help

```text
Usage: vyre-dbg dump-wgsl [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
      --lines                    
  -h, --help                     Print help
```

### `emit-replay` help

```text
Usage: vyre-dbg emit-replay --kdesc <KDESC>

Options:
      --kdesc <KDESC>  
  -h, --help           Print help
```

### `failure-trace` help

```text
Usage: vyre-dbg failure-trace --dir <DIR> --id <ID>

Options:
      --dir <DIR>  
      --id <ID>    
  -h, --help       Print help
```

### `find-dangling` help

```text
Usage: vyre-dbg find-dangling [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
      --json                     
  -h, --help                     Print help
```

### `find-uncarriered` help

```text
Usage: vyre-dbg find-uncarriered [OPTIONS] --prog <PROG>

Options:
      --prog <PROG>              
      --num-tokens <NUM_TOKENS>  
      --json                     
  -h, --help                     Print help
```

### `pipeline-cache-clear` help

```text
Usage: vyre-dbg pipeline-cache-clear

Options:
  -h, --help  Print help
```

## `vyre-lints`

Package: `vyre-lints`. Audience: public.

Hardware: No accelerator is required.

Environment: No environment variables alter CLI behavior.

Configuration: Workspace root, allowlist, format, library roots, and evidence outputs are explicit options.

Failure behavior: Malformed configuration, unreadable sources, and lint findings return non-zero with file and rule context.

Exit codes: 0 with no blocking findings or on help, 1 on lint or I/O failure, 2 on invalid arguments.

### Top-level help

```text
Lego-block enforcement lints for vyre

Usage: vyre-lints [OPTIONS]

Options:
      --workspace-root <WORKSPACE_ROOT>
          Workspace root (the dir containing vyre-libs/, vyre-foundation/, ...) [default: .]
      --allowlist <ALLOWLIST>
          Allowlist file. If omitted, defaults to <workspace_root>/vyre-lints/allowlist.toml
      --format <FORMAT>
          Output format: text (default) or json [default: text] [possible values: text, json]
      --lib-root <LIB_ROOT>
          Override the lib roots scanned. Defaults to vyre-libs/src
      --check-drift
          Run the allowlist drift sentinel: fail if any allowlist entry is older than `--drift-budget-days` (default 14). Skips the raw-IR scan when set
      --drift-budget-days <DRIFT_BUDGET_DAYS>
          Age budget for the drift sentinel, in days [default: 14]
      --today <TODAY>
          Today's date in YYYY-MM-DD form. Defaults to the OS clock
      --check-production-cpu-fallbacks
          Run the production CPU fallback guard instead of the raw-IR lint
      --production-root <PRODUCTION_ROOT>
          Override production roots scanned by `--check-production-cpu-fallbacks`. Defaults to Vyre-owned production crates, excluding reference/conform crates. External consumers can be scanned by passing this flag repeatedly
      --check-consumer-coupling
          Run the consumer-name coupling guard over platform docs/comments
      --consumer-root <CONSUMER_ROOT>
          Override roots scanned by `--check-consumer-coupling`. Defaults to current docs plus platform source crates
      --check-module-forks
          Run the same-name module fork scanner over selected authority roots
      --module-fork-root <MODULE_FORK_ROOT>
          Override roots scanned by `--check-module-forks`. Defaults to graph authority roots where fork drift has historically appeared
      --check-gpu-skip-guards
          Run the GPU skip guard over CUDA/WGPU/runtime validation paths
      --gpu-skip-root <GPU_SKIP_ROOT>
          Override roots scanned by `--check-gpu-skip-guards`
  -h, --help
          Print help
  -V, --version
          Print version
```
