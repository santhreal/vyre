# vyre-driver-wgpu  -  Configurability

The `vyre-wgpu` binary and the wgpu backend library expose a small,
explicit Tier A surface.

## Tier A  -  operational config

CLI flags + environment variables. Compiled defaults < env < CLI.

| Flag / env                       | Default | Purpose                                                                  |
|----------------------------------|---------|--------------------------------------------------------------------------|
| `--bench-only`                   | off     | Run the latency / throughput micro-benches only; skip parity checks.    |
| `--adapter <name>`               | auto    | Force a specific wgpu adapter (substring match against `Adapter::info`).|
| `--features <list>`              | runtime | Override the `wgpu::Features` mask (subgroup ops, timestamp queries…). |
| env `VYRE_PIPELINE_CACHE_ENTRIES`| `4096`  | In-memory pipeline-cache entry budget (audit item #18).                 |
| env `VYRE_PIPELINE_CACHE_BYTES`  | `512 MiB` | In-memory pipeline-cache byte budget (audit item #17).                  |
| env `VYRE_DISK_CACHE_DIR`        | `~/.cache/vyre/wgpu` | Override the on-disk pipeline-cache root.                                |
| env `VYRE_DISK_CACHE_MAX_BYTES`  | `4 GiB` | Hard ceiling for the disk pipeline cache.                              |
| env `VYRE_DISK_CACHE_TTL_DAYS`   | `30`    | Disk-cache eviction age. `0` disables time-based eviction.              |
| env `VYRE_TRACE_DISPATCH`        | `0`     | `1` = print one line per dispatch (timing, buffer count, hit/miss).     |

Every env var has a documented default in `vyre-driver-wgpu/src/runtime/`
and an integration test that round-trips parsing through the public
`WgpuBackendStats` surface.

## Tier B  -  community knowledge

The wgpu backend reads no rule corpus. Which operations it runs is decided in
Rust: `vyre-emit-naga/src/emitter/op_lookup.rs` lowers each IR node to Naga, and
a node it has no lowering for fails emission with an `EmitError` naming the
node. New backend coverage is a change there, not a data file.

A workspace op corpus and its schema were described here and have never existed
in this repository. The rule corpora that do exist are per crate:
`vyre-libs/rules`, `vyre-lower/rules` and `vyre-lints/rules`.
