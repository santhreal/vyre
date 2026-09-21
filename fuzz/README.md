# vyre-fuzz

The libFuzzer targets. Each target takes arbitrary bytes, drives them through a
parser or a dispatch path, and asserts that the failure is an error with a
`Fix:` hint rather than a panic.

This crate is not a workspace member. It carries its own lock file and builds
under `cargo fuzz`, which needs a nightly toolchain.

## Targets

- `fuzz_targets/dispatch.rs`: arbitrary bytes to `Program::from_wire`, then
  validation, optimization and a wgpu dispatch with zeroed inputs. A wire blob
  over 64 MiB is rejected before parsing, and dispatch runs only when the
  declared buffers fit under 8 MiB, so a huge declaration fails the size check
  instead of the allocator.

## Run it

```sh
./cargo_full fuzz build dispatch
./cargo_full fuzz run dispatch -- -max_total_time=60
```

Dispatch needs a visible GPU adapter. Without one the target still exercises
parsing, validation and optimization, and reports no dispatch.
