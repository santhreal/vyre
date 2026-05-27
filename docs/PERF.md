# Build Performance & Optimization Guide (PGO, SCCache, LTO)

This document describes the performance optimizations configured for the Vyre execution substrate and details the Profile Guided Optimization (PGO) workflow.

---

## 1. Profile Guided Optimization (PGO)

PGO is a compiler optimization technique that uses profiles collected from actual program execution to guide the compiler in making better optimization decisions (e.g., inlining hot functions, reordering branches).

To optimize a release build using PGO, follow this procedure:

### Step 1: Install `cargo-pgo`
Ensure you have the `cargo-pgo` helper subcommand installed:
```bash
cargo install cargo-pgo --locked
```

### Step 2: Build with Instrumentation
Compile the workspace with profiling instrumentation enabled:
```bash
cargo pgo build
```
This generates binaries instrumented to collect execution profiles.

### Step 3: Collect Profile Data (Run Workloads)
Run your benchmark suite, unit tests, or representative workloads using the instrumented binaries to collect execution profiles:
```bash
cargo pgo test
```
*(Optionally run your benchmarks: `cargo pgo bench`)*

### Step 4: Build Optimized Binary
Compile the final, highly optimized release binary using the collected profile data:
```bash
cargo pgo optimize
```
This generates optimized binaries built specifically for your hot paths.

---

## 2. SCCache (Shared Compilation Cache)

We have enabled `sccache` by default in all `.cargo/config.toml` files using:
```toml
[build]
rustc-wrapper = "sccache"
```

To take advantage of sccache locally, make sure it is installed and in your `PATH`:
- **Linux**: `cargo install sccache --locked`
- **macOS**: `brew install sccache`
- **Windows**: `choco install sccache`

---

## 3. Thin LTO (Link-Time Optimization)

Workspace release builds are configured to use **Thin LTO** and single codegen units to maximize runtime execution performance without the massive compile-time penalty of Fat LTO.

This is set in the workspace `Cargo.toml` as:
```toml
[profile.release]
lto = "thin"
codegen-units = 1
```
