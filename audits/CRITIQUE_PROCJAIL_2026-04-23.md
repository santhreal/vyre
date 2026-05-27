# CRITIQUE: procjail  -  Sandboxing / Process-Jail Primitive

**Date:** 2026-04-23  
**Scope:** `libs/runtime/procjail/src/` (read-only audit)  
**Auditor:** Kimi Code CLI  
**Commit basis:** HEAD of working tree  
**Methodology:** Static source analysis against LAWS 0-8, adversarial test review, competitor comparison (bubblewrap, firejail, systemd-nspawn, gVisor runsc).  

---

## Executive Summary

`procjail` has a **solid architectural foundation** but contains **multiple critical security regressions** that degrade or silently disable containment. The most severe are:

1. **Seccomp errors are swallowed** with `let _ = apply_seccomp_filter()`, allowing the sandbox to boot without BPF protection when the kernel or container runtime blocks seccomp.
2. **Seccomp allowlist includes `prctl` and `seccomp` syscalls**, giving attackers a direct path to manipulate the filter or privileges.
3. **Custom provider bypass** leaves rlimits and seccomp entirely unapplied.
4. **No inherited-fd sealing** before `exec` leaks parent file descriptors into the jail.
5. **cgroup v2 code has no v1 fallback** and fails silently on hybrid or v1-only hosts.

**Competitor delta:** bubblewrap closes all fds >2 with `--close-fd`, firejail drops all capabilities and sets `NO_NEW_PRIVS` automatically, systemd-nspawn validates cgroup version before writing. `procjail` does none of these.

---

## Findings

### F1  -  CRITICAL | process/builder.rs:124 | Seccomp errors silently swallowed; sandbox boots unprotected

```rust
let _ = crate::seccomp::apply_seccomp_filter();
```

**Description:** The `pre_exec` closure ignores the return value of `apply_seccomp_filter()`. If the kernel was compiled without `CONFIG_SECCOMP_FILTER`, or if a Docker/seccomp profile blocks `prctl(PR_SET_SECCOMP)`, or if the architecture is unsupported, the error is discarded and the untrusted payload executes **without any syscall filtering**.

**Fix:** Change to `crate::seccomp::apply_seccomp_filter()?;` so `pre_exec` returns `Err(std::io::Error::last_os_error())` on failure. Propagate a descriptive error: `"Fix: verify kernel has CONFIG_SECCOMP=y and container runtime does not block prctl(PR_SET_SECCOMP)."`

**Test hint:** Spawn inside a Docker container with `--security-opt seccomp=unconfined` removed and a custom seccomp profile that blocks `prctl`; assert spawn fails with seccomp error, not success.

---

### F2  -  CRITICAL | seccomp/mod.rs:209 | `SYS_prctl` is in seccomp allowlist

```rust
allow_syscall(&mut rules, libc::SYS_prctl);
```

**Description:** `prctl` is a massive attack surface. Sub-commands include `PR_SET_SECCOMP` (filter disable/installation), `PR_CAP_AMBIENT` (privilege escalation), `PR_SET_NO_NEW_PRIVS` (attempted rollback), and kernel-specific debug leaks. The adversarial tests (`seccomp_adversarial.rs`) only test a few subcommands and rely on `NO_NEW_PRIVS` to block some paths, but `prctl` itself should never be blanket-allowed.

**Fix:** Remove `SYS_prctl` from the allowlist. If specific subcommands are genuinely needed (e.g., `PR_SET_NAME`), implement argument-inspected rules via `seccompiler` conditional filters, or allow only `prctl` with `arg0 == PR_SET_NAME`.

**Test hint:** `verify_seccomp_prctl_not_allowed` (existing audit test) must fail on this codebase until fixed.

---

### F3  -  CRITICAL | seccomp/mod.rs:300 | `SYS_seccomp` is in seccomp allowlist

```rust
allow_syscall(&mut rules, libc::SYS_seccomp);
```

**Description:** Allowing the `seccomp` syscall lets a compromised child install nested filters. While nested filters are ANDed, this exposes kernel attack surface and may allow filter exhaustion (DoS) or future kernel bugs. It is also unnecessary for typical untrusted code.

**Fix:** Remove `SYS_seccomp` from the allowlist. The comment says "to avoid issues if nested"  -  the correct fix is to block it entirely.

**Test hint:** `verify_seccomp_seccomp_not_allowed` (existing audit test) must fail until fixed.

---

### F4  -  CRITICAL | process/builder.rs:65-70 | Custom provider path disables seccomp and rlimits

```rust
unsafe {
    cmd.pre_exec(|| {
        // let _ = crate::seccomp::apply_seccomp_filter();
        Ok(())
    });
}
```

**Description:** When a `custom_provider` is configured, the `pre_exec` closure is a no-op. Seccomp is commented out, and `RLIMIT_AS` / `RLIMIT_CPU` are never set. The custom provider is expected to enforce everything, but there is no validation or fallback. This is a hard bypass of the entire deep-isolation layer.

**Fix:** Remove the custom-provider special case from `pre_exec`. Apply the standard rlimits and seccomp unconditionally **after** the provider mutates `cmd`. If the provider needs to skip them, make that an explicit opt-in in the `SandboxProvider` trait, not a silent default.

**Test hint:** `verify_custom_provider_enforces_limits` (existing audit test) must fail until fixed.

---

### F5  -  HIGH | detect.rs:54-63 | Bubblewrap (stronger FS isolation) ranked below Unshare

```rust
let best_strategy = if has_unshare {
    Strategy::Unshare
} else if has_bubblewrap {
    Strategy::Bubblewrap
} ...
```

**Description:** Auto-detection prefers `Unshare` over `Bubblewrap`. Unshare only provides mount-namespace isolation (changes don't leak) but **does not hide host filesystem contents**. Bubblewrap creates a tmpfs root with explicit read-only bind mounts. Preferring the weaker primitive violates the "auto-select the best available" contract.

**Fix:** Reverse the priority: `has_bubblewrap` → `has_unshare` → `has_firejail` → `RlimitsOnly`.

**Test hint:** `verify_bubblewrap_preferred_over_unshare` (existing audit test) must fail until fixed.

---

### F6  -  HIGH | seccomp/mod.rs:256 | Wrong stat syscall on aarch64

```rust
#[cfg(target_arch = "aarch64")]
allow_syscall(&mut rules, libc::SYS_fstatat64);
```

**Description:** `SYS_fstatat64` is the 32-bit ARM (`arm`) syscall, not `aarch64`. On `aarch64` this constant may be invalid or resolve to a different syscall number, causing the BPF filter to reference a non-existent or wrong syscall. This can break file operations or create a filter hole.

**Fix:** Remove the `aarch64` cfg block for `SYS_fstatat64`. On aarch64, `SYS_fstatat` (or `newfstatat`) is the correct syscall.

**Test hint:** `verify_aarch64_fstatat_config` (existing audit test) must fail until fixed.

---

### F7  -  HIGH | cgroups.rs:39-52 | Memory limit does not cap swap; swap escape possible

```rust
pub fn set_memory_limit(&self, bytes: u64) -> std::io::Result<()> {
    let mem_max = self.path.join("memory.max");
    fs::write(mem_max, bytes.to_string())
}
```

**Description:** Only `memory.max` is written. If the host has swap enabled, the sandboxed process can allocate beyond the memory limit by swapping, then trigger slow OOM or disk exhaustion. Hardware isolation is incomplete.

**Fix:** Also write `memory.swap.max` (e.g., `0` to disable swap, or equal to `bytes`):
```rust
fs::write(self.path.join("memory.swap.max"), "0")?;
```

**Test hint:** `verify_cgroup_swap_max_set` (existing audit test) must fail until fixed.

---

### F8  -  HIGH | cgroups.rs (missing) | No `pids.max` enforcement in cgroup

**Description:** `CgroupV2` has no method to set `pids.max`. A fork bomb inside the cgroup is only throttled by the wall-clock timeout watchdog, not by the kernel. The watchdog sleeps in 25ms increments and may not kill the cascade before the host PID table is exhausted.

**Fix:** Add `set_pids_limit(&self, max: u64)` that writes `pids.max`. Call it during spawn with `config.max_processes`.

**Test hint:** `verify_cgroup_pids_max_set` (existing audit test) must fail until fixed.

---

### F9  -  HIGH | process/builder.rs:103-128 | No inherited fd closure before exec; fd leak into sandbox

**Description:** The `pre_exec` closure sets rlimits and seccomp but does **not** close file descriptors inherited from the parent. Any open sockets, database connections, log files, or other fds held by the parent process are leaked into the untrusted child. This is a classic sandbox escape / information leak vector.

**Fix:** In the `pre_exec` closure, call `libc::close_range(3, u32::MAX, libc::CLOSE_RANGE_CLOEXEC)` (Linux 5.11+). For older kernels, iterate `/proc/self/fd` and close everything except stdin/stdout/stderr. Document: "Fix: close all fds >2 before exec to prevent information leaks."

**Test hint:** Open a known fd in the parent (e.g., `memfd_create`), spawn sandbox with `Strategy::None`, and from the harness list `/proc/self/fd`; assert the leaked fd is absent.

---

### F10  -  HIGH | process/kill.rs:52-69 | pidfd returned without O_CLOEXEC, leaked to child

```rust
let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid_i32, 0) };
```

**Description:** `pidfd_open` is called with flags `0`. The resulting fd does **not** have `O_CLOEXEC`. When `Command::spawn()` forks, this fd is inherited by the child. The child then holds a capability to send signals to itself via `pidfd_send_signal`, or at minimum it can deduce the parent's intent.

**Fix:** Use `libc::syscall(libc::SYS_pidfd_open, pid_i32, libc::PIDFD_NONBLOCK)` if available, and immediately `fcntl(fd, F_SETFD, FD_CLOEXEC)`. At minimum, set CLOEXEC.

**Test hint:** After spawning, read `/proc/{child_pid}/fd` and assert no pidfd is present in the child's fd table.

---

### F11  -  HIGH | process/builder.rs:103-128 | Signal mask and handlers not reset in child

**Description:** The `pre_exec` closure does not call `sigprocmask` or `signal`. If the parent has:
- Blocked `SIGPIPE`, the child will also block it, breaking standard-library I/O assumptions.
- Set `SIGPIPE` to `SIG_IGN`, the child inherits this, causing `write()` to return `EPIPE` instead of delivering `SIGPIPE`.
- Blocked `SIGCHLD`, the child may not receive it from its own descendants.

**Fix:** In `pre_exec`, reset the signal mask with `sigemptyset` + `sigprocmask(SIG_SETMASK, ...)`, and reset `SIGPIPE` to `SIG_DFL`.

**Test hint:** Block `SIGPIPE` in the parent, spawn sandbox, and from the harness assert `sigaction(SIGPIPE)` returns `SIG_DFL` and the signal is unblocked.

---

### F12  -  HIGH | cgroups.rs:19-33 | No cgroup v1 vs v2 detection; v1 systems silently unprotected

```rust
let base_path = Path::new("/sys/fs/cgroup");
if !base_path.exists() {
    return Err(...);
}
let cg_path = base_path.join("procjail").join(name);
fs::create_dir_all(&cg_path)?;
```

**Description:** The code only checks that `/sys/fs/cgroup` exists. It does **not** verify that the mounted filesystem is cgroup2. On:
- **v1-only systems** (older kernels, some enterprise distros): `memory.max` does not exist; writes fail.
- **hybrid v1/v2 systems**: the top-level may be v1 with a v2 subtree; writing `memory.max` at the root fails.
- **systems where cgroupfs is not mounted at all but the directory exists**: false positive.

There is **no v1 fallback** using `memory.limit_in_bytes` / `cpu.cfs_quota_us`.

**Fix:** Probe for `cgroup.controllers` (v2-only file) or read `/proc/filesystems` to confirm `cgroup2`. If v1 is active, fall back to v1 controller paths or at least emit a clear error: `"Fix: cgroups v1 detected; migrate to v2 or use firejail/bwrap for resource containment."`

**Test hint:** Run `CgroupV2::new` on a v1-only host (or in a container with v1 bind-mounted); assert it returns a descriptive error, not `Ok` with subsequent write failures.

---

### F13  -  HIGH | detect.rs:49-52 | Capability probe errors silently discarded

```rust
let has_unshare = check_unshare().unwrap_or(false);
let has_bubblewrap = check_bubblewrap().unwrap_or(false);
let has_firejail = check_firejail().unwrap_or(false);
```

**Description:** If `unshare` fails because `kernel.unprivileged_userns_clone=0` (common on Debian/Ubuntu hardening), the error is dropped. The user sees `best_strategy = RlimitsOnly` with no explanation. No "Fix: ..." hint is emitted.

**Fix:** Store the `Result` (or at least the `Err` string) in `ContainmentLevel` and expose it via `Display` or a logging method. When `has_unshare` is false due to a kernel policy error, log: `"Fix: enable unprivileged user namespaces (sysctl kernel.unprivileged_userns_clone=1) or install bubblewrap."`

**Test hint:** Run `probe_capabilities()` on a host with `kernel.unprivileged_userns_clone=0`; assert the returned struct contains a diagnostic message, not just `has_unshare=false`.

---

### F14  -  MEDIUM | detect.rs:119-126 | Firejail probe does not exercise real sandbox

```rust
fn check_firejail() -> std::io::Result<bool> {
    Command::new("firejail")
        .arg("--version")
        ...
}
```

**Description:** The probe only checks that `firejail --version` exits 0. It does not verify that the SUID bit is set, that the kernel supports the required namespaces, or that the actual sandbox invocation works. A broken firejail installation passes detection but fails at spawn time.

**Fix:** Change probe to a minimal real sandbox: `firejail --noprofile -- echo ok`.

**Test hint:** `verify_firejail_probe_runs_real_sandbox` (existing audit test) must fail until fixed.

---

### F15  -  MEDIUM | process/builder.rs:232-277 | Bwrap mount paths not canonicalized; symlink race

```rust
let wd = work_dir.to_string_lossy();
cmd.args(["--ro-bind", &wd, &wd]);
```

**Description:** `work_dir` is validated as absolute but never canonicalized. Between validation and `bwrap` execution, a malicious symlink can redirect the bind mount to a sensitive host path (e.g., `/etc`). `bwrap` itself resolves symlinks, but the race window exists.

**Fix:** Canonicalize `work_dir` and all `readonly_mounts` / `writable_mounts` host paths with `std::fs::canonicalize()` before passing to bwrap. Document: "Fix: canonicalize all mount paths to prevent symlink races."

**Test hint:** Create a harness directory that is a symlink to `/etc`, attempt spawn; assert the canonicalization detects the escape and aborts.

---

### F16  -  MEDIUM | process/builder.rs:333-365 | `which()` has TOCTOU race

```rust
if candidate.exists() {
    if let Ok(meta) = std::fs::metadata(&candidate) {
        if meta.is_file() && (meta.permissions().mode() & 0o111 != 0) {
            return Ok(candidate);
        }
    }
}
```

**Description:** `candidate.exists()` and `std::fs::metadata(&candidate)` are separate syscalls. A symlink can be swapped between them, causing `which()` to return a path to an attacker-controlled executable.

**Fix:** Use a single `metadata()` call and handle `NotFound` via `match` instead of pre-checking `exists()`.

**Test hint:** `verify_which_checks_executable` (existing) covers non-executable rejection, but a new adversarial test should swap a symlink between `exists()` and `metadata()` calls via a racing thread.

---

### F17  -  MEDIUM | cgroups.rs:65-79 | cgroup.peak and cgroup.kill require recent kernels; no fallback

```rust
pub fn current_memory_peak(&self) -> Option<u64> {
    let peak = self.path.join("memory.peak");
    std::fs::read_to_string(peak).ok()?.trim().parse().ok()
}
```

**Description:** `memory.peak` requires Linux 5.19+. `cgroup.kill` requires 5.14+. On older kernels these files do not exist and the methods silently return `None` / `Err`. There is no fallback to `memory.max_usage_in_bytes` (v1) or `/proc/<pid>/status` VmPeak.

**Fix:** Check kernel version or file existence, and fall back to `/proc/<pid>/status` VmPeak for `current_memory_peak`. For `kill_all`, fall back to `SIGKILL` via pidfd or PID if `cgroup.kill` is absent.

**Test hint:** Run on Linux 5.4 (common in LTS containers); assert `current_memory_peak` returns `Some` via `/proc` fallback, not `None`.

---

### F18  -  MEDIUM | process/kill.rs:132-134 | CPU time parse failure silently returns 0.0

```rust
let utime = parts[11].parse::<f64>().unwrap_or(0.0);
let stime = parts[12].parse::<f64>().unwrap_or(0.0);
```

**Description:** If `/proc/{pid}/stat` has an unexpected format (kernel change, corrupted procfs, or PID reuse race), the CPU time is silently reported as `0.0`. This hides resource exhaustion from monitoring.

**Fix:** Return `Option<f64>` from the parse step and propagate `None` instead of masking with `unwrap_or(0.0)`.

**Test hint:** Inject a mock `/proc/{pid}/stat` with non-numeric fields; assert `cpu_time_secs_for_process` returns `None`, not `Some(0.0)`.

---

### F19  -  MEDIUM | process/builder.rs:105-119 | `RLIMIT_AS` is virtual address space, not RSS

**Description:** `RLIMIT_AS` limits total virtual address space. This includes shared libraries, mmap of read-only files, and thread stacks. A runtime like Node.js or Python with many shared libs may hit the `RLIMIT_AS` ceiling while using far less physical RAM, causing premature OOM kills. The kernel's true memory limit is `memory.max` (cgroup v2), but when cgroups are unavailable, `RLIMIT_AS` is a poor proxy.

**Fix:** Document the behavior explicitly: `"RLIMIT_AS limits virtual address space, not resident memory. For accurate RAM containment, ensure cgroups v2 is available."` Consider also setting `RLIMIT_DATA` (brk heap) as a tighter bound.

**Test hint:** Spawn a process with `max_memory_bytes=256MB` on a system where the runtime's shared libs exceed 256MB virtual; assert it is killed immediately by `RLIMIT_AS`, and document that this is expected behavior.

---

### F20  -  MEDIUM | process/builder.rs:121-122 | `max_fds` and `max_processes` are not kernel-enforced in pre_exec

```rust
// Do not set NOFILE or NPROC here; these can break shells and test harnesses.
```

**Description:** The comment admits that `RLIMIT_NOFILE` and `RLIMIT_NPROC` are skipped in `pre_exec`. They are passed as environment variables (`SANTH_MAX_FDS`, `SANTH_MAX_PROCESSES`) to the harness, which must voluntarily enforce them. A malicious or buggy harness ignores these vars. This is containment by convention, not by kernel.

**Fix:** Set `RLIMIT_NOFILE` and `RLIMIT_NPROC` unconditionally in `pre_exec`. If legitimate harnesses break, fix the harnesses or raise the defaults. Security invariants must not depend on untrusted code cooperating.

**Test hint:** `verify_rlimit_nofile_nproc_set` (existing audit test) must fail until fixed.

---

### F21  -  MEDIUM | seccomp/mod.rs:97-100 | Non-Linux stub returns Ok() with no diagnostic

```rust
#[cfg(not(target_os = "linux"))]
pub fn apply_seccomp_filter() -> Result<()> {
    Ok(())
}
```

**Description:** On macOS, BSD, or Windows builds, `apply_seccomp_filter()` silently succeeds. Callers believe the sandbox is active when no BPF filter was ever installed. This is a false sense of security.

**Fix:** Return `Err(SeccompError::UnsupportedArch)` or at minimum log a warning: `"Fix: seccomp is only available on Linux. Use bubblewrap, firejail, or run inside a VM for syscall filtering on this platform."`

**Test hint:** Compile for `cfg(not(target_os = "linux"))` and assert `apply_seccomp_filter()` returns an error.

---

### F22  -  LOW | cgroups.rs:82-86 | Cgroup directory leaked on Drop if processes survive

```rust
impl Drop for CgroupV2 {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.path);
    }
}
```

**Description:** If any process is still inside the cgroup (zombie, ignored SIGKILL, or descendant the parent didn't reap), `remove_dir` returns `EBUSY` and the cgroup slice is leaked. Over time this fills `/sys/fs/cgroup/procjail/` with dead slices.

**Fix:** In `Drop`, first write `"1"` to `cgroup.kill` (if available), then retry `remove_dir` with a short bounded loop. If still failing, log the leak: `"Fix: cgroup {path} still has living processes; manual cleanup required."`

**Test hint:** Spawn a child that ignores SIGKILL (via `PR_SET_PDEATHSIG` trick or ptraced state), drop the `CgroupV2`, and assert the directory is either removed or logged.

---

### F23  -  LOW | process/mod.rs:471-478 | Drop calls kill on already-exited process

```rust
impl Drop for SandboxedProcess {
    fn drop(&mut self) {
        self.watchdog_cancel.store(true, Ordering::Release);
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => self.kill(),
        }
    }
}
```

**Description:** If `try_wait()` returns `Err(_)`, `kill()` is invoked. This can send `SIGKILL` to a PID that was already reaped by `waitpid` in another thread, potentially hitting a reused PID. The race window is small but non-zero.

**Fix:** Guard `kill()` with a second `try_wait()` inside the `Err` branch, or use `std::sync::Mutex` around `wait()` / `kill()` operations.

**Test hint:** Stress-test rapid spawn/drop with a tight loop and check `dmesg` for "process X does not exist" or audit for PID-reuse collateral damage.

---

### F24  -  LOW | process/builder.rs:288-290 | Firejail double-sets rlimits that pre_exec also sets

```rust
cmd.arg(format!("--rlimit-as={}", config.max_memory_bytes));
cmd.arg(format!("--rlimit-nofile={}", config.max_fds));
cmd.arg(format!("--rlimit-nproc={}", config.max_processes));
```

**Description:** Firejail sets rlimits via its own wrapper, and then `pre_exec` sets `RLIMIT_AS` and `RLIMIT_CPU` again. While Linux rlimits are idempotent for lower values, conflicting values (e.g., firejail `--rlimit-as` vs pre_exec `RLIMIT_AS`) can cause unpredictable behavior depending on which runs last.

**Fix:** For strategies that set their own rlimits (firejail), skip the corresponding `setrlimit` calls in `pre_exec`. Document the delegation.

**Test hint:** Spawn with `Strategy::Firejail` and tight memory limits; assert the effective limit matches `config.max_memory_bytes` via `/proc/{pid}/limits`.

---

### F25  -  INFO | INTERNAL_SPEC.md:29 | Spec falsely claims `#![forbid(unsafe_code)]`

**Description:** The internal spec states: `"#![forbid(unsafe_code)]: yes (although comments note unsafe for libc usage in process.rs, the forbid directive is active at the crate level)"`. In reality, `lib.rs` has `#![cfg_attr(not(test), deny(clippy::unwrap_used, ...))]`  -  not `forbid(unsafe_code)`. The crate contains extensive `unsafe` blocks for `libc` syscalls in `process/builder.rs`, `process/kill.rs`, and `seccomp/mod.rs`.

**Fix:** Update `INTERNAL_SPEC.md` to reflect the actual unsafe policy: `"Unsafe code is confined to libc syscall wrappers in process/ and seccomp/ modules. All unsafe blocks are annotated with SAFETY comments."`

**Test hint:** `grep -n 'unsafe' procjail/src/**/*.rs | wc -l` should be documented in the spec.

---

### F26  -  INFO | process/builder.rs:419-420 | Watchdog cgroup kill has unlogged fallback race

```rust
if let Some(cg_path) = cgroup_path {
    let _ = std::fs::write(cg_path.join("cgroup.kill"), "1");
}
```

**Description:** The watchdog attempts `cgroup.kill` but ignores the result. If the cgroup was already cleaned up by `Drop`, this silently fails. The fallback `kill_process(pid)` then runs, but by now the original PID may have exited and been reused.

**Fix:** Check the `cgroup.kill` write result. If it fails with `ENOENT`, the cgroup is already gone and the process likely exited naturally; skip the `kill_process` fallback unless `timed_out` is still true and the child is confirmed alive via `pidfd` or `kill(pid, 0)`.

**Test hint:** Spawn a short-lived process with a long timeout; verify the watchdog does not emit `[santh-sandbox] kill(...) failed` after natural exit.

---

## Hunt Area Checklist

| Hunt Area | Findings | Status |
|---|---|---|
| 1. Syscall fallback Fix: hints | F1, F13, F21 | **CRITICAL gaps**  -  seccomp and namespace failures are swallowed or return Ok() with no diagnostic. |
| 2. unwrap() on /proc paths |  -  | **No literal `.unwrap()` found**, but F18 shows `unwrap_or(0.0)` masking parse errors on `/proc/{pid}/stat`. |
| 3. Path canonicalization before syscalls | F15, F16 | **TOCTOU races** in `which()` and bwrap mounts. No `canonicalize()` anywhere. |
| 4. cgroup v1 vs v2 handling | F12, F7, F8, F17 | **No v1 detection or fallback**. v2-only files used without kernel-version guards. |
| 5. fd handling (CLOEXEC, inherited fds) | F9, F10 | **No fd sealing before exec**. pidfd lacks CLOEXEC. |
| 6. Signal handling in child | F11 | **Signal mask and handlers not reset**. SIGPIPE, SIGCHLD propagation undefined. |
| 7. Resource limits (rlimits) | F19, F20, F24 | `RLIMIT_AS` misdocumented as memory limit; NOFILE/NPROC left to harness trust. |

---

## Competitor Comparison

| Primitive | bubblewrap | firejail | procjail (current) |
|---|---|---|---|
| Close all fds >2 | `--close-fd` | automatic | ❌ missing |
| Set `NO_NEW_PRIVS` | automatic | automatic | ✅ in seccomp init |
| Seccomp `prctl` block | custom filter | custom filter | ❌ **allowed** |
| cgroup v2 swap limit | N/A (delegates to caller) | N/A | ❌ **missing** |
| v1 fallback | N/A | N/A | ❌ **missing** |
| Path canonicalization | resolves all binds | `--whitelist` resolves | ❌ **none** |
| Signal reset before exec | N/A | resets handlers | ❌ **missing** |

---

## Remediation Priority

1. **Immediate (P0):** F1, F2, F3, F4  -  seccomp is either not applied, bypassed, or the filter itself is too permissive. These are sandbox escapes.
2. **This sprint (P1):** F5, F6, F7, F8, F9, F10, F11, F12  -  detection, cgroup completeness, fd sealing, and signal hygiene.
3. **Next sprint (P2):** F13, F14, F15, F16, F17, F18, F19, F20, F21  -  diagnostics, races, and fallback paths.
4. **Cleanup (P3):** F22, F23, F24, F25, F26  -  leaks, docs, and edge-case races.

---

*End of audit.*
