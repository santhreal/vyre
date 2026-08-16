//! What may not appear on a dispatch path.
//!
//! Seven shell ratchets covered this ground, each with its own copy of a source
//! walk, its own ceiling constant, and its own `--strict` mode that no workflow
//! ever passed. The ceilings are pinned in `xtask/gate-baselines.toml` now, so
//! lowering one is a diff against the pin rather than an edit to the rule, and
//! the reviewed allowances are checked on every run rather than only under a
//! flag nothing set.
//!
//! Two of the ratchets, blocking waits and unbounded caches, deliberately failed
//! below their ceiling as well as above it, on the argument that slack is where a
//! regression hides. The runner now reports a count below the pin so the pin can
//! be lowered, which is the same guarantee without a red tree for progress.

use std::path::Path;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Rule, Tree};

/// Blocking waits, busy waits and polled maintenance on throughput paths.
pub struct BlockingWait;

impl Gate for BlockingWait {
    fn name(&self) -> &'static str {
        "hot-path-blocking-wait"
    }

    fn help(&self) -> &'static str {
        "block-on, sleep, park and Maintain::Wait in driver production sources"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        // The spellings a blocking wait actually has in this tree. `PollType::Wait`
        // and `PollType::wait_for` are wgpu's current names for the device wait
        // that used to be `Maintain::Wait`; searching only the retired name left
        // every live device wait uncovered.
        const NEEDLES: &[&str] = &[
            "PollType::Wait",
            "PollType::wait_for",
            "pollster::block_on",
            "std::thread::sleep(",
            "std::thread::yield_now()",
            "thread::park()",
            "park_timeout",
        ];
        let tree = Tree::open(&ctx.root)?;
        scan::ratchet(
            &tree,
            &Rule {
                roots: &["vyre-driver-wgpu/src"],
                // Two files own waiting and are out of scope rather than reviewed:
                // wait_backoff.rs is the sanctioned home for adaptive backoff, and
                // acquire.rs holds the single blocking wait a synchronous driver API
                // owes an asynchronous device request, plus the one device poll every
                // readback path calls. Everything else is a dispatch path, where a
                // wait is a defect.
                skip: &|path| {
                    is_test_path(path)
                        || path.to_string_lossy().contains("/benches/")
                        || path == Path::new("vyre-driver-wgpu/src/wait_backoff.rs")
                        || path == Path::new("vyre-driver-wgpu/src/runtime/device/acquire.rs")
                },
                line: &|line| scan::contains_any(line, NEEDLES),
                reviewed: &[],
                reviewed_line: None,
                statement: None,
                message: "blocking wait on a throughput path",
                fix: "prefer a nonblocking PollType::Poll, a fence callback, or the backoff \
                      helper, then lower the pin",
                unreviewed_message: "blocking wait outside the two files that own waiting",
                unreviewed_fix: "call the device-acquisition wait or the backoff helper instead \
                                 of blocking here, or move the work off the dispatch path",
            },
        )
    }
}

/// Unbounded associative containers in dispatcher and runtime wiring.
pub struct UnboundedCache;

impl Gate for UnboundedCache {
    fn name(&self) -> &'static str {
        "hot-path-unbounded-cache"
    }

    fn help(&self) -> &'static str {
        "bare HashMap::new and VecDeque::new in dispatcher and runtime wiring"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        scan::ratchet(
            &tree,
            &Rule {
                roots: &["vyre-driver-wgpu/src", "vyre-runtime/src"],
                skip: &scan::is_test_tree,
                line: &|line| {
                    scan::contains_word(line, "HashMap::new()")
                        || scan::contains_word(line, "VecDeque::new()")
                },
                reviewed: &[],
                reviewed_line: None,
                statement: None,
                message: "unbounded associative container construction",
                fix: "construct with a bound (capacity, eviction, pool) or move the site \
                      off the hot tier, then lower the pin",
                unreviewed_message: "unbounded container construction on a hot tier",
                unreviewed_fix: "give the container an explicit capacity budget, tier eviction \
                                 or a pool, and document the bound in the module",
            },
        )
    }
}

/// Unbounded synchronous reads of external files on dispatch-critical paths.
pub struct UnboundedRead;

impl Gate for UnboundedRead {
    fn name(&self) -> &'static str {
        "hot-path-unbounded-read"
    }

    fn help(&self) -> &'static str {
        "read_to_end over an arbitrary file on a dispatch-critical path"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        scan::ratchet(
            &tree,
            &Rule {
                roots: &["vyre-driver-wgpu/src"],
                skip: &is_test_path,
                line: &|line| line.contains("read_to_end"),
                reviewed: &[],
                reviewed_line: None,
                // A read-all is bounded by the `take` in its own chain, wherever
                // that chain spells it, so the statement is what decides. Judging
                // the line alone forced a whole-module exemption, under which an
                // unbounded read added beside a bounded one reported nothing.
                statement: Some(&|statement| !statement.contains(".take(")),
                message: "unbounded synchronous read-all",
                fix: "read behind a bound (explicit max bytes, chunked read, capped mmap), \
                      then lower the pin",
                unreviewed_message: "read-all with no byte bound in its own chain",
                unreviewed_fix: "cap the read with `take` in the same expression, or read a \
                                 bounded prefix, so the bound is visible where the read is",
            },
        )
    }
}

/// Reserve calls that pass remaining capacity instead of additional length.
///
/// `try_reserve` takes additional capacity relative to `len()`. Deriving the
/// argument from `capacity()` reserves less than the caller asked for, which
/// reallocates inside the loop the reserve was meant to hoist out of.
pub struct ReserveArgument;

impl Gate for ReserveArgument {
    fn name(&self) -> &'static str {
        "hot-path-reserve"
    }

    fn help(&self) -> &'static str {
        "reserve calls deriving additional capacity from capacity()"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        let files = tree.all_rust();
        report.note(format!("scanned {} Rust source file(s)", files.len()));
        for file in &files {
            let text = tree.read(file)?;
            // A quoted call is a fixture or an example, not a reserve this rule
            // can be wrong about, so the scan reads code with literals masked and
            // quotes the original line back when it reports.
            let masked = scan::mask_literals(&text);
            let lines: Vec<&str> = masked.lines().collect();
            let quoted: Vec<&str> = text.lines().collect();
            let mut index = 0;
            while index < lines.len() {
                let line = lines[index];
                // A doc comment quotes the wrong form on purpose, as the thing
                // being fixed, and that lesson stays readable.
                if !line.contains(".try_reserve") || line.trim_start().starts_with("//") {
                    index += 1;
                    continue;
                }
                let (uses_capacity, end) = block_uses_capacity(&lines, index);
                if uses_capacity {
                    report.find(Finding::at(
                        file.clone(),
                        u32::try_from(index + 1).unwrap_or(u32::MAX),
                        format!(
                            "reserve derives additional capacity from capacity(): {}",
                            quoted.get(index).unwrap_or(&line).trim()
                        ),
                        "pass target_capacity - collection.len() after a capacity guard",
                    ));
                }
                index = (end + 1).max(index + 1);
            }
        }
        Ok(report)
    }
}

/// Whether the call block starting at `start` references `capacity()`.
///
/// The call can span lines, so the block runs to the statement's semicolon, with
/// an eight-line bound so an unterminated statement cannot swallow the file.
fn block_uses_capacity(lines: &[&str], start: usize) -> (bool, usize) {
    let call_at = lines[start].find(".try_reserve").unwrap_or(0);
    let mut block = String::from(&lines[start][call_at..]);
    let mut end = start;
    while end + 1 < lines.len() && !block.contains(';') {
        end += 1;
        block.push('\n');
        block.push_str(lines[end]);
        if end - start > 8 {
            break;
        }
    }
    (block.contains(".capacity()"), end)
}

/// Test trees are not the dispatch path.
fn is_test_path(path: &Path) -> bool {
    let path = path.to_string_lossy();
    path.contains("/test/") || path.contains("/tests/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateCtx;
    use crate::gates::fixture_checkout;

    /// WHY: the under-reserve defect is only visible across lines, because the
    /// argument expression is usually wrapped. A single-line check found none of
    /// the real sites.
    #[test]
    fn a_reserve_argument_is_read_across_the_whole_statement() {
        let lines = vec![
            "buffer.try_reserve(",
            "    target",
            "        .saturating_sub(buffer.capacity()),",
            ")?;",
        ];
        let (uses, end) = block_uses_capacity(&lines, 0);
        assert!(uses);
        assert_eq!(end, 3);

        let good = vec!["buffer.try_reserve(target - buffer.len())?;"];
        assert!(!block_uses_capacity(&good, 0).0);
    }

    /// WHY: an unterminated statement must not swallow the rest of the file and
    /// attribute a `capacity()` call fifty lines away to this reserve.
    #[test]
    fn a_reserve_block_is_bounded() {
        let mut lines = vec!["buffer.try_reserve("];
        for _ in 0..40 {
            lines.push("    more");
        }
        lines.push("    x.capacity()");
        let (uses, end) = block_uses_capacity(&lines, 0);
        assert!(!uses);
        assert_eq!(end, 9);
    }

    /// WHY: the rule reads every Rust file in the tree, so the fixtures above are
    /// themselves inside its scope, and the gate reported its own test data. A
    /// rule that reports the text of its own examples spends a pin on the size of
    /// its test module, and the only way to lower that pin is to stop writing
    /// examples. Masking literals keeps the example readable and the rule honest,
    /// and this proves both directions on one tree.
    #[test]
    fn a_quoted_reserve_is_not_a_call_and_a_real_one_still_is() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "quoted.rs",
                "fn fixture() {\n    let lines = vec![\"buffer.try_reserve(\", \"    x.capacity(),\", \")?;\"];\n}\n",
            ),
            (
                "real.rs",
                "fn stage(buffer: &mut Vec<u8>, target: usize) {\n    buffer.try_reserve(\n        target.saturating_sub(buffer.capacity()),\n    ).expect(\"Fix: reserve the staging buffer.\");\n}\n",
            ),
        ]);

        let report = ReserveArgument
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads the fixture tree");
        let named = report.named_files();
        assert_eq!(
            named,
            ["real.rs"],
            "a quoted call is data and a real one is a defect: {named:?}"
        );
    }
}
