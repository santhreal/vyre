#!/usr/bin/env python3
"""
divergence-gate: vyre-vs-clang AST divergence sweep + PR quality gate.

USAGE
-----
  Sweep mode (cold corpus run):
    divergence-gate.py sweep \\
        --corpus /path/to/corpus \\
        --vyrec  /path/to/vyrec \\
        --clang  clang \\
        --out    /tmp/divergences.json \\
        [--limit N]

  Gate mode (PR review hook):
    divergence-gate.py gate \\
        --corpus /path/to/corpus \\
        --vyrec  /path/to/vyrec \\
        --clang  clang \\
        --baseline-divergences /tmp/divergences-main.json \\
        --pr-tree /path/to/pr/worktree \\
        --pr-ref  HEAD \\
        --baseline-ref origin/main \\
        --new-tests-glob 'vyre-frontend-c/tests/divergence/*.rs' \\
        --baseline-bench-report /tmp/bench-main.json \\
        --pr-bench-report /tmp/bench-pr.json \\
        --max-bench-regression 0.05

DESIGN
------
v1 ships *coarse* divergence detection: per-file, compare structural
shape only  -  node count, max depth, kind histogram. The kind-mapping
(strict per-node correspondence between vyre and clang AST classes) is
deliberately deferred to v2 once we see what the divergences actually
look like in the real corpus.

The gate enforces:
  1. Divergence count strictly decreases vs. baseline.
  2. No previously-passing file regresses.
  3. Diff lints: no hardcoded token literals / file paths / identifier
     strings appearing in BOTH a new fixture and the kernel patch.
  4. Each new test FAILS on the baseline tree (proof it actually tests).
  5. PR benchmark report stays within the configured regression tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# ---------- vyre AST extraction ----------

def run_vyre(vyrec: str, source: Path, dump_dir: Path) -> tuple[dict | None, str]:
    """Run vyrec on `source` with VYRE_DUMP_TYPED_VAST set.

    Returns `(parsed_json_or_None, last_stderr_line)`. The stderr line is
    captured so the sweep summary can categorise *why* vyrec failed
    (parser stage vs. link vs. timeout).
    """
    env = os.environ.copy()
    env["VYRE_DUMP_TYPED_VAST"] = str(dump_dir)
    out = dump_dir / "vyrec.out"
    try:
        proc = subprocess.run(
            [vyrec, str(source), "-o", str(out)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return (None, "timeout")
    last_err = ""
    for line in stderr_text.splitlines():
        line = line.strip()
        if line and ("fatal error" in line or "dispatch failed" in line
                     or "panicked" in line or line.startswith("error")):
            last_err = line
    json_path = dump_dir / f"{source.name}.vast.json"
    if not json_path.exists():
        return (None, last_err or "no-dump-no-error")
    try:
        return (json.loads(json_path.read_text()), "")
    except json.JSONDecodeError:
        return (None, "json-decode-failed")


# ---------- clang AST extraction ----------

def run_clang(clang: str, source: Path) -> dict | None:
    """Get clang's JSON AST for `source`. Returns parsed JSON or None on failure."""
    try:
        proc = subprocess.run(
            [clang, "-Xclang", "-ast-dump=json", "-fsyntax-only",
             "-x", "c", str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        # clang itself rejected the file  -  count as "no clang baseline";
        # we don't compare in that case.
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


# ---------- shape extraction ----------

@dataclass
class Shape:
    """Coarse structural fingerprint of an AST."""
    node_count: int
    max_depth: int
    kind_histogram: dict[str, int]

    def differs_from(self, other: "Shape") -> tuple[bool, list[str]]:
        diffs: list[str] = []
        if self.node_count != other.node_count:
            diffs.append(f"node_count: vyre={self.node_count} clang={other.node_count}")
        if self.max_depth != other.max_depth:
            diffs.append(f"max_depth: vyre={self.max_depth} clang={other.max_depth}")
        # kind histogram comparison is informational in v1  -  we don't yet
        # have a kind-mapping table, so we just report the bucket totals.
        if sum(self.kind_histogram.values()) != sum(other.kind_histogram.values()):
            diffs.append("kind_total: differs")
        return (len(diffs) > 0, diffs)


def vyre_shape(dump: dict) -> Shape:
    """Coarse shape of a vyre VAST dump."""
    nodes = dump.get("nodes") or []
    # Each node is [kind, parent, fc, ns, ...]. Build child-count via
    # parent indices to compute depth.
    n = len(nodes)
    parents = [row[1] if len(row) > 1 else 0xFFFFFFFF for row in nodes]
    SENT = 0xFFFFFFFF
    # Depth via memoized walk-up.
    depths = [-1] * n
    def depth_of(i: int) -> int:
        if depths[i] >= 0:
            return depths[i]
        if i >= n:
            return 0
        p = parents[i]
        if p == SENT or p >= n or p == i:
            depths[i] = 0
        else:
            depths[i] = depth_of(p) + 1
        return depths[i]
    max_depth = max((depth_of(i) for i in range(n)), default=0)
    histo: dict[str, int] = {}
    for row in nodes:
        kind = str(row[0]) if row else "0"
        histo[kind] = histo.get(kind, 0) + 1
    return Shape(node_count=n, max_depth=max_depth, kind_histogram=histo)


def clang_shape(dump: dict) -> Shape:
    """Coarse shape of a clang AST. Walks `inner` recursively."""
    histo: dict[str, int] = {}
    max_d = [0]
    count = [0]
    def walk(node: dict, depth: int) -> None:
        if not isinstance(node, dict):
            return
        count[0] += 1
        max_d[0] = max(max_d[0], depth)
        kind = str(node.get("kind", "?"))
        histo[kind] = histo.get(kind, 0) + 1
        for child in node.get("inner", []) or []:
            walk(child, depth + 1)
    walk(dump, 0)
    return Shape(node_count=count[0], max_depth=max_d[0], kind_histogram=histo)


# ---------- sweep ----------

@dataclass
class FileResult:
    path: str
    status: str          # "match" | "divergent" | "vyre_failed" | "clang_failed"
    diffs: list[str]
    vyre_shape: dict | None
    clang_shape: dict | None
    error: str = ""      # last fatal/dispatch line from vyrec stderr (if any)


def iter_corpus(corpus: Path, limit: int | None) -> Iterable[Path]:
    n = 0
    for p in sorted(corpus.rglob("*.c")):
        yield p
        n += 1
        if limit and n >= limit:
            return


def sweep(corpus: Path, vyrec: str, clang: str, limit: int | None) -> list[FileResult]:
    results: list[FileResult] = []
    with tempfile.TemporaryDirectory() as td:
        dump_root = Path(td)
        for source in iter_corpus(corpus, limit):
            file_dump_dir = dump_root / source.stem
            file_dump_dir.mkdir(parents=True, exist_ok=True)
            v_json, v_err = run_vyre(vyrec, source, file_dump_dir)
            c_json = run_clang(clang, source)
            if v_json is None:
                results.append(FileResult(str(source), "vyre_failed", [], None, None, v_err))
                continue
            if c_json is None:
                results.append(FileResult(str(source), "clang_failed", [], None, None, ""))
                continue
            vshape = vyre_shape(v_json)
            cshape = clang_shape(c_json)
            differs, diffs = vshape.differs_from(cshape)
            status = "divergent" if differs else "match"
            results.append(FileResult(
                str(source),
                status,
                diffs,
                asdict(vshape),
                asdict(cshape),
            ))
    return results


# ---------- gate lints ----------

def lint_diff_for_hardcodes(pr_tree: Path, baseline_ref: str, pr_ref: str) -> list[str]:
    """
    Reject patches where a kernel file added a string literal that is
    *also* present in a newly-added fixture. That's the "fixture-shaped
    fix" smell: special-casing a token rather than extending the real
    classification table.
    """
    issues: list[str] = []
    # diff name-status to find new fixtures
    proc = subprocess.run(
        ["git", "diff", "--name-status", f"{baseline_ref}...{pr_ref}"],
        cwd=pr_tree, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    new_fixtures = []
    kernel_files = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        if status == "A" and (path.endswith(".c") or path.endswith(".h")) \
                and "tests/" in path:
            new_fixtures.append(path)
        if "vyre-libs/src/parsing/c/" in path or "vyre-frontend-c/src/" in path:
            kernel_files.append(path)
    # Read kernel-file diff hunks; for each new fixture, scan for shared
    # string literals.
    for fixture in new_fixtures:
        fpath = pr_tree / fixture
        if not fpath.exists():
            continue
        ftext = fpath.read_text(errors="replace")
        # Pull identifier tokens from the fixture.
        idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", ftext))
        # Filter out C keywords and very common names.
        idents -= {
            "int", "char", "void", "if", "else", "for", "while", "return",
            "do", "switch", "case", "break", "continue", "static", "const",
            "extern", "inline", "typedef", "struct", "union", "enum",
            "main", "include", "define", "ifdef", "ifndef", "endif",
            "unsigned", "signed", "short", "long", "float", "double",
        }
        for kfile in kernel_files:
            kpath = pr_tree / kfile
            if not kpath.exists():
                continue
            # Get the *patch* hunks for this kernel file vs baseline.
            patch = subprocess.run(
                ["git", "diff", f"{baseline_ref}...{pr_ref}", "--", kfile],
                cwd=pr_tree, stdout=subprocess.PIPE, text=True,
            ).stdout
            # Collect added lines only.
            added = [ln[1:] for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
            added_text = "\n".join(added)
            for ident in idents:
                if re.search(r"\b" + re.escape(ident) + r"\b", added_text):
                    issues.append(
                        f"hardcode-suspicion: identifier '{ident}' appears in "
                        f"both new fixture {fixture} and kernel diff {kfile}"
                    )
    return issues


def find_new_tests(pr_tree: Path, baseline_ref: str, pr_ref: str,
                   glob_pattern: str) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-status", "--diff-filter=A",
         f"{baseline_ref}...{pr_ref}"],
        cwd=pr_tree, stdout=subprocess.PIPE, text=True,
    )
    pat = re.compile(glob_pattern.replace("*", ".*"))
    return [line.split("\t", 1)[1] for line in proc.stdout.splitlines()
            if line.startswith("A") and pat.search(line.split("\t", 1)[1])]


def assert_test_fails_on_baseline(pr_tree: Path, baseline_ref: str,
                                  test_path: str) -> str | None:
    """Checkout the test file from PR onto baseline tree, run it, expect FAIL."""
    with tempfile.TemporaryDirectory() as td:
        # Materialize PR's version of the test.
        pr_test = subprocess.run(
            ["git", "show", f"HEAD:{test_path}"],
            cwd=pr_tree, stdout=subprocess.PIPE,
        ).stdout
        if not pr_test:
            return f"missing-test-file: {test_path}"
        # Worktree at baseline.
        baseline_wt = Path(td) / "baseline"
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline_wt),
                        baseline_ref], cwd=pr_tree, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        try:
            target = baseline_wt / test_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pr_test)
            # Run it  -  assume it's a Rust test. Filter by file stem.
            stem = Path(test_path).stem
            proc = subprocess.run(
                ["./cargo_full", "test", "--test", stem],
                cwd=baseline_wt, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            if proc.returncode == 0:
                return (f"test-passes-on-baseline: {test_path}  -  the test "
                        f"doesn't actually exercise the fix")
            return None
        finally:
            subprocess.run(["git", "worktree", "remove", "--force",
                            str(baseline_wt)], cwd=pr_tree,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)


def collect_bench_values(value: object, out: list[float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "wall_ns",
                "baseline_wall_ns",
                "median_wall_ns",
                "mean_wall_ns",
                "p50_wall_ns",
                "elapsed_ns",
            }:
                collect_bench_values(child, out)
            elif isinstance(child, (dict, list)):
                collect_bench_values(child, out)
        return
    if isinstance(value, list):
        for child in value:
            collect_bench_values(child, out)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            out.append(numeric)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compare_bench_reports(
    baseline_report: str | None,
    pr_report: str | None,
    max_regression: float,
) -> list[str]:
    if not baseline_report or not pr_report:
        return [
            "missing-bench-report: gate mode requires --baseline-bench-report "
            "and --pr-bench-report so parser progress cannot hide a perf regression"
        ]
    if max_regression < 0.0:
        return ["invalid-bench-tolerance: --max-bench-regression must be non-negative"]
    try:
        baseline_json = json.loads(Path(baseline_report).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid-baseline-bench-report: {baseline_report}: {exc}"]
    try:
        pr_json = json.loads(Path(pr_report).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid-pr-bench-report: {pr_report}: {exc}"]
    baseline_values: list[float] = []
    pr_values: list[float] = []
    collect_bench_values(baseline_json, baseline_values)
    collect_bench_values(pr_json, pr_values)
    issues: list[str] = []
    if len(baseline_values) < 30:
        issues.append(
            f"insufficient-baseline-bench-samples: found {len(baseline_values)}, need >= 30"
        )
    if len(pr_values) < 30:
        issues.append(f"insufficient-pr-bench-samples: found {len(pr_values)}, need >= 30")
    if issues:
        return issues
    baseline_median = median(baseline_values)
    pr_median = median(pr_values)
    allowed = baseline_median * (1.0 + max_regression)
    if pr_median > allowed:
        issues.append(
            "bench-regression: median wall time "
            f"{pr_median:.0f}ns exceeds allowed {allowed:.0f}ns "
            f"(baseline {baseline_median:.0f}ns, tolerance {max_regression:.2%})"
        )
    return issues


# ---------- CLI ----------

def cmd_sweep(args: argparse.Namespace) -> int:
    results = sweep(Path(args.corpus), args.vyrec, args.clang, args.limit)
    Path(args.out).write_text(json.dumps(
        [asdict(r) for r in results], indent=2,
    ))
    n = len(results)
    matched = sum(1 for r in results if r.status == "match")
    diverged = sum(1 for r in results if r.status == "divergent")
    vfail = sum(1 for r in results if r.status == "vyre_failed")
    cfail = sum(1 for r in results if r.status == "clang_failed")
    print(f"sweep: {n} files | match={matched} divergent={diverged} "
          f"vyre_failed={vfail} clang_failed={cfail}", file=sys.stderr)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    issues: list[str] = []

    # 1. Re-run sweep on PR tree.
    pr_results = sweep(Path(args.corpus), args.vyrec, args.clang, args.limit)
    pr_divergent = {r.path for r in pr_results if r.status == "divergent"}

    # 2. Compare to baseline divergences.
    baseline = json.loads(Path(args.baseline_divergences).read_text())
    base_divergent = {r["path"] for r in baseline if r.get("status") == "divergent"}
    base_matched = {r["path"] for r in baseline if r.get("status") == "match"}

    regressed = (base_matched & pr_divergent)
    if regressed:
        issues.append(
            f"regressions: {len(regressed)} previously-passing files now diverge: "
            + ", ".join(sorted(list(regressed))[:5])
            + (" …" if len(regressed) > 5 else "")
        )
    if len(pr_divergent) >= len(base_divergent):
        issues.append(
            f"no-progress: pr divergence count {len(pr_divergent)} "
            f">= baseline {len(base_divergent)}"
        )

    # 3. Hardcode lint.
    issues.extend(lint_diff_for_hardcodes(
        Path(args.pr_tree), args.baseline_ref, args.pr_ref,
    ))

    # 4. Each new test must fail on baseline.
    for test in find_new_tests(
        Path(args.pr_tree), args.baseline_ref, args.pr_ref,
        args.new_tests_glob,
    ):
        msg = assert_test_fails_on_baseline(
            Path(args.pr_tree), args.baseline_ref, test,
        )
        if msg:
            issues.append(msg)

    # 5. Parser progress cannot regress AST throughput or dispatch evidence.
    issues.extend(compare_bench_reports(
        args.baseline_bench_report,
        args.pr_bench_report,
        args.max_bench_regression,
    ))

    if issues:
        print("GATE: REJECT", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 1
    print("GATE: PASS", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="divergence-gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="Run vyre+clang on corpus, emit divergences JSON")
    s.add_argument("--corpus", required=True)
    s.add_argument("--vyrec", required=True)
    s.add_argument("--clang", default="clang")
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_sweep)

    g = sub.add_parser("gate", help="Quality-gate a PR worktree")
    g.add_argument("--corpus", required=True)
    g.add_argument("--vyrec", required=True)
    g.add_argument("--clang", default="clang")
    g.add_argument("--baseline-divergences", required=True,
                   help="Path to divergences.json produced by `sweep` on baseline")
    g.add_argument("--pr-tree", required=True, help="PR worktree root")
    g.add_argument("--pr-ref", default="HEAD")
    g.add_argument("--baseline-ref", default="origin/main")
    g.add_argument("--new-tests-glob",
                   default="vyre-frontend-c/tests/divergence/*.rs")
    g.add_argument("--baseline-bench-report",
                   help="Baseline JSON benchmark report with wall_ns samples")
    g.add_argument("--pr-bench-report",
                   help="PR JSON benchmark report with wall_ns samples")
    g.add_argument("--max-bench-regression", type=float, default=0.05,
                   help="Maximum allowed median wall-time regression fraction")
    g.add_argument("--limit", type=int, default=None)
    g.set_defaults(func=cmd_gate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
