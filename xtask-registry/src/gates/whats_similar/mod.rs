//! `cargo xtask whats-similar --op-id <id>`  -  pre-write similarity query.
//!
//! Surfaces the "should I reimplement?" question at write-time, before
//! a near-duplicate op lands in the registry. Walks every registered
//! op (Tier 2, 2.5, and 3), fingerprints them, and reports the top-N
//! nearest matches by bigram-cosine structural similarity.
//!
//! The fingerprint is the same one `lego-audit` check 1 uses  -  bigram
//! cosine over the IR-shape fingerprint. Two ops with score ≥ 0.80 are
//! candidates for merging or for extracting a shared Tier-2.5 primitive.
//!
//! ## Usage
//!
//! ```text
//! # Score a registered op against everything else.
//! cargo xtask whats-similar --op-id vyre-libs::math::matmul_strassen_2x2
//!
//! # Top 10 instead of the default 5.
//! cargo xtask whats-similar --op-id vyre-libs::math::matmul --top 10
//!
//! # Lower the floor (defaults to 0.20  -  anything weaker is noise).
//! cargo xtask whats-similar --op-id ... --min 0.05
//!
//! # Scan the whole registered-op surface for near duplicates.
//! cargo xtask whats-similar --all --top 50
//! ```
//!
//! Pre-write workflow: submit the candidate as an `OperationRegistration`, run
//! whats-similar against its id, decide whether to reuse, merge, or ship as new.
//! The fingerprint sees the IR shape, not the function name, so renaming will
//! not hide a duplicate.
//!
//! ## Why not file-based?
//!
//! A `.rs` file with un-registered ops cannot produce a Program without
//! the inventory plumbing, so the fingerprint cannot be computed
//! directly from source. `--op-id` requires the candidate to be a
//! registered (even draft) entry. This is the right gate: if you
//! cannot register the op, you do not yet know what shape it builds.//!
//! ## Layout
//!
//! - `cli` the selection this gate accepts
//! - `query` the one-op and all-pairs scans
//! - `pair_facts` what a pair shares: contract, family, tier
//! - `report` the duplicate family report written as JSON

use xtask::gate::{Gate, GateCtx, GateError, Report};

use crate::gates::lego_audit::collect_ops;

use self::cli::{parse_args, Mode};
use self::query::{run_all_pairs_query, run_target_query};

mod cli;
mod pair_facts;
mod query;
mod report;

/// Reports registered operations that duplicate each other by IR shape.
pub struct WhatsSimilar;

impl Gate for WhatsSimilar {
    fn name(&self) -> &'static str {
        "whats-similar"
    }

    fn help(&self) -> &'static str {
        "Report duplicate operations by IR shape across the whole registry; --op-id ID narrows to one"
    }

    fn usage(&self) -> &'static [&'static str] {
        &[
            "--op-id ID narrows the comparison to one registered operation",
        ]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut argv = vec![String::from("xtask"), String::from("whats-similar")];
        argv.extend(ctx.args.iter().cloned());
        if ctx.flag("--op-id").is_none() && !ctx.has("--all") {
            argv.push(String::from("--all"));
        }
        let cli = parse_args(&argv).map_err(|error| {
            GateError::new(
                error,
                "pass --op-id ID to narrow to one operation, or no selection to scan every pair",
            )
        })?;
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        match &cli.mode {
            Mode::Target(op_id) => run_target_query(
                &mut report,
                &ops,
                op_id,
                cli.top_n,
                cli.min_score,
                cli.duplicate_report_json.as_ref(),
            )?,
            Mode::All => run_all_pairs_query(
                &mut report,
                &ops,
                cli.top_n,
                cli.min_score,
                cli.duplicate_report_json.as_ref(),
            )?,
        }
        Ok(report)
    }
}
