use crate::markdown_table::markdown_cells;

pub(crate) const VX_PLAN_TABLE_HEADER: &str =
    "| ID | Axis | Local evidence | Research basis | Work | Proof gate | Dedup seam |";
pub(crate) const VX_PLAN_MIN_ROWS: usize = 480;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct RawVxPlanRow {
    pub(crate) line: usize,
    pub(crate) id: String,
    pub(crate) axis: String,
    pub(crate) local_evidence: String,
    pub(crate) research_basis: String,
    pub(crate) work: String,
    pub(crate) proof_gate: String,
    pub(crate) dedup_seam: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct RawVxPlanTable {
    pub(crate) rows: Vec<RawVxPlanRow>,
    pub(crate) failures: Vec<String>,
    pub(crate) saw_header: bool,
}

/// Parse a VX plan-row id (`VX-001`, `VX-1004`) into its number.
///
/// The single owner of the row-id shape. Three call sites each carried their own copy that
/// required exactly three digits, so the ledger and the plan table diverged from the gate
/// the moment the plan passed VX-999: `RESEARCH_SOURCE_LEDGER.toml` cites `VX-1000` and up,
/// and the gate rejected every source row that referenced one.
///
/// The shape is `VX-` then at least three digits. A number below 1000 keeps its
/// zero-padded three-digit form (`VX-007`, never `VX-0007`) so one row has one spelling and
/// ids sort the same as text and as numbers.
pub(crate) fn vx_row_number(id: &str) -> Option<u32> {
    let digits = id.trim().strip_prefix("VX-")?;
    if digits.len() < 3 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if digits.len() > 3 && digits.starts_with('0') {
        return None;
    }
    digits.parse::<u32>().ok()
}

/// True when `id` is a well-formed VX plan-row id; see [`vx_row_number`].
pub(crate) fn is_vx_row_id(id: &str) -> bool {
    vx_row_number(id).is_some()
}

pub(crate) fn parse_raw_vx_plan_table(plan: &str) -> RawVxPlanTable {
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut saw_header = false;
    for (idx, line) in plan.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim() == VX_PLAN_TABLE_HEADER {
            saw_header = true;
        }
        if !line.starts_with("| VX-") {
            continue;
        }
        let cells = markdown_cells(line);
        if cells.len() != 7 {
            failures.push(format!(
                "line {line_no}: VX row has {} cells, expected 7",
                cells.len()
            ));
            continue;
        }
        rows.push(RawVxPlanRow {
            line: line_no,
            id: cells[0].to_string(),
            axis: cells[1].to_string(),
            local_evidence: cells[2].to_string(),
            research_basis: cells[3].to_string(),
            work: cells[4].to_string(),
            proof_gate: cells[5].to_string(),
            dedup_seam: cells[6].to_string(),
        });
    }
    RawVxPlanTable {
        rows,
        failures,
        saw_header,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plan outgrew three digits; the id shape has to follow it.
    ///
    /// `RESEARCH_SOURCE_LEDGER.toml` cites `VX-1000` and above. Three separate copies of
    /// this predicate hardcoded `digits.len() != 3`, so the research-ledger gate rejected
    /// every source row referencing a four-digit row and six xtask suites failed.
    #[test]
    fn vx_row_ids_accept_three_and_four_digit_numbers() {
        assert_eq!(vx_row_number("VX-001"), Some(1));
        assert_eq!(vx_row_number("VX-480"), Some(480));
        assert_eq!(vx_row_number("VX-999"), Some(999));
        assert_eq!(vx_row_number("VX-1000"), Some(1000));
        assert_eq!(vx_row_number("VX-1016"), Some(1016));
    }

    /// One row must have exactly one spelling.
    ///
    /// Allowing `VX-0007` alongside `VX-007` would let two ids name the same row, so
    /// dedup-by-id silently splits and a row can be worked twice.
    #[test]
    fn vx_row_ids_reject_alternate_spellings_of_the_same_number() {
        assert_eq!(vx_row_number("VX-0007"), None);
        assert_eq!(vx_row_number("VX-01000"), None);
    }

    /// Everything that is not a VX row id is rejected.
    #[test]
    fn vx_row_ids_reject_malformed_input() {
        for id in ["", "VX-", "VX-1", "VX-12", "vx-001", "VX-00a", "VX 001", "001"] {
            assert!(
                !is_vx_row_id(id),
                "Fix: `{id}` is not a well-formed VX plan-row id."
            );
        }
    }

    #[test]
    fn raw_vx_plan_rows_preserve_line_and_cells() {
        let plan = "\n| VX-001 | coordination | local | `MLIR_PASS` | Fix: x | Proof gate. | seam |\n";

        let table = parse_raw_vx_plan_table(plan);

        assert_eq!(table.failures, Vec::<String>::new());
        assert_eq!(table.rows[0].line, 2);
        assert_eq!(table.rows[0].id, "VX-001");
        assert_eq!(table.rows[0].research_basis, "`MLIR_PASS`");
    }
}
