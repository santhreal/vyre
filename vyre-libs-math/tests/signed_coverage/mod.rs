//! Negative-operand coverage accounting for the signed fixed-point parity sweeps.
//!
//! A signed sweep proves nothing until it also proves it swept: that it fed
//! negative operands, that the kernel returned negative results, and that most
//! cases moved a value at all. Five suites stated those three counters and their
//! assertion inline, so the floors drifted apart while the accounting stayed
//! identical. The counters are stated once here and each suite supplies only its
//! own floors.

/// Negative-operand, negative-result and non-zero-result counts for one sweep.
#[derive(Default)]
pub(crate) struct SignedCoverage {
    neg_operands: u32,
    neg_results: u32,
    moved: u32,
    cases: u32,
}

impl SignedCoverage {
    /// Count the negative operands one case fed the kernel.
    pub(crate) fn operands<'a>(&mut self, values: impl IntoIterator<Item = &'a u32>) {
        self.neg_operands += values
            .into_iter()
            .filter(|&&value| (value as i32) < 0)
            .count() as u32;
    }

    /// Count one case's result: its negative entries, and whether it moved.
    pub(crate) fn result(&mut self, want: &[u32]) {
        self.cases += 1;
        if want.iter().any(|&value| value != 0) {
            self.moved += 1;
        }
        self.neg_results += want.iter().filter(|&&value| (value as i32) < 0).count() as u32;
    }

    /// Fail unless the sweep cleared every floor, naming the count that fell
    /// short and the kernel whose coverage it describes.
    pub(crate) fn assert_floors(
        &self,
        kernel: &str,
        operand_floor: u32,
        result_floor: u32,
        moved_floor: u32,
    ) {
        assert!(
            self.neg_operands > operand_floor,
            "{kernel}: sweep must feed more than {operand_floor} negative operands, got {}",
            self.neg_operands
        );
        assert!(
            self.neg_results > result_floor,
            "{kernel}: signed operands must produce more than {result_floor} negative result \
             entries, got {}",
            self.neg_results
        );
        assert!(
            self.moved > moved_floor,
            "{kernel}: only {}/{} results were non-zero, the kernel is not being exercised",
            self.moved,
            self.cases
        );
    }
}
