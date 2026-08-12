//! Direct packed-byte GPU literal scanner.
//!
//! This is the focused public entry point for the packed-haystack scanner
//! implemented by [`GpuLiteralSet`]. Keeping this wrapper thin prevents a
//! second scanner implementation from drifting out of conformance with the
//! literal-set engine.

use crate::literal_set::GpuLiteralSet;
use vyre_foundation::ir::Program;
pub use vyre_foundation::match_result::ByteRange;

/// State for a pipelined direct-to-GPU scan.
pub struct DirectGpuScanner {
    literal_set: GpuLiteralSet,
}

impl DirectGpuScanner {
    /// Compile a set of literal patterns into a direct GPU matcher.
    #[must_use]
    pub fn compile(patterns: &[&[u8]]) -> Self {
        Self {
            literal_set: GpuLiteralSet::compile(patterns),
        }
    }

    /// Return the compiled packed-byte GPU program.
    #[must_use]
    pub fn program(&self) -> &Program {
        &self.literal_set.program
    }

    /// Cache identity of the underlying literal set. Used by the
    /// `MatchScan::cache_key` impl so DirectGpuScanner caches don't
    /// fork from the literal-set caches.
    #[must_use]
    pub fn literal_set_cache_key(&self) -> String {
        use crate::MatchScan;
        MatchScan::cache_key(&self.literal_set)
    }

    /// CPU oracle for parity and tests.
    #[must_use]
    pub fn reference_scan(&self, haystack: &[u8]) -> Vec<ByteRange> {
        self.literal_set.reference_scan(haystack)
    }

    /// Dispatch the direct matcher through a registered artifact target.
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] when compilation, materialization,
    /// submission, readback, or resident cleanup fails.
    pub fn scan(
        &self,
        backend_id: &str,
        haystack: &[u8],
        max_matches: u32,
    ) -> Result<Vec<ByteRange>, vyre_driver::BackendError> {
        let session =
            self.literal_set
                .prepare_resident_scan(backend_id, haystack.len(), max_matches)?;
        let mut matches = Vec::new();
        let mut scratch = Vec::new();
        let scan_result = session.scan_into(haystack, &mut matches, &mut scratch);
        let free_result = session.free();
        scan_result?;
        free_result?;
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_gpu_scanner_reuses_real_literal_set_program() {
        let patterns: [&[u8]; 2] = [b"abc", b"bc"];
        let scanner = DirectGpuScanner::compile(&patterns);
        let literal_set = GpuLiteralSet::compile(&patterns);
        assert_eq!(
            scanner.reference_scan(b"zabc"),
            vec![ByteRange::new(0, 1, 4), ByteRange::new(1, 2, 4)]
        );
        assert_eq!(
            scanner.program().fingerprint(),
            literal_set.program.fingerprint()
        );
        assert_eq!(
            scanner.program().workgroup_size(),
            literal_set.program.workgroup_size()
        );
        assert!(!scanner.program().entry().is_empty());
    }
}
