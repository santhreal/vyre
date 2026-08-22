//! Content-addressed cache identity for one authenticated neutral artifact.

use vyre_megakernel::Artifact;

/// The exact canonical artifact digest used as a pipeline-cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineFingerprint(pub [u8; 32]);

const _: fn(&Artifact) -> PipelineFingerprint = PipelineFingerprint::of;

impl PipelineFingerprint {
    /// Derive the cache fingerprint from the authenticated neutral artifact identity.
    ///
    /// Dispatch inputs, device generation, and runtime policy cannot alter this
    /// key. Target payload format and device identity belong in cache metadata.
    #[must_use]
    pub fn of(artifact: &Artifact) -> Self {
        Self(artifact.digest().0)
    }

    /// Hex-encode the fingerprint for human display + path-safe
    /// storage. Lowercase, no separators, 64 chars.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(64);
        self.push_hex(&mut out);
        out
    }

    pub(super) fn push_hex(&self, out: &mut String) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for &byte in &self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
}

// Inline: `vyre_runtime::pipeline_cache::fingerprint` is `private`, so no integration test can
// reach what this suite exercises.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_cache::test_artifact_fixtures::{artifact_for_program, tiny_artifact};
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    #[test]
    fn fingerprint_is_deterministic() {
        let a = PipelineFingerprint::of(&tiny_artifact());
        let b = PipelineFingerprint::of(&tiny_artifact());
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_hex_is_64_chars() {
        let fp = PipelineFingerprint::of(&tiny_artifact());
        assert_eq!(fp.hex().len(), 64);
    }

    #[test]
    fn distinct_artifacts_do_not_share_fingerprint() {
        let first = tiny_artifact();
        let second = artifact_for_program(Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(43))],
        ));
        assert_ne!(
            PipelineFingerprint::of(&first),
            PipelineFingerprint::of(&second)
        );
    }

    #[test]
    fn fingerprint_changes_when_declared_program_shape_changes() {
        let base = tiny_artifact();
        let widened = artifact_for_program(Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        ));

        assert_ne!(
            PipelineFingerprint::of(&base),
            PipelineFingerprint::of(&widened),
            "neutral artifact geometry must change the fingerprint"
        );
    }
}
