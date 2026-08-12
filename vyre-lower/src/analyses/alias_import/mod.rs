//! Alias-fact import boundary for descriptor analysis.
//!
//! External dataflow integrations provide substrate-neutral alias facts through
//! this explicit API. Semantic optimization consumes facts on `Program` in
//! `vyre-foundation`.

pub use crate::analyses::alias_facts::{AliasFactSet, NoAliasFact};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_import_api_preserves_bidirectional_no_alias_facts() {
        let mut facts = AliasFactSet::default();
        facts.insert_no_alias(NoAliasFact {
            left_binding: 0,
            left_index: 10,
            right_binding: 1,
            right_index: 20,
        });

        assert!(facts.proves_no_alias(0, 10, 1, 20));
        assert!(facts.proves_no_alias(1, 20, 0, 10));
    }
}
