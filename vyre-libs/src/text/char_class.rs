//! `vyre-libs::text::char_class`: the registered composition over
//! [`vyre_primitives::text::char_class`].
//!
//! The op id is `vyre-libs::text::char_class` and the builder, reference
//! oracle and lookup table live in `vyre-primitives::text`, so every parser
//! dialect classifies bytes through one kernel instead of one per dialect.

#[cfg(any(test, feature = "cpu-parity"))]
pub use vyre_primitives::text::reference_char_class;
pub use vyre_primitives::text::{
    build_char_class_table, char_class, C_ALPHA, C_AMP, C_BACKSLASH, C_BANG, C_CARET,
    C_CLOSE_BRACE, C_CLOSE_BRACKET, C_CLOSE_PAREN, C_COMMA, C_DIGIT, C_DOT, C_DQUOTE, C_EOF,
    C_EQUALS, C_GT, C_HASH, C_LT, C_MINUS, C_NEWLINE, C_OPEN_BRACE, C_OPEN_BRACKET, C_OPEN_PAREN,
    C_OTHER, C_PERCENT, C_PIPE, C_PLUS, C_QUOTE, C_SEMICOLON, C_SLASH, C_STAR, C_TILDE, C_WS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn run(bytes: &[u8]) -> Vec<u32> {
        let table = build_char_class_table();
        let n = bytes.len().max(1) as u32;
        let program = char_class("source", "classified", n);
        let inputs = vec![
            Value::Bytes(vyre_primitives::wire::pack_bytes_as_u32_slice(bytes).into()),
            Value::Bytes(vyre_primitives::wire::pack_u32_slice(&table).into()),
            Value::Bytes(vec![0u8; (n as usize) * 4].into()),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: char_class must run; restore this invariant before continuing.");
        vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
    }

    #[test]
    fn classifies_ascii_letter_as_alpha() {
        assert_eq!(run(b"Hello"), vec![C_ALPHA; 5]);
    }

    #[test]
    fn classifies_digits() {
        assert_eq!(run(b"123"), vec![C_DIGIT; 3]);
    }

    #[test]
    fn classifies_whitespace_and_newline() {
        assert_eq!(run(b" \t\n"), vec![C_WS, C_WS, C_NEWLINE]);
    }

    #[test]
    fn classifies_operators() {
        assert_eq!(run(b"+-*/"), vec![C_PLUS, C_MINUS, C_STAR, C_SLASH]);
    }

    #[test]
    fn classifies_punctuation() {
        assert_eq!(
            run(b"(){}"),
            vec![C_OPEN_PAREN, C_CLOSE_PAREN, C_OPEN_BRACE, C_CLOSE_BRACE]
        );
    }

    #[test]
    fn identifier_chars_include_underscore() {
        assert_eq!(run(b"_a"), vec![C_ALPHA, C_ALPHA]);
    }
}
