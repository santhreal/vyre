//! TOML scalars a gate writes into a document it generates.

/// One TOML basic string, so a value containing a quote or a backslash
/// round-trips through the document.
#[must_use]
pub fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// One TOML array of basic strings on a single line.
pub fn array<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut text = String::from("[");
    for (index, value) in values.into_iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        text.push_str(&quote(value));
    }
    text.push(']');
    text
}

#[cfg(test)]
mod tests {
    use super::{array, quote};

    #[test]
    fn a_quoted_value_escapes_the_characters_toml_reserves() {
        assert_eq!(quote("plain"), "\"plain\"");
        assert_eq!(quote("a\"b"), "\"a\\\"b\"");
        assert_eq!(quote("a\\b"), "\"a\\\\b\"");
        assert_eq!(quote("a\\\"b"), "\"a\\\\\\\"b\"");
    }

    #[test]
    fn an_array_separates_quoted_values_and_stays_on_one_line() {
        assert_eq!(array(Vec::<&str>::new()), "[]");
        assert_eq!(array(["one"]), "[\"one\"]");
        assert_eq!(array(["one", "two"]), "[\"one\", \"two\"]");
    }
}
