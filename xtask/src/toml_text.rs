//! TOML scalars a gate reads from a row and writes into a document.

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

/// One string field of a row, empty when the key is absent or not a string.
///
/// Every generated-document row declares its fields as strings, and each field
/// has its own rule that reports the empty case with the sentence naming that
/// field, which reads better than one sentence about a malformed row. The two
/// document generators held the same closure.
#[must_use]
pub fn string_field(row: &toml::Table, key: &str) -> String {
    row.get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The strings of a TOML array value, empty when the value is absent, is not
/// an array, or holds no string.
///
/// A declared list of names is read the same way wherever it appears: a
/// `covers` list on a documentation page, a `features` list on a dependency, a
/// roster in a generated document. Three copies of the same
/// `as_array`/`filter_map(as_str)` chain drift by dropping the
/// non-string case, which reads as an empty list at one caller and a panic at
/// another.
#[must_use]
pub fn string_array(value: Option<&toml::Value>) -> Vec<String> {
    value
        .and_then(toml::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{array, quote, string_array};

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

    #[test]
    fn a_string_array_keeps_the_strings_and_drops_everything_else() {
        let table: toml::Table =
            toml::from_str("mixed = [\"one\", 2, \"three\", true]\nempty = []\nscalar = \"one\"\n")
                .expect("Fix: the fixture must be valid TOML.");

        assert_eq!(
            string_array(table.get("mixed")),
            vec!["one".to_string(), "three".to_string()]
        );
        assert!(string_array(table.get("empty")).is_empty());
        assert!(string_array(table.get("scalar")).is_empty());
        assert!(string_array(table.get("absent")).is_empty());
    }
}
