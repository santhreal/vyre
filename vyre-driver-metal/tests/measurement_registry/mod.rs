//! One measurement-registry read, shared by every registry case in this suite.
//!
//! Each case reads a TOML registry under `docs/optimization/`, requires an array
//! of rows, and reads typed fields off every row with a `Fix:` message naming
//! the row kind. Written per case, that read drifted: two copies stated the same
//! missing-field failure differently, and one accepted an empty roster the other
//! refused. The read is stated once here and each case states only what its
//! registry must contain.

#![allow(dead_code)]

use std::collections::BTreeSet;

/// One registry file, read with the noun its failures name.
pub(crate) struct Registry {
    table: toml::Table,
    noun: &'static str,
}

impl Registry {
    /// Parse `text` as the registry `noun` describes.
    #[must_use]
    pub(crate) fn parse(text: &str, noun: &'static str) -> Self {
        let table = toml::from_str::<toml::Table>(text)
            .unwrap_or_else(|error| panic!("Fix: the {noun} registry must parse as TOML: {error}"));
        Self { table, noun }
    }

    /// Require the string at `key` to be `expected`.
    pub(crate) fn declares(&self, key: &str, expected: &str) {
        assert_eq!(
            self.table.get(key).and_then(toml::Value::as_str),
            Some(expected),
            "Fix: a {} registry `{key}` change must be recorded in this case.",
            self.noun
        );
    }

    /// Require the integer at `key` to be `expected`.
    pub(crate) fn declares_integer(&self, key: &str, expected: i64) {
        assert_eq!(
            self.table.get(key).and_then(toml::Value::as_integer),
            Some(expected),
            "Fix: a {} registry `{key}` change must be recorded in this case.",
            self.noun
        );
    }

    /// Every string the array at `key` states, which must state at least one.
    #[must_use]
    pub(crate) fn roster(&self, key: &str) -> BTreeSet<&str> {
        let values = self
            .table
            .get(key)
            .and_then(toml::Value::as_array)
            .unwrap_or_else(|| panic!("Fix: the {} registry must declare `{key}`.", self.noun));
        assert!(
            !values.is_empty(),
            "Fix: `{key}` must state at least one entry."
        );
        values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .unwrap_or_else(|| panic!("Fix: every `{key}` entry must be a string."))
            })
            .collect()
    }

    /// Every `[[key]]` row, each addressed by the identifier at `id_key`.
    #[must_use]
    pub(crate) fn rows(&self, key: &str, id_key: &'static str) -> Vec<Row<'_>> {
        let rows = self
            .table
            .get(key)
            .and_then(toml::Value::as_array)
            .unwrap_or_else(|| {
                panic!(
                    "Fix: the {} registry must declare [[{key}]] rows.",
                    self.noun
                )
            });
        assert!(
            !rows.is_empty(),
            "Fix: the {} registry must declare at least one [[{key}]] row.",
            self.noun
        );
        rows.iter()
            .map(|row| {
                let table = row
                    .as_table()
                    .unwrap_or_else(|| panic!("Fix: every [[{key}]] row must be a table."));
                let id = table
                    .get(id_key)
                    .and_then(toml::Value::as_str)
                    .unwrap_or("<unnamed>");
                Row {
                    table,
                    noun: self.noun,
                    id,
                }
            })
            .collect()
    }
}

/// One registry row, addressed by the identifier it states.
pub(crate) struct Row<'reg> {
    table: &'reg toml::Table,
    noun: &'static str,
    id: &'reg str,
}

impl<'reg> Row<'reg> {
    /// The identifier this row states.
    #[must_use]
    pub(crate) const fn id(&self) -> &'reg str {
        self.id
    }

    /// Every key this row declares.
    #[must_use]
    pub(crate) fn keys(&self) -> BTreeSet<&'reg str> {
        self.table.keys().map(String::as_str).collect()
    }

    /// Require this row to declare exactly `expected`.
    pub(crate) fn declares_exactly(&self, expected: &BTreeSet<&str>) {
        assert_eq!(
            &self.keys(),
            expected,
            "Fix: {} case `{}` must record exactly the required metrics.",
            self.noun,
            self.id
        );
    }

    /// The value at `key`, which this row must declare.
    #[must_use]
    pub(crate) fn value(&self, key: &str) -> &'reg toml::Value {
        self.table.get(key).unwrap_or_else(|| {
            panic!(
                "Fix: {} case `{}` must declare `{key}`.",
                self.noun, self.id
            )
        })
    }

    /// The string at `key`.
    #[must_use]
    pub(crate) fn text(&self, key: &str) -> &'reg str {
        self.value(key).as_str().unwrap_or_else(|| {
            panic!(
                "Fix: {} case `{}` must declare `{key}` as a string.",
                self.noun, self.id
            )
        })
    }

    /// The string at `key`, which must not be empty.
    pub(crate) fn stated(&self, key: &str) -> &'reg str {
        let text = self.text(key);
        assert!(
            !text.is_empty(),
            "Fix: {} case `{}` must state `{key}`.",
            self.noun,
            self.id
        );
        text
    }

    /// A positive measurement in nanoseconds at `key`.
    pub(crate) fn nanos(&self, key: &str) -> i64 {
        let value = self.value(key).as_integer().unwrap_or_else(|| {
            panic!(
                "Fix: {} case `{}` must declare `{key}` as an integer.",
                self.noun, self.id
            )
        });
        assert!(
            value > 0,
            "Fix: {} case `{}` must record `{key}` > 0.",
            self.noun,
            self.id
        );
        value
    }

    /// A `prefix` digest at `key` that no earlier row in `seen` recorded.
    ///
    /// Two rows sharing a digest cannot disagree about anything the digest
    /// identifies, which is what a comparison registry exists to state.
    pub(crate) fn digest(
        &self,
        key: &str,
        prefix: &str,
        seen: &mut BTreeSet<&'reg str>,
    ) -> &'reg str {
        let digest = self.text(key);
        assert!(
            digest.starts_with(prefix),
            "Fix: {} case `{}` must record `{key}` as a `{prefix}` digest.",
            self.noun,
            self.id
        );
        assert!(
            seen.insert(digest),
            "Fix: {} case `{}` repeats digest `{digest}`, so two rows cannot disagree.",
            self.noun,
            self.id
        );
        digest
    }
}
