//! Serde for maps keyed by a child-body path.
//!
//! A `BTreeMap<Vec<usize>, V>` has a non-string key, which most formats cannot
//! express as an object. This module owns the one representation the debug
//! reports use for such a map: a sequence of key/value pairs, written once and
//! generic over the value type rather than restated per value width.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

pub(crate) fn serialize<S, V>(
    map: &BTreeMap<Vec<usize>, V>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    V: Serialize,
{
    let entries: Vec<(&Vec<usize>, &V)> = map.iter().collect();
    entries.serialize(serializer)
}

pub(crate) fn deserialize<'de, D, V>(deserializer: D) -> Result<BTreeMap<Vec<usize>, V>, D::Error>
where
    D: Deserializer<'de>,
    V: DeserializeOwned + Ord,
{
    let entries = Vec::<(Vec<usize>, V)>::deserialize(deserializer)?;
    Ok(entries.into_iter().collect())
}
