//! Render a unix epoch second as an RFC 3339 UTC instant.
//!
//! A release provenance document states when the release it describes was
//! produced. A wall clock read makes that value differ on every run, which a
//! byte-compared artifact cannot survive, so the second comes from the tree
//! itself and this turns it into the format the documents declare.

/// Seconds in one day.
const DAY: i64 = 86_400;

/// `seconds` since the unix epoch as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// The date arithmetic is the days-from-civil inverse: shift the era so a leap
/// day lands at the end of a 400 year cycle, which removes every special case
/// from the month and year split.
#[must_use]
pub fn rfc3339_utc(seconds: i64) -> String {
    let days = seconds.div_euclid(DAY);
    let time = seconds.rem_euclid(DAY);
    let (year, month, day) = civil_from_days(days);
    let hour = time / 3600;
    let minute = (time % 3600) / 60;
    let second = time % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// The proleptic Gregorian `(year, month, day)` `days` after 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    (year + i64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the format is what a CycloneDX or in-toto consumer parses, and the
    /// date arithmetic has to survive a leap day, a century that is not a leap
    /// year, a century that is, and the epoch itself. Each case below is a
    /// known instant, so a rewrite of the era arithmetic cannot pass by
    /// agreeing with itself.
    ///
    /// What it does not catch: leap seconds. Unix time does not carry them and
    /// neither does this.
    #[test]
    fn a_known_instant_renders_in_rfc_3339() {
        for (seconds, rendered) in [
            (0_i64, "1970-01-01T00:00:00Z"),
            (1, "1970-01-01T00:00:01Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (4_107_542_400, "2100-03-01T00:00:00Z"),
            (1_789_016_022, "2026-09-10T04:53:42Z"),
            (-1, "1969-12-31T23:59:59Z"),
        ] {
            assert_eq!(rfc3339_utc(seconds), rendered, "at {seconds}");
        }
    }

    /// WHY: a document rendered twice from the same tree must be byte
    /// identical, which is the whole reason the value is not read from a
    /// clock.
    #[test]
    fn the_same_second_renders_the_same_text() {
        assert_eq!(rfc3339_utc(1_789_016_022), rfc3339_utc(1_789_016_022));
    }
}
