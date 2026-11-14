//! The campaign data: each platform's daily spend and results.

use crate::EngineError;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use ts_rs::TS;

/// The bundled sample: six ad platforms' daily totals for about a year. Each volume is scaled
/// by the same constant, so every yield per unit of spend is the real one and the absolute
/// amounts aren't.
const BUNDLED: &str = include_str!("../../data/platforms_daily.csv");

/// What a day's spend bought, as a result per unit of spend.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, serde::Deserialize, TS,
)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Impressions,
    Clicks,
    LinkClicks,
    Views,
}

impl Metric {
    /// Every metric, in the order results list them.
    pub const ALL: [Metric; 4] = [
        Metric::Impressions,
        Metric::Clicks,
        Metric::LinkClicks,
        Metric::Views,
    ];
}

/// One platform's totals on one day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Totals {
    pub spend: f64,
    pub impressions: f64,
    pub clicks: f64,
    pub link_clicks: f64,
    pub views: f64,
}

impl Totals {
    /// The metric's total, per unit of spend. Only meaningful where something was spent.
    pub fn yield_of(&self, metric: Metric) -> f64 {
        let total = match metric {
            Metric::Impressions => self.impressions,
            Metric::Clicks => self.clicks,
            Metric::LinkClicks => self.link_clicks,
            Metric::Views => self.views,
        };
        total / self.spend
    }
}

/// Each platform's daily totals, by date and then platform. Dates are ISO 8601, `YYYY-MM-DD`,
/// so they sort as they come.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sample {
    days: BTreeMap<String, BTreeMap<String, Totals>>,
}

/// The columns a sample has, in order.
const COLUMNS: &str = "date,platform,spend,impressions,clicks,link_clicks,views";

/// A summary of a sample's platforms, for choosing which to allocate across.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct PlatformSummary {
    pub platform: String,
    /// The first and last dates with a row for it.
    pub first: String,
    pub last: String,
    /// How many of its days had any spend.
    pub days_with_spend: usize,
}

impl Sample {
    /// The bundled sample, parsed once.
    pub fn bundled() -> &'static Sample {
        static SAMPLE: OnceLock<Sample> = OnceLock::new();
        SAMPLE.get_or_init(|| Sample::parse(BUNDLED).expect("the bundled sample parses"))
    }

    /// Parses CSV with the columns `date,platform,spend,impressions,clicks,link_clicks,views`.
    pub fn parse(csv: &str) -> Result<Sample, EngineError> {
        let mut lines = csv.lines().enumerate();
        match lines.next() {
            Some((_, header)) if header.trim() == COLUMNS => {}
            _ => {
                return Err(EngineError::Sample {
                    line: 1,
                    message: format!("expected the header `{COLUMNS}`"),
                });
            }
        }
        let mut sample = Sample::default();
        for (i, line) in lines.filter(|(_, l)| !l.trim().is_empty()) {
            let error = |message: String| EngineError::Sample {
                line: i + 1,
                message,
            };
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [date, platform, values @ ..] = fields.as_slice() else {
                return Err(error("expected 7 values".into()));
            };
            if values.len() != 5 {
                return Err(error(format!("expected 7 values, not {}", fields.len())));
            }
            if !is_date(date) {
                return Err(error(format!("`{date}` isn't a YYYY-MM-DD date")));
            }
            let mut numbers = [0.0; 5];
            for (number, value) in numbers.iter_mut().zip(values) {
                *number = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| error(format!("`{value}` isn't a nonnegative number")))?;
            }
            let [spend, impressions, clicks, link_clicks, views] = numbers;
            let totals = Totals {
                spend,
                impressions,
                clicks,
                link_clicks,
                views,
            };
            let day = sample.days.entry((*date).to_owned()).or_default();
            if day.insert((*platform).to_owned(), totals).is_some() {
                return Err(error(format!("`{platform}` appears twice on {date}")));
            }
        }
        Ok(sample)
    }

    /// The platforms, with their dates and how many days they spent on, by name.
    pub fn platforms(&self) -> Vec<PlatformSummary> {
        let mut summaries: BTreeMap<&str, PlatformSummary> = BTreeMap::new();
        for (date, day) in &self.days {
            for (platform, totals) in day {
                let summary =
                    summaries
                        .entry(platform.as_str())
                        .or_insert_with(|| PlatformSummary {
                            platform: platform.clone(),
                            first: date.clone(),
                            last: date.clone(),
                            days_with_spend: 0,
                        });
                summary.last = date.clone();
                summary.days_with_spend += usize::from(totals.spend > 0.0);
            }
        }
        summaries.into_values().collect()
    }

    /// Whether `platform` has a row on any day.
    pub fn has_platform(&self, platform: &str) -> bool {
        self.days.values().any(|day| day.contains_key(platform))
    }

    /// The days from `from` to `to`, inclusive, either open, on which every one of `platforms`
    /// spent something, oldest first, with their totals in the order of `platforms`.
    pub fn days_with_spend<'a>(
        &'a self,
        platforms: &'a [String],
        from: Option<&'a str>,
        to: Option<&'a str>,
    ) -> impl Iterator<Item = (&'a str, Vec<Totals>)> + 'a {
        self.days
            .iter()
            .filter(move |(date, _)| {
                from.is_none_or(|from| date.as_str() >= from)
                    && to.is_none_or(|to| date.as_str() <= to)
            })
            .filter_map(move |(date, day)| {
                let totals: Option<Vec<Totals>> = platforms
                    .iter()
                    .map(|p| day.get(p).copied().filter(|t| t.spend > 0.0))
                    .collect();
                totals.map(|totals| (date.as_str(), totals))
            })
    }
}

/// Whether `date` is a real `YYYY-MM-DD` date: one whose month and day exist.
pub(crate) fn is_date(date: &str) -> bool {
    let bytes = date.as_bytes();
    let shaped = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit());
    if !shaped {
        return false;
    }
    // Every slice is ASCII digits, so each parses.
    let number = |range: std::ops::Range<usize>| date[range].parse::<u32>().unwrap_or(0);
    let (year, month, day) = (number(0..4), number(5..7), number(8..10));
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = "date,platform,spend,impressions,clicks,link_clicks,views
2025-09-01,A,100,1000,10,5,200
2025-09-01,B,50,2000,4,2,100
2025-09-02,A,0,0,0,0,0
2025-09-02,B,40,1500,3,1,80
2025-09-03,A,80,900,9,4,150
2025-09-03,B,60,2500,5,3,120
";

    #[test]
    fn a_sample_is_each_platforms_totals_by_day() {
        let sample = Sample::parse(SMALL).unwrap();
        let summaries = sample.platforms();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].platform, "A");
        assert_eq!(
            (summaries[0].first.as_str(), summaries[0].last.as_str()),
            ("2025-09-01", "2025-09-03")
        );
        assert_eq!(summaries[0].days_with_spend, 2);
        assert_eq!(summaries[1].days_with_spend, 3);
        assert!(sample.has_platform("A") && sample.has_platform("B"));
        assert!(!sample.has_platform("C"));
    }

    #[test]
    fn only_days_every_platform_spent_on_count() {
        let sample = Sample::parse(SMALL).unwrap();
        let both = ["A".to_owned(), "B".to_owned()];
        let days: Vec<&str> = sample
            .days_with_spend(&both, None, None)
            .map(|(d, _)| d)
            .collect();
        assert_eq!(days, ["2025-09-01", "2025-09-03"]);
        let b = ["B".to_owned()];
        assert_eq!(sample.days_with_spend(&b, None, None).count(), 3);
        let window: Vec<&str> = sample
            .days_with_spend(&b, Some("2025-09-02"), Some("2025-09-02"))
            .map(|(d, _)| d)
            .collect();
        assert_eq!(window, ["2025-09-02"]);
        let (_, totals) = sample.days_with_spend(&both, None, None).next().unwrap();
        assert_eq!(totals[1].yield_of(Metric::Impressions), 40.0);
        assert_eq!(totals[0].yield_of(Metric::Clicks), 0.1);
    }

    #[test]
    fn a_malformed_sample_says_where() {
        for (csv, line) in [
            ("date,platform\n", 1),
            (
                "date,platform,spend,impressions,clicks,link_clicks,views\n2025-09-01,A,1,2\n",
                2,
            ),
            (
                "date,platform,spend,impressions,clicks,link_clicks,views\n2025-9-1,A,1,2,3,4,5\n",
                2,
            ),
            (
                "date,platform,spend,impressions,clicks,link_clicks,views\n2025-09-01,A,-1,2,3,4,5\n",
                2,
            ),
            (
                "date,platform,spend,impressions,clicks,link_clicks,views\n\
                 2025-09-01,A,1,2,3,4,5\n2025-09-01,A,1,2,3,4,5\n",
                3,
            ),
        ] {
            match Sample::parse(csv) {
                Err(EngineError::Sample { line: at, .. }) => assert_eq!(at, line, "{csv}"),
                other => panic!("{csv}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_date_is_a_real_day() {
        for date in ["2025-09-01", "2024-02-29", "2000-02-29", "2025-12-31"] {
            assert!(is_date(date), "{date}");
        }
        for date in [
            "2025-9-1",
            "2025-02-29",
            "1900-02-29",
            "2026-02-30",
            "2025-04-31",
            "2025-13-01",
            "2025-00-10",
            "2025-01-00",
            "2025-99-99",
        ] {
            assert!(!is_date(date), "{date}");
        }
    }

    #[test]
    fn the_bundled_sample_has_six_platforms() {
        let platforms: Vec<String> = Sample::bundled()
            .platforms()
            .into_iter()
            .map(|p| p.platform)
            .collect();
        assert_eq!(
            platforms,
            ["Google", "Meta", "Reddit", "Snap", "Spotify", "TikTok"]
        );
    }
}
