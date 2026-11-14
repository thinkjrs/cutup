use super::*;
use cutup::WeightBounds;
use serde_json::json;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// Two platforms over four days, with nothing spent on A on the second.
const SMALL: &str = "date,platform,spend,impressions,clicks,link_clicks,views
2025-09-01,A,100,1000,10,5,200
2025-09-01,B,50,2000,4,2,100
2025-09-02,A,0,0,0,0,0
2025-09-02,B,40,1500,3,2,80
2025-09-03,A,80,900,9,4,150
2025-09-03,B,60,2500,5,2,120
2025-09-04,A,90,1100,12,6,170
2025-09-04,B,30,900,2,2,70
";

fn request(json: serde_json::Value) -> Request {
    serde_json::from_value(json).unwrap()
}

fn small() -> Sample {
    Sample::parse(SMALL).unwrap()
}

#[test]
fn a_request_leaves_out_what_has_defaults() {
    let read = request(json!({"method": {"name": "equal_weight"}}));
    assert_eq!(read.platforms, DEFAULT_PLATFORMS.map(String::from));
    assert_eq!(read.metrics, Metric::ALL);
    assert_eq!(read.method, Method::EqualWeight(EqualWeight));
    assert_eq!(
        (read.from, read.to, read.frontier, read.rolling),
        (None, None, None, None)
    );
    let read = request(json!({
        "platforms": ["A"],
        "metrics": ["link_clicks"],
        "method": {"name": "min_variance", "bounds": {"upper": 0.6}},
        "rolling": {"window": 3}
    }));
    assert_eq!(
        read.method,
        Method::MinVariance(MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.6)))
    );
    assert_eq!(read.rolling, Some(RollingRequest { window: 3, step: 1 }));
    for bad in [
        json!({"method": {"name": "min_variance", "uper": 0.6}}),
        // A method with no settings refuses them rather than allocating without them.
        json!({"method": {"name": "inverse_volatility", "bounds": {"upper": 0.3}}}),
        json!({"method": {"name": "best"}}),
        json!({"method": {"name": "equal_weight"}, "platform": ["A"]}),
        json!({"platforms": ["A"]}),
    ] {
        assert!(
            serde_json::from_value::<Request>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn each_metric_allocates_on_its_normalized_yields() {
    let read = request(json!({
        "platforms": ["A", "B"],
        "metrics": ["impressions", "clicks"],
        "method": {"name": "min_variance"}
    }));
    let response = allocate_over(&small(), &read).unwrap();
    // 2025-09-02 has no spend on A.
    assert_eq!(response.days, ["2025-09-01", "2025-09-03", "2025-09-04"]);
    assert_eq!(response.platforms, ["A", "B"]);
    assert_eq!(response.results.len(), 2);
    for (result, metric) in response
        .results
        .iter()
        .zip([Metric::Impressions, Metric::Clicks])
    {
        assert_eq!(result.metric, metric);
        // The same as cutup on the yields by hand, each divided by their average.
        let totals: [[(f64, f64); 2]; 3] = [
            [
                (
                    100.0,
                    if metric == Metric::Clicks {
                        10.0
                    } else {
                        1000.0
                    },
                ),
                (
                    50.0,
                    if metric == Metric::Clicks {
                        4.0
                    } else {
                        2000.0
                    },
                ),
            ],
            [
                (80.0, if metric == Metric::Clicks { 9.0 } else { 900.0 }),
                (
                    60.0,
                    if metric == Metric::Clicks {
                        5.0
                    } else {
                        2500.0
                    },
                ),
            ],
            [
                (
                    90.0,
                    if metric == Metric::Clicks {
                        12.0
                    } else {
                        1100.0
                    },
                ),
                (30.0, if metric == Metric::Clicks { 2.0 } else { 900.0 }),
            ],
        ];
        let yields: Vec<[f64; 2]> = totals
            .iter()
            .map(|day| day.map(|(spend, total)| total / spend))
            .collect();
        let mean = yields.iter().flatten().sum::<f64>() / 6.0;
        let rows: Vec<[f64; 2]> = yields.iter().map(|day| day.map(|y| y / mean)).collect();
        let data = Returns::new(&rows).unwrap().labeled(["A", "B"]).unwrap();
        let expected = data.allocate(&MinVariance::default()).unwrap();
        assert_eq!(result.weights.as_ref(), Some(&expected), "{metric:?}");
        let stats = result.stats.as_ref().unwrap();
        assert!(
            (stats.expected_return - data.stats(&expected).unwrap().expected_return()).abs()
                < 1e-15
        );
        assert!(result.error.is_none() && result.frontier.is_none() && result.rolling.is_none());
    }
}

#[test]
fn a_metric_the_method_cant_allocate_has_its_own_error() {
    // B's link clicks are 0.04 per unit of spend every day, so their yield doesn't vary, and
    // min variance can't allocate link clicks. Clicks still allocate.
    let flat = "date,platform,spend,impressions,clicks,link_clicks,views
2025-09-01,A,100,1000,10,5,200
2025-09-01,B,50,2000,4,2,100
2025-09-02,A,80,900,9,4,150
2025-09-02,B,100,2500,5,4,120
2025-09-03,A,90,1100,12,6,170
2025-09-03,B,25,900,2,1,70
";
    let sample = Sample::parse(flat).unwrap();
    let read = request(json!({
        "platforms": ["A", "B"],
        "metrics": ["clicks", "link_clicks"],
        "method": {"name": "min_variance"}
    }));
    let response = allocate_over(&sample, &read).unwrap();
    assert!(response.results[0].weights.is_some());
    // B's link clicks are 0.04 per unit of spend every day.
    let link_clicks = &response.results[1];
    assert_eq!(
        link_clicks.error,
        Some(CutupError::ZeroVariance {
            column: 1,
            asset: "B".into()
        })
    );
    assert!(link_clicks.weights.is_none() && link_clicks.stats.is_none());
}

#[test]
fn a_yield_that_overflows_is_its_metrics_error() {
    // A spend so small that 1 over it is infinite.
    let tiny = "date,platform,spend,impressions,clicks,link_clicks,views
2025-09-01,A,1e-310,1,1,1,1
2025-09-02,A,1,1,1,1,1
2025-09-03,A,2,1,1,1,1
";
    let sample = Sample::parse(tiny).unwrap();
    let read = request(json!({
        "platforms": ["A"],
        "metrics": ["clicks"],
        "method": {"name": "equal_weight"}
    }));
    let response = allocate_over(&sample, &read).unwrap();
    assert!(matches!(
        response.results[0].error,
        Some(CutupError::NonFiniteValue { .. })
    ));
}

#[test]
fn a_frontier_and_rolling_windows_come_with_each_metric() {
    let read = request(json!({
        "platforms": ["A", "B"],
        "metrics": ["views"],
        "method": {"name": "mean_variance", "risk_aversion": 2.0},
        "frontier": 4,
        "rolling": {"window": 2}
    }));
    let response = allocate_over(&small(), &read).unwrap();
    let result = &response.results[0];
    assert!(!result.frontier.as_ref().unwrap().is_empty());
    let windows = result.rolling.as_ref().unwrap();
    assert_eq!(
        windows.iter().map(Window::periods).collect::<Vec<_>>(),
        [0..2, 1..3]
    );
    assert!(result.frontier_error.is_none());

    // A frontier doesn't take rebalancing settings, which the allocation does: it says so, and
    // the weights still come.
    let read = request(json!({
        "platforms": ["A", "B"],
        "metrics": ["views"],
        "method": {
            "name": "mean_variance",
            "risk_aversion": 2.0,
            "current_weights": [0.5, 0.5],
            "turnover_penalty": 0.1
        },
        "frontier": 4
    }));
    let result = &allocate_over(&small(), &read).unwrap().results[0];
    assert!(result.weights.is_some() && result.frontier.is_none());
    assert_eq!(
        result.frontier_error,
        Some(CutupError::NotSupported {
            setting: "turnover_penalty",
            with: "a frontier"
        })
    );
}

#[test]
fn a_request_that_cant_be_allocated_says_why() {
    let sample = small();
    let method = json!({"name": "equal_weight"});
    for (json, error) in [
        (
            json!({"platforms": [], "method": method}),
            EngineError::NoPlatforms,
        ),
        (
            json!({"platforms": ["A", "C"], "method": method}),
            EngineError::UnknownPlatform {
                platform: "C".into(),
            },
        ),
        (
            json!({"platforms": ["A", "A"], "method": method}),
            EngineError::DuplicatePlatform {
                platform: "A".into(),
            },
        ),
        (
            json!({"platforms": ["A"], "metrics": [], "method": method}),
            EngineError::NoMetrics,
        ),
        (
            json!({"platforms": ["A"], "from": "2025-9-1", "method": method}),
            EngineError::InvalidDate {
                date: "2025-9-1".into(),
            },
        ),
        (
            json!({"platforms": ["A"], "to": "2026-02-30", "method": method}),
            EngineError::InvalidDate {
                date: "2026-02-30".into(),
            },
        ),
        (
            json!({"platforms": ["A", "B"], "from": "2025-09-04", "method": method}),
            EngineError::TooFewDays { days: 1 },
        ),
        (
            json!({"platforms": ["A"], "frontier": 3, "method": method}),
            EngineError::FrontierNeedsMeanVariance,
        ),
        (
            json!({"platforms": ["A"], "frontier": 1, "method": {"name": "mean_variance", "risk_aversion": 1.0}}),
            EngineError::FrontierPoints { points: 1 },
        ),
        (
            json!({
                "platforms": ["A"],
                "frontier": MAX_FRONTIER_POINTS + 1,
                "method": {"name": "mean_variance", "risk_aversion": 1.0}
            }),
            EngineError::FrontierPoints {
                points: MAX_FRONTIER_POINTS + 1,
            },
        ),
        (
            json!({"platforms": ["A", "B"], "rolling": {"window": 4}, "method": method}),
            EngineError::Rolling {
                error: CutupError::InvalidConfig {
                    field: "window",
                    value: 4.0,
                },
            },
        ),
    ] {
        assert_eq!(
            allocate_over(&sample, &request(json.clone())),
            Err(error),
            "{json}"
        );
    }
}

#[test]
fn a_reply_is_a_response_or_an_error_with_its_status() {
    let ok = respond(r#"{"method": {"name": "equal_weight"}, "metrics": ["views"]}"#);
    assert_eq!(ok.status, 200);
    let response: serde_json::Value = serde_json::from_str(&ok.body).unwrap();
    assert_eq!(response["results"][0]["metric"], "views");
    for (request, status, kind) in [
        ("{", 400, "request"),
        (r#"{"method": {"name": "best"}}"#, 400, "request"),
        (
            r#"{"platforms": ["Myspace"], "method": {"name": "equal_weight"}}"#,
            422,
            "unknown_platform",
        ),
    ] {
        let reply = respond(request);
        assert_eq!(reply.status, status, "{request}");
        let body: serde_json::Value = serde_json::from_str(&reply.body).unwrap();
        assert!(body["error"][kind].is_object(), "{request}: {body}");
    }
    let not_allowed = not_allowed();
    assert_eq!(not_allowed.status, 405);
    assert_eq!(
        not_allowed.body,
        r#"{"error":{"request":{"message":"use POST, with a JSON request"}}}"#
    );
    let platforms: Vec<serde_json::Value> = serde_json::from_str(&platforms()).unwrap();
    assert_eq!(platforms.len(), 6);
    assert_eq!(platforms[1]["platform"], "Meta");
    assert_eq!(platforms[1]["days_with_spend"], 395);
}

#[test]
fn the_bundled_sample_allocates_every_metric_by_default() {
    let response = allocate(&request(json!({"method": {"name": "min_variance"}}))).unwrap();
    assert_eq!(response.platforms, DEFAULT_PLATFORMS);
    // Meta, Google and TikTok spent on every day of the sample.
    assert_eq!(response.days.len(), 395);
    assert_eq!(response.results.len(), 4);
    for result in &response.results {
        let weights = result.weights.as_ref().unwrap();
        assert!(
            (weights.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "{result:?}"
        );
        assert_eq!(weights.labels(), DEFAULT_PLATFORMS);
    }
    // And it writes JSON a client can read: every result's weights by platform.
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["results"][0]["metric"], "impressions");
    assert!(json["results"][3]["weights"]["TikTok"].is_number());
}

#[test]
fn the_typescript_types_are_the_json() {
    /// A declaration without its doc comments, on one line.
    fn bare(decl: String) -> String {
        let mut out = String::new();
        let mut rest = decl.as_str();
        while let Some(start) = rest.find("/**") {
            out.push_str(&rest[..start]);
            let end = rest[start..].find("*/").expect("a closed comment");
            rest = &rest[start + end + 2..];
        }
        out.push_str(rest);
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    let cfg = ts_rs::Config::new();
    assert_eq!(
        bare(Request::decl(&cfg)),
        "type Request = { platforms?: Array<string>, from?: string, to?: string, \
         metrics?: Array<Metric>, method: Method, frontier?: number, \
         rolling?: RollingRequest, };"
    );
    assert_eq!(
        bare(RollingRequest::decl(&cfg)),
        "type RollingRequest = { window: number, step?: number, };"
    );
    assert_eq!(
        bare(MetricResult::decl(&cfg)),
        "type MetricResult = { metric: Metric, weights?: Weights, stats?: Stats, \
         error?: CutupError, frontier?: Array<FrontierPoint>, frontier_error?: CutupError, \
         rolling?: Array<Window>, };"
    );
    let method = Method::decl(&cfg);
    assert!(
        method.starts_with("type Method = { \"name\": \"equal_weight\" } & {} | "),
        "{method}"
    );
    assert!(
        method.contains("{ \"name\": \"min_variance\" } & MinVariance"),
        "{method}"
    );
    assert_eq!(
        Metric::decl(&cfg),
        "type Metric = \"impressions\" | \"clicks\" | \"link_clicks\" | \"views\";"
    );
}
