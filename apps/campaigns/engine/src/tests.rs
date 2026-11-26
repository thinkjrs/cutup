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
        // The same as cutup on the yields by hand, each divided by their average, less 1.
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
        let rows: Vec<[f64; 2]> = yields
            .iter()
            .map(|day| day.map(|y| y / mean - 1.0))
            .collect();
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
        windows.iter().map(|w| w.start..w.end).collect::<Vec<_>>(),
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
            json!({"platforms": ["A"], "metrics": ["clicks", "views", "clicks"], "method": method}),
            EngineError::DuplicateMetric {
                metric: Metric::Clicks,
            },
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
    let rejected = reject(405, "use POST, with a JSON request");
    assert_eq!(rejected.status, 405);
    assert_eq!(
        rejected.body,
        r#"{"error":{"request":{"message":"use POST, with a JSON request"}}}"#
    );
    let platforms: Vec<serde_json::Value> = serde_json::from_str(&platforms()).unwrap();
    assert_eq!(platforms.len(), 6);
    assert_eq!(platforms[1]["platform"], "Meta");
    assert_eq!(platforms[1]["days_with_spend"], 395);
}

#[test]
fn too_many_rolling_windows_are_refused_before_any_is_solved() {
    // 395 days in windows of 2, 1 apart, are 394 windows; 14 apart, (395 − 2) / 14 + 1 = 29.
    let many = request(json!({"rolling": {"window": 2}, "method": {"name": "min_cvar"}}));
    assert_eq!(
        allocate(&many),
        Err(EngineError::RollingWindows { windows: 394 })
    );
    let spaced = request(json!({
        "metrics": ["clicks"],
        "rolling": {"window": 2, "step": 14},
        "method": {"name": "equal_weight"}
    }));
    let response = allocate(&spaced).unwrap();
    assert_eq!(response.results[0].rolling.as_ref().unwrap().len(), 29);
    // Exactly the most is allowed: 395 days in windows of 98, 3 apart, are 100.
    let most = request(json!({
        "metrics": ["clicks"],
        "rolling": {"window": 98, "step": 3},
        "method": {"name": "equal_weight"}
    }));
    let windows = allocate(&most).unwrap().results[0]
        .rolling
        .as_ref()
        .unwrap()
        .len();
    assert_eq!(windows, MAX_ROLLING_WINDOWS);
}

#[test]
fn too_many_groups_are_refused_before_any_is_solved() {
    let sample = small();
    let group = json!({"members": ["A", "B"], "lower": 0.0, "upper": 1.0});
    let grouped = |mut method: serde_json::Value, groups: usize| {
        method["bounds"] = json!({"groups": vec![group.clone(); groups]});
        request(json!({"platforms": ["A", "B"], "metrics": ["clicks"], "method": method}))
    };
    // Every method that takes bounds takes groups.
    for method in [
        json!({"name": "min_variance"}),
        json!({"name": "mean_variance", "risk_aversion": 2.0}),
        json!({"name": "min_cvar"}),
        json!({"name": "min_mad"}),
        json!({"name": "min_cdar"}),
    ] {
        assert_eq!(
            allocate_over(&sample, &grouped(method.clone(), MAX_GROUPS + 1)),
            Err(EngineError::Groups {
                groups: MAX_GROUPS + 1
            }),
            "{method}"
        );
    }
    // Exactly the most is allowed, and so is a method with no bounds to count.
    let least = json!({"name": "min_variance"});
    let most = allocate_over(&sample, &grouped(least, MAX_GROUPS)).unwrap();
    assert!(most.results[0].weights.is_some(), "{most:?}");
    let none = request(json!({"method": {"name": "hrp"}}));
    assert_eq!(none.method.groups(), Some(0));
    // serde reads a struct from an array of its fields in order too, so bounds spelled
    // `[lower, upper, assets, groups]` with each group `[members, lower, upper]` have no
    // `groups` key in the request. They're counted all the same.
    let spelled = vec![json!([["A", "B"], 0.0, 1.0]); MAX_GROUPS + 1];
    let positional = request(json!({
        "platforms": ["A", "B"],
        "method": {"name": "min_variance", "bounds": [0.0, null, [], spelled]}
    }));
    assert_eq!(
        allocate_over(&sample, &positional),
        Err(EngineError::Groups {
            groups: MAX_GROUPS + 1
        })
    );
}

#[test]
fn groups_that_cant_be_counted_get_no_frontier() {
    // A risk aversion that isn't finite, which JSON can't carry, doesn't serialize, so the
    // method's groups aren't counted. Its allocation refuses the risk aversion, and its
    // frontier, which doesn't take it, would be traced with every one of them.
    let bounds = (0..=MAX_GROUPS).fold(WeightBounds::default(), |bounds, _| {
        bounds.group(["A", "B"], 0.0, 1.0)
    });
    let mut read = request(json!({
        "platforms": ["A", "B"],
        "metrics": ["clicks"],
        "method": {"name": "equal_weight"},
        "frontier": 4
    }));
    read.method = Method::MeanVariance(MeanVariance::risk_aversion(f64::INFINITY).bounds(bounds));
    assert_eq!(read.method.groups(), None);
    let result = &allocate_over(&small(), &read).unwrap().results[0];
    let refused = Some(CutupError::InvalidConfig {
        field: "risk_aversion",
        value: f64::INFINITY,
    });
    assert_eq!(result.error, refused);
    assert!(result.weights.is_none() && result.frontier.is_none());
    assert_eq!(result.frontier_error, refused);
}

#[test]
fn a_request_longer_than_the_engine_reads_is_refused_unread() {
    let request = r#"{"method": {"name": "equal_weight"}, "metrics": ["views"]}"#;
    // JSON allows whitespace after its value, so a request can be padded to any length.
    let padded = |bytes: usize| format!("{request}{}", " ".repeat(bytes - request.len()));
    assert_eq!(respond(&padded(MAX_REQUEST_BYTES)).status, 200);
    let long = respond(&padded(MAX_REQUEST_BYTES + 1));
    assert_eq!(long.status, 413);
    let body: serde_json::Value = serde_json::from_str(&long.body).unwrap();
    let message = body["error"]["request"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("at most {MAX_REQUEST_BYTES}")),
        "{message}"
    );
    // Unread: text that isn't JSON gets the same reply, where a shorter one is a 400.
    assert_eq!(respond(&"[".repeat(MAX_REQUEST_BYTES + 1)).status, 413);
    assert_eq!(respond(&"[".repeat(MAX_REQUEST_BYTES)).status, 400);
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
         rolling?: Array<RollingWindow>, };"
    );
    // The same JSON as cutup's own windows.
    assert_eq!(
        bare(RollingWindow::decl(&cfg)),
        "type RollingWindow = { start: number, end: number, weights?: Weights, \
         error?: CutupError, };"
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

#[test]
fn each_risk_method_has_the_least_of_its_risk_on_the_sample() {
    // The returns are each yield's excess over the average, so a run of days below it is a
    // drawdown: on yields over the average alone, every running sum rose and every CDaR was 0.
    let stats = |method: serde_json::Value| -> Stats {
        let read = request(json!({"metrics": ["clicks"], "method": method}));
        allocate(&read).unwrap().results[0].stats.clone().unwrap()
    };
    // The drawdowns include the drift, as on any P&L: a portfolio below the benchmark falls
    // further behind it each day. So every portfolio here has some.
    let cdar = stats(json!({"name": "min_cdar"}));
    let mad = stats(json!({"name": "min_mad"}));
    let shortfall = stats(json!({"name": "min_mad", "downside": 0.0}));
    assert!(cdar.cdar > 0.0, "{cdar:?}");
    for other in [
        json!({"name": "equal_weight"}),
        json!({"name": "min_variance"}),
        json!({"name": "min_cvar"}),
    ] {
        let other = stats(other);
        assert!(cdar.cdar <= other.cdar + 1e-12, "{cdar:?} vs {other:?}");
        assert!(mad.mad <= other.mad + 1e-12, "{mad:?} vs {other:?}");
        assert!(
            shortfall.downside_mad <= other.downside_mad + 1e-12,
            "{shortfall:?} vs {other:?}"
        );
    }
}

#[test]
fn rolling_windows_are_laid_out_as_cutup_lays_them_out() {
    let data = cutup::Returns::new(&vec![[0.01, -0.02]; 30]).unwrap();
    for (window, step) in [(2, 1), (5, 3), (7, 7), (30, 4), (11, 2)] {
        let rolling = data
            .rolling(window)
            .step(step)
            .allocate(cutup::EqualWeight)
            .unwrap();
        let count = rolling.len();
        let theirs: Vec<_> = rolling.map(|w| w.periods()).collect();
        let ours: Vec<_> = windows(30, RollingRequest { window, step }, count).collect();
        assert_eq!(ours, theirs, "window {window}, step {step}");
    }
}

#[test]
fn each_window_is_measured_from_its_own_average() {
    // Max Sharpe at a rate of 0 needs some platform above the benchmark. Measured from the
    // whole request's average, windows where every platform trailed it failed: 7 of the
    // impressions windows and 11 of the link clicks windows at the form's default windows.
    let read = request(json!({
        "metrics": ["impressions", "link_clicks"],
        "method": {"name": "max_sharpe"},
        "rolling": {"window": 60, "step": 7}
    }));
    let response = allocate(&read).unwrap();
    for result in &response.results {
        let windows = result.rolling.as_ref().unwrap();
        assert!(windows.len() > 40);
        for window in windows {
            assert!(window.error.is_none(), "{:?}: {window:?}", result.metric);
        }
    }
    // A window's weights are the method's on its own days' yields, over their own average.
    let platforms = default_platforms();
    let (_, totals): (Vec<&str>, Vec<Vec<Totals>>) = Sample::bundled()
        .days_with_spend(&platforms, None, None)
        .unzip();
    let all = yields(&totals, Metric::Impressions);
    let window = &response.results[0].rolling.as_ref().unwrap()[3];
    let own = excess(&all[window.start..window.end], &platforms).unwrap();
    let expected = own.allocate(&MaxSharpe::default()).unwrap();
    assert_eq!(window.weights.as_ref(), Some(&expected));
}
