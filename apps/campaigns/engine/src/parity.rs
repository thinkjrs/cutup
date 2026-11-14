//! The native and wasm backends' parity: every request in `fixtures/requests.json` gets the
//! reply committed in `fixtures/replies.json`, natively and on wasm, so the two agree.
//!
//! Basic arithmetic and `sqrt` round the same on both, but `ln`, which risk parity uses, comes
//! from a different math library on wasm and may differ in the last bit. So numbers are
//! compared to 1e-14 of their size rather than exactly.
//!
//! `UPDATE_FIXTURES=1 cargo test` rewrites the replies from a native run.

use crate::respond;
use serde_json::{Map, Value};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// Each fixture's request: a JSON value, or a string to send as it is, such as JSON that
/// doesn't parse.
const REQUESTS: &str = include_str!("../../fixtures/requests.json");

/// Each fixture's reply: its status, and its body as JSON.
const REPLIES: &str = include_str!("../../fixtures/replies.json");

/// How close two numbers must be, relative to the larger in size.
const TOLERANCE: f64 = 1e-14;

/// The text to send for a fixture's request.
fn text(request: &Value) -> String {
    match request {
        Value::String(raw) => raw.clone(),
        json => json.to_string(),
    }
}

/// Each fixture's reply now, as `{"status": .., "body": ..}`.
fn replies(requests: &Map<String, Value>) -> Map<String, Value> {
    requests
        .iter()
        .map(|(name, request)| {
            let reply = respond(&text(request));
            let body: Value = serde_json::from_str(&reply.body).expect("a reply is JSON");
            let reply = serde_json::json!({"status": reply.status, "body": body});
            (name.clone(), reply)
        })
        .collect()
}

/// Where `got` differs from `want`, by path, beyond [`TOLERANCE`] for numbers.
fn differences(path: &str, got: &Value, want: &Value, out: &mut Vec<String>) {
    match (got, want) {
        (Value::Number(g), Value::Number(w)) => {
            let (g, w) = (g.as_f64().unwrap(), w.as_f64().unwrap());
            if (g - w).abs() > TOLERANCE * g.abs().max(w.abs()) {
                out.push(format!("{path}: {g:e} vs {w:e}"));
            }
        }
        (Value::Array(g), Value::Array(w)) if g.len() == w.len() => {
            for (i, (g, w)) in g.iter().zip(w).enumerate() {
                differences(&format!("{path}[{i}]"), g, w, out);
            }
        }
        // serde_json's maps are sorted, so the same keys come in the same order.
        (Value::Object(g), Value::Object(w)) if g.keys().eq(w.keys()) => {
            for (key, g) in g {
                differences(&format!("{path}.{key}"), g, &w[key], out);
            }
        }
        _ if got == want => {}
        _ => out.push(format!("{path}: {got} vs {want}")),
    }
}

#[test]
fn every_fixture_gets_its_committed_reply() {
    let requests: Map<String, Value> = serde_json::from_str(REQUESTS).unwrap();
    let got = replies(&requests);
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("UPDATE_FIXTURES").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures/replies.json");
        let text = serde_json::to_string_pretty(&got).unwrap() + "\n";
        std::fs::write(path, text).unwrap();
        return;
    }
    let want: Map<String, Value> = serde_json::from_str(REPLIES).unwrap();
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "every request has a committed reply; rerun with UPDATE_FIXTURES=1"
    );
    let mut out = Vec::new();
    for (name, reply) in &got {
        differences(name, reply, &want[name], &mut out);
    }
    assert!(
        out.is_empty(),
        "{} differences:\n{}",
        out.len(),
        out.join("\n")
    );
}

#[test]
fn numbers_differ_only_beyond_the_tolerance() {
    let mut out = Vec::new();
    let one = serde_json::json!(1.0);
    differences("a", &one, &serde_json::json!(1.0 + 1e-15), &mut out);
    assert!(out.is_empty());
    differences("a", &one, &serde_json::json!(1.0 + 1e-13), &mut out);
    differences(
        "b",
        &serde_json::json!({"x": 1}),
        &serde_json::json!({"y": 1}),
        &mut out,
    );
    differences(
        "c",
        &serde_json::json!([1, 2]),
        &serde_json::json!([1]),
        &mut out,
    );
    assert_eq!(out.len(), 3, "{out:?}");
}
