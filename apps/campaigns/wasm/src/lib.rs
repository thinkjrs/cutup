//! The campaign engine for the browser: the same JSON in and out as the native functions, so
//! the frontend can send the same request to either. Built with the `wasm-release` profile and
//! `wasm-bindgen --target web`, as the app's README shows, and run in a Web Worker so a large
//! allocation doesn't block the page.
//!
//! The request and the reply's body are strings rather than JS objects, so that the JSON is
//! the engine's own, number for number, and the parity the engine's tests check on wasm holds
//! here too.

use wasm_bindgen::prelude::wasm_bindgen;

/// A reply, as `POST /api/allocate` sends it: its HTTP status, and its JSON body.
#[wasm_bindgen]
pub struct Reply(campaign_engine::Reply);

#[wasm_bindgen]
impl Reply {
    /// 200 for a response, 400 for a request that isn't JSON for one, and 422 for one that
    /// can't be allocated.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> u16 {
        self.0.status
    }

    /// A response, or `{"error": ..}`, as JSON.
    #[wasm_bindgen(getter)]
    pub fn body(&self) -> String {
        self.0.body.clone()
    }
}

/// The reply to a JSON request.
#[wasm_bindgen]
pub fn allocate(request: &str) -> Reply {
    Reply(campaign_engine::respond(request))
}

/// The bundled sample's platforms, as `GET /api/platforms` sends them.
#[wasm_bindgen]
pub fn platforms() -> String {
    campaign_engine::platforms()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn a_reply_is_the_engines() {
        let request = r#"{"method": {"name": "equal_weight"}, "metrics": ["clicks"]}"#;
        let reply = allocate(request);
        let engine = campaign_engine::respond(request);
        assert_eq!((reply.status(), reply.body()), (engine.status, engine.body));
        assert_eq!(allocate("{").status(), 400);
        assert_eq!(platforms(), campaign_engine::platforms());
    }
}
