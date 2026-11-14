//! `POST /api/allocate`: a JSON [`Request`](campaign_engine::Request) in, the engine's JSON
//! reply out, with its status: 200 for a response, 400 for a request that isn't JSON for one,
//! 422 for one that can't be allocated, and 405 for anything but `POST`.

use http_body_util::BodyExt;
use vercel_runtime::{Error, Request, Response, ResponseBody, run, service_fn};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(request: Request) -> Result<Response<ResponseBody>, Error> {
    let reply = if request.method() == "POST" {
        let body = request.into_body().collect().await?.to_bytes();
        campaign_engine::respond(&String::from_utf8_lossy(&body))
    } else {
        campaign_engine::not_allowed()
    };
    Ok(Response::builder()
        .status(reply.status)
        .header("content-type", "application/json")
        .body(ResponseBody::from(reply.body))?)
}
