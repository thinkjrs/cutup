//! `POST /api/allocate`: a JSON [`Request`](campaign_engine::Request) in, the engine's JSON
//! reply out, with its status: 200 for a response, 400 for a body that isn't UTF-8 or a
//! request that isn't JSON for one, 422 for one that can't be allocated, and 405 for anything
//! but `POST`.

use http_body_util::BodyExt;
use vercel_runtime::{Error, Request, Response, ResponseBody, run, service_fn};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(request: Request) -> Result<Response<ResponseBody>, Error> {
    let response = Response::builder().header("content-type", "application/json");
    let (response, reply) = if request.method() == "POST" {
        let body = request.into_body().collect().await?.to_bytes();
        let reply = match String::from_utf8(body.into()) {
            // An allocation is CPU-bound, so it runs off the runtime's async workers.
            Ok(text) => {
                tokio::task::spawn_blocking(move || campaign_engine::respond(&text)).await?
            }
            Err(_) => campaign_engine::reject(400, "the body isn't UTF-8"),
        };
        (response, reply)
    } else {
        (
            response.header("allow", "POST"),
            campaign_engine::reject(405, "use POST, with a JSON request"),
        )
    };
    Ok(response
        .status(reply.status)
        .body(ResponseBody::from(reply.body))?)
}
