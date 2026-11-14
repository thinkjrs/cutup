//! `GET /api/platforms`: the bundled sample's platforms, each with its first and last dates and
//! how many days it spent on, as JSON. `HEAD` gets the same status and headers without the
//! body, as RFC 9110 asks of a server that answers `GET`, and anything else a 405.

use campaign_engine::Reply;
use vercel_runtime::{Error, Request, Response, ResponseBody, run, service_fn};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(request: Request) -> Result<Response<ResponseBody>, Error> {
    let response = Response::builder().header("content-type", "application/json");
    let method = request.method();
    let (response, reply) = if method == "GET" || method == "HEAD" {
        let body = if method == "GET" {
            campaign_engine::platforms()
        } else {
            String::new()
        };
        (response, Reply { status: 200, body })
    } else {
        (
            response.header("allow", "GET, HEAD"),
            campaign_engine::reject(405, "use GET"),
        )
    };
    Ok(response
        .status(reply.status)
        .body(ResponseBody::from(reply.body))?)
}
