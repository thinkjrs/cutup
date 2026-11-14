//! `GET /api/platforms`: the bundled sample's platforms, each with its first and last dates and
//! how many days it spent on, as JSON.

use vercel_runtime::{Error, Request, Response, ResponseBody, run, service_fn};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(_request: Request) -> Result<Response<ResponseBody>, Error> {
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(ResponseBody::from(campaign_engine::platforms()))?)
}
