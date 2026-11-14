//! `GET /api/platforms`: the bundled sample's platforms, each with its first and last dates and
//! how many days it spent on, as JSON, and a 405 for anything but `GET`.

use vercel_runtime::{Error, Request, Response, ResponseBody, run, service_fn};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(request: Request) -> Result<Response<ResponseBody>, Error> {
    let response = Response::builder().header("content-type", "application/json");
    let (response, status, body) = if request.method() == "GET" {
        (response, 200, campaign_engine::platforms())
    } else {
        let reply = campaign_engine::reject(405, "use GET");
        (response.header("allow", "GET"), reply.status, reply.body)
    };
    Ok(response.status(status).body(ResponseBody::from(body))?)
}
