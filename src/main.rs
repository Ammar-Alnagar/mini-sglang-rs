use anyhow::Result;
use clap::Parser;
use minisgl::server::args::Args;
use minisgl::server::launch;

#[tokio::main]
async fn main() -> Result<()> {
    minisgl::utils::logger::init_logger();
    let args = Args::parse();
    launch::run(args).await
}
