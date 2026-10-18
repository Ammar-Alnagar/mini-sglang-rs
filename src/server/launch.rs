use crate::server::api_server;
use crate::server::args::Args;
use anyhow::Result;

pub async fn run(args: Args) -> Result<()> {
    if args.shell {
        println!("Interactive Shell Mode (Mini-SGLang Pure Rust)");
        println!("Model: {}", args.model);
        println!("Type '/reset' to clear chat history, or press Ctrl+C to exit.\n");
        println!("User: ");
        return Ok(());
    }

    api_server::start_server(&args.host, args.port).await
}
