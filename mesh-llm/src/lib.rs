mod affinity;
mod api;
mod autoupdate;
mod benchmark;
mod cli;
mod hardware;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugin_mcp;
mod plugins;
mod protocol;
mod rewrite;
pub(crate) mod runtime;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use autoupdate::{latest_release_version, version_newer};
pub(crate) use inference::{election, launch, moe, pipeline};
pub(crate) use network::{nostr, proxy, router, tunnel};
pub use plugins::blackboard;
pub use plugins::blackboard::mcp as blackboard_mcp;

use anyhow::Result;

pub const VERSION: &str = "0.54.0";

pub async fn run() -> Result<()> {
    runtime::run().await
}
