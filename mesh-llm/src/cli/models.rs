use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List built-in recommended models.
    Recommended,
    /// List installed local models from the HF cache or legacy storage.
    Installed,
    /// List built-in catalog models.
    #[command(hide = true)]
    List,
    /// Search for GGUF models in the catalog or on Hugging Face.
    Search {
        /// Search terms.
        #[arg(required = true)]
        query: Vec<String>,
        /// Search only the built-in catalog.
        #[arg(long)]
        catalog: bool,
        /// Maximum number of results to show.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show details for one exact model reference.
    Show {
        /// Exact catalog id, Hugging Face ref, or direct URL.
        model: String,
    },
    /// Download one exact model reference.
    Download {
        /// Exact catalog id, Hugging Face ref, or direct URL.
        model: String,
        /// Also download the recommended draft model for speculative decoding.
        #[arg(long)]
        draft: bool,
    },
    /// Inspect or migrate deprecated ~/.models content into the Hugging Face cache.
    Migrate {
        /// Materialize recognized Hugging Face-backed legacy models into the HF cache.
        #[arg(long)]
        apply: bool,
        /// Remove recognized legacy GGUF files that already exist in the Hugging Face cache.
        #[arg(long)]
        prune: bool,
    },
    /// Check or refresh cached Hugging Face repos.
    #[command(visible_alias = "update")]
    Updates {
        /// Repo id like Qwen/Qwen3-8B-GGUF.
        repo: Option<String>,
        /// Operate on every cached Hugging Face repo.
        #[arg(long)]
        all: bool,
        /// Check for newer upstream revisions without refreshing local cache.
        #[arg(long)]
        check: bool,
    },
}

pub async fn dispatch_models_command(command: &ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Recommended | ModelsCommand::List => crate::models::run_model_recommended(),
        ModelsCommand::Installed => crate::models::run_model_installed(),
        ModelsCommand::Search {
            query,
            catalog,
            limit,
        } => crate::models::run_model_search(query, *catalog, *limit).await?,
        ModelsCommand::Show { model } => crate::models::run_model_show(model).await?,
        ModelsCommand::Download { model, draft } => {
            crate::models::run_model_download(model, *draft).await?
        }
        ModelsCommand::Migrate { apply, prune } => crate::models::run_migrate(*apply, *prune)?,
        ModelsCommand::Updates { repo, all, check } => {
            crate::models::run_update(repo.as_deref(), *all, *check)?
        }
    }
    Ok(())
}
