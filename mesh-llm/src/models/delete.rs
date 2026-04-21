use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::models::usage::delete_model_by_path;

#[derive(Debug, Default)]
pub struct DeleteResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub removed_metadata_files: usize,
    pub removed_usage_records: usize,
}

pub async fn delete_model_at_path(identifier: &str, _dry_run: bool) -> Result<DeleteResult> {
    let is_path = identifier.contains('/') || identifier.contains('\\');
    let path: std::path::PathBuf = if is_path {
        std::path::PathBuf::from(identifier)
    } else {
        crate::models::find_model_path(identifier)
    };

    if !path.exists() {
        bail!("Model not found: {}", identifier);
    }
    if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        bail!("Not a GGUF file: {}", path.display());
    }

    let hf_root = crate::models::huggingface_hub_cache_dir();
    let mesh_root = crate::models::model_usage_cache_dir();
    if !path.starts_with(&hf_root) && !path.starts_with(&mesh_root) {
        bail!(
            "Deletion target outside known model roots: {}",
            path.display()
        );
    }

    let cleanup_result = delete_model_by_path(&path).context("delete model")?;

    Ok(DeleteResult {
        deleted_paths: vec![path],
        reclaimed_bytes: cleanup_result.reclaimed_bytes,
        removed_metadata_files: cleanup_result.removed_metadata_files,
        removed_usage_records: cleanup_result.removed_records,
    })
}
