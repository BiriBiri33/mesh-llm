use hf_hub::Cache;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Directories to scan for GGUF models.
pub fn model_dirs() -> Vec<PathBuf> {
    let canonical = huggingface_hub_cache_dir();
    let legacy = legacy_models_dir();
    let mut dirs = vec![canonical];
    if legacy.exists() {
        dirs.push(legacy);
    }
    dirs
}

fn hf_hub_cache_override() -> Option<PathBuf> {
    let path = std::env::var("HF_HUB_CACHE").ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

/// Build the effective Hugging Face cache handle.
///
/// `hf-hub` already resolves `HF_HOME` and the default cache location.
/// We only patch in `HF_HUB_CACHE` here because the crate does not honor it.
pub fn huggingface_hub_cache() -> Cache {
    if let Some(path) = hf_hub_cache_override() {
        Cache::new(path)
    } else {
        Cache::from_env()
    }
}

pub fn huggingface_hub_cache_dir() -> PathBuf {
    huggingface_hub_cache().path().clone()
}

pub fn legacy_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".models")
}

pub fn legacy_models_present() -> bool {
    let legacy_dir = legacy_models_dir();
    if !legacy_dir.exists() {
        return false;
    }
    tree_contains_gguf(&legacy_dir)
}

pub fn path_is_in_legacy_models_dir(path: &Path) -> bool {
    path.starts_with(legacy_models_dir())
}

fn push_model_name(
    path: &Path,
    names: &mut Vec<String>,
    seen: &mut HashSet<String>,
    min_size_bytes: u64,
) {
    if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return;
    }
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return;
    };
    if stem.contains("mmproj") {
        return;
    }
    let size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if size <= min_size_bytes {
        return;
    }
    let name = split_gguf_base_name(stem).unwrap_or(stem).to_string();
    if seen.insert(name.clone()) {
        names.push(name);
    }
}

fn tree_contains_gguf(root: &Path) -> bool {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if (file_type.is_file() || file_type.is_symlink())
                && path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
            {
                return true;
            }
        }
    }
    false
}

fn scan_hf_cache_models(
    root: &Path,
    names: &mut Vec<String>,
    seen: &mut HashSet<String>,
    min_size_bytes: u64,
) {
    // `hf-hub` does not currently expose cached repo enumeration, so local
    // discovery still walks the cache tree directly.
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                push_model_name(&path, names, seen, min_size_bytes);
                continue;
            }

            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !name.starts_with("models--") {
                continue;
            }
            let snapshots = path.join("snapshots");
            if !snapshots.is_dir() {
                continue;
            }
            if let Ok(snapshot_dirs) = std::fs::read_dir(&snapshots) {
                for snapshot in snapshot_dirs.flatten() {
                    scan_model_tree(&snapshot.path(), names, seen, min_size_bytes);
                }
            }
        }
    }
}

fn scan_model_tree(
    root: &Path,
    names: &mut Vec<String>,
    seen: &mut HashSet<String>,
    min_size_bytes: u64,
) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() || file_type.is_symlink() {
                push_model_name(&path, names, seen, min_size_bytes);
            }
        }
    }
}

fn scan_models_with_min_size(min_size_bytes: u64) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    let canonical_dir = huggingface_hub_cache_dir();
    if canonical_dir.exists() {
        scan_hf_cache_models(&canonical_dir, &mut names, &mut seen, min_size_bytes);
    }
    let legacy_dir = legacy_models_dir();
    if legacy_dir.exists() {
        scan_model_tree(&legacy_dir, &mut names, &mut seen, min_size_bytes);
    }
    names.sort();
    names
}

/// Scan model directories for GGUF files and return their stem names.
pub fn scan_local_models() -> Vec<String> {
    scan_models_with_min_size(500_000_000)
}

/// Scan installed GGUF models, including small draft models.
pub fn scan_installed_models() -> Vec<String> {
    scan_models_with_min_size(0)
}

fn find_hf_cache_model_path(root: &Path, stem: &str) -> Option<PathBuf> {
    let filename = format!("{stem}.gguf");
    let direct = root.join(&filename);
    if direct.exists() {
        return Some(direct);
    }

    let split_prefix = format!("{stem}-00001-of-");
    // `hf-hub` can resolve individual files in known repos, but it does not
    // expose a "find cached model by filename" API, so lookup stays local.
    let Ok(entries) = std::fs::read_dir(root) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            if name == filename {
                return Some(path);
            }
            if name.starts_with(&split_prefix) && name.ends_with(".gguf") {
                return Some(path);
            }
        }
        let Some(dir_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !dir_name.starts_with("models--") {
            continue;
        }
        let snapshots = path.join("snapshots");
        let Ok(snapshot_dirs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for snapshot in snapshot_dirs.flatten() {
            if let Some(found) = find_model_tree_path(&snapshot.path(), stem) {
                return Some(found);
            }
        }
    }
    None
}

fn find_model_tree_path(root: &Path, stem: &str) -> Option<PathBuf> {
    let filename = format!("{stem}.gguf");
    let split_prefix = format!("{stem}-00001-of-");
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name == filename || (name.starts_with(&split_prefix) && name.ends_with(".gguf")) {
                return Some(path);
            }
        }
    }
    None
}

/// Extract the base model name from a split GGUF stem.
/// "GLM-5-UD-IQ2_XXS-00001-of-00006" → Some("GLM-5-UD-IQ2_XXS")
/// "Qwen3-8B-Q4_K_M" → None (not a split file)
fn split_gguf_base_name(stem: &str) -> Option<&str> {
    let suffix = stem.rfind("-of-")?;
    let part_num = &stem[suffix + 4..];
    if part_num.len() != 5 || !part_num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let dash = stem[..suffix].rfind('-')?;
    let seq = &stem[dash + 1..suffix];
    if seq.len() != 5 || !seq.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&stem[..dash])
}

/// Find a GGUF model file by stem name, searching all model directories.
/// Returns the first match found (prefers the Hugging Face cache, then legacy ~/.models).
/// For split GGUFs, finds the first part (name-00001-of-NNNNN.gguf).
pub fn find_model_path(stem: &str) -> PathBuf {
    let filename = format!("{stem}.gguf");
    let canonical_dir = huggingface_hub_cache_dir();
    if let Some(found) = find_hf_cache_model_path(&canonical_dir, stem) {
        return found;
    }

    let legacy_dir = legacy_models_dir();
    if let Some(found) = find_model_tree_path(&legacy_dir, stem) {
        return found;
    }

    canonical_dir.join(&filename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn huggingface_cache_prefers_explicit_hub_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("HF_HUB_CACHE", "/tmp/mesh-llm-hub-cache");
        std::env::set_var("HF_HOME", "/tmp/mesh-llm-hf-home");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-xdg");

        assert_eq!(
            huggingface_hub_cache_dir(),
            PathBuf::from("/tmp/mesh-llm-hub-cache")
        );

        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    #[serial]
    fn huggingface_cache_falls_back_to_hf_home() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::remove_var("HF_HUB_CACHE");
        std::env::set_var("HF_HOME", "/tmp/mesh-llm-hf-home");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-xdg");

        assert_eq!(
            huggingface_hub_cache_dir(),
            PathBuf::from("/tmp/mesh-llm-hf-home").join("hub")
        );

        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn legacy_tree_detection_finds_nested_gguf_files() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-legacy-detect-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = temp.join("nested").join("models");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("Qwen3-8B-Q4_K_M.gguf"), b"gguf").unwrap();

        assert!(tree_contains_gguf(&temp));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_split_gguf_base_name() {
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00001-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00006-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(split_gguf_base_name("Qwen3-8B-Q4_K_M"), None);
        assert_eq!(split_gguf_base_name("model-001-of-003"), None);
        assert_eq!(split_gguf_base_name("model-00001-of-00003"), Some("model"));
    }

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }
}
