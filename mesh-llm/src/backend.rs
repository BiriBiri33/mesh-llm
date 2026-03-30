use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Llama,
    Mlx,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub requires_rpc_server: bool,
    pub supports_rpc_split: bool,
}

impl BackendKind {
    pub const ALL: [BackendKind; 2] = [BackendKind::Llama, BackendKind::Mlx];

    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Llama => "llama",
            BackendKind::Mlx => "mlx",
        }
    }

    pub fn process_label(self) -> &'static str {
        match self {
            BackendKind::Llama => "llama-server",
            BackendKind::Mlx => "mlx_lm.server",
        }
    }

    pub fn capabilities(self) -> BackendCapabilities {
        match self {
            BackendKind::Llama => BackendCapabilities {
                requires_rpc_server: true,
                supports_rpc_split: true,
            },
            BackendKind::Mlx => BackendCapabilities {
                requires_rpc_server: false,
                supports_rpc_split: false,
            },
        }
    }
}

pub fn detect_backend(model_path: &Path) -> BackendKind {
    if looks_like_mlx_model_dir(model_path) {
        return BackendKind::Mlx;
    }

    BackendKind::Llama
}

fn looks_like_mlx_model_dir(model_path: &Path) -> bool {
    if !model_path.is_dir() {
        return false;
    }

    let has_config = model_path.join("config.json").exists();
    let has_tokenizer = model_path.join("tokenizer_config.json").exists()
        || model_path.join("tokenizer.json").exists();
    let has_weights = model_path.join("model.safetensors").exists()
        || model_path.join("model.safetensors.index.json").exists();

    has_config && has_tokenizer && has_weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{unique}"))
    }

    #[test]
    fn detect_backend_uses_mlx_for_mlx_model_dir() {
        let dir = temp_dir("mesh-llm-mlx-model");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();

        assert_eq!(detect_backend(&dir), BackendKind::Mlx);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn detect_backend_falls_back_to_llama_for_non_mlx_dir() {
        let dir = temp_dir("mesh-llm-not-mlx");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();

        assert_eq!(detect_backend(&dir), BackendKind::Llama);

        let _ = std::fs::remove_dir_all(dir);
    }
}
