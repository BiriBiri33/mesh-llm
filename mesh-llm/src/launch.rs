//! Process management for inference backends.
//!
//! Starts rpc-server and backend inference processes as child processes,
//! wired up to the mesh tunnel ports.

use crate::backend;
use anyhow::{Context, Result};
use clap::ValueEnum;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokio::net::TcpListener;
use tokio::process::Command;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }

    fn preferred_devices(self) -> &'static [&'static str] {
        match self {
            BinaryFlavor::Cpu => &["CPU"],
            BinaryFlavor::Cuda => &["CUDA0", "CPU"],
            BinaryFlavor::Rocm => &["HIP0", "CPU"],
            BinaryFlavor::Vulkan => &["Vulkan0", "CPU"],
            BinaryFlavor::Metal => &["MTL0", "CPU"],
        }
    }

    fn primary_device(self) -> &'static str {
        self.preferred_devices()[0]
    }
}

#[derive(Clone, Debug)]
struct ResolvedBinary {
    path: PathBuf,
    flavor: Option<BinaryFlavor>,
}

pub(crate) fn platform_bin_name(name: &str) -> String {
    #[cfg(windows)]
    {
        if Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            name.to_string()
        } else {
            format!("{name}.exe")
        }
    }

    #[cfg(not(windows))]
    {
        name.to_string()
    }
}

fn flavored_bin_name(name: &str, flavor: BinaryFlavor) -> String {
    platform_bin_name(&format!("{name}-{}", flavor.suffix()))
}

fn bare_bin_name(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    #[cfg(windows)]
    {
        // On Windows, strip a `.exe` extension in a case-insensitive way.
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            Some(path.file_stem()?.to_string_lossy().to_string())
        } else {
            Some(file_name.to_string())
        }
    }

    #[cfg(not(windows))]
    {
        Some(file_name.to_string())
    }
}

fn infer_binary_flavor(name: &str, path: &Path) -> Option<BinaryFlavor> {
    let file_name = bare_bin_name(path)?;
    for flavor in BinaryFlavor::ALL {
        if file_name == format!("{name}-{}", flavor.suffix()) {
            return Some(flavor);
        }
    }
    None
}

fn resolve_binary_path(
    bin_dir: &Path,
    name: &str,
    requested_flavor: Option<BinaryFlavor>,
) -> Result<ResolvedBinary> {
    if let Some(flavor) = requested_flavor {
        let flavored = bin_dir.join(flavored_bin_name(name, flavor));
        if flavored.exists() {
            return Ok(ResolvedBinary {
                path: flavored,
                flavor: Some(flavor),
            });
        }

        let generic = bin_dir.join(platform_bin_name(name));
        if generic.exists() {
            return Ok(ResolvedBinary {
                path: generic,
                flavor: Some(flavor),
            });
        }

        anyhow::bail!(
            "{} not found in {} for requested flavor '{}'",
            flavored.display(),
            bin_dir.display(),
            flavor.suffix()
        );
    }

    let generic = bin_dir.join(platform_bin_name(name));
    if generic.exists() {
        let flavor = infer_binary_flavor(name, &generic);
        return Ok(ResolvedBinary {
            path: generic,
            flavor,
        });
    }

    let matches: Vec<ResolvedBinary> = BinaryFlavor::ALL
        .into_iter()
        .map(|flavor| ResolvedBinary {
            path: bin_dir.join(flavored_bin_name(name, flavor)),
            flavor: Some(flavor),
        })
        .filter(|candidate| candidate.path.exists())
        .collect();

    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => anyhow::bail!(
            "{} not found in {}",
            bin_dir.join(platform_bin_name(name)).display(),
            bin_dir.display()
        ),
        _ => {
            let options = matches
                .iter()
                .filter_map(|candidate| candidate.flavor.map(|flavor| flavor.suffix()))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "multiple {} flavors found in {} ({options}). Pass --llama-flavor to choose one.",
                name,
                bin_dir.display()
            );
        }
    }
}

pub struct ModelLaunchSpec<'a> {
    pub backend: backend::BackendKind,
    pub model: &'a Path,
    pub http_port: u16,
    pub tunnel_ports: &'a [u16],
    pub tensor_split: Option<&'a str>,
    pub draft: Option<&'a Path>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub my_vram: u64,
    pub mmproj: Option<&'a Path>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram: Option<u64>,
    pub mlx_hostfile_json: Option<&'a str>,
    pub mlx_rank: Option<usize>,
    pub mlx_bind_addrs: &'a [String],
    pub mlx_mode: Option<MlxDistributedMode>,
    pub mlx_serves_http: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxDistributedMode {
    Tensor,
    Pipeline,
}

impl MlxDistributedMode {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            MlxDistributedMode::Tensor => "tensor",
            MlxDistributedMode::Pipeline => "pipeline",
        }
    }

    #[allow(dead_code)]
    fn extra_server_args(self) -> &'static [&'static str] {
        match self {
            MlxDistributedMode::Tensor => &[],
            MlxDistributedMode::Pipeline => &["--pipeline"],
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxQuicShimRoute {
    pub local_listen_addr: String,
    pub remote_rank: usize,
    pub remote_target_addr: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxDistributedRankPlan {
    pub rank: usize,
    pub world_size: usize,
    pub bind_addrs: Vec<String>,
    pub hostfile_json: String,
    pub shim_routes: Vec<MlxQuicShimRoute>,
    pub serves_http: bool,
}

#[allow(dead_code)]
impl MlxDistributedRankPlan {
    pub fn env(&self, hostfile_path: &Path) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("MLX_RANK".to_string(), self.rank.to_string()),
            (
                "MLX_HOSTFILE".to_string(),
                hostfile_path.to_string_lossy().to_string(),
            ),
        ])
    }

    pub fn server_args(
        &self,
        model: &Path,
        http_host: &str,
        http_port: u16,
        mode: MlxDistributedMode,
    ) -> Vec<String> {
        let mut args = vec![
            "--model".to_string(),
            model.to_string_lossy().to_string(),
            "--host".to_string(),
            http_host.to_string(),
            "--port".to_string(),
            http_port.to_string(),
        ];
        args.extend(mode.extra_server_args().iter().map(|arg| arg.to_string()));
        args
    }
}

#[allow(dead_code)]
fn validate_rank_endpoint_matrix(rank_endpoints: &[Vec<String>]) -> Result<usize> {
    anyhow::ensure!(
        !rank_endpoints.is_empty(),
        "mlx distributed launch requires at least one rank"
    );

    let connections_per_rank = rank_endpoints[0].len();
    anyhow::ensure!(
        connections_per_rank > 0,
        "mlx distributed launch requires at least one address per rank"
    );

    let mut seen = HashSet::new();
    for (rank, addrs) in rank_endpoints.iter().enumerate() {
        anyhow::ensure!(
            addrs.len() == connections_per_rank,
            "rank {rank} has {} addresses, expected {connections_per_rank}",
            addrs.len()
        );
        for addr in addrs {
            let _: std::net::SocketAddr = addr
                .parse()
                .with_context(|| format!("invalid mlx distributed address '{addr}'"))?;
            anyhow::ensure!(
                seen.insert(addr.clone()),
                "duplicate mlx distributed address '{addr}'"
            );
        }
    }

    Ok(connections_per_rank)
}

#[allow(dead_code)]
pub fn allocate_mlx_loopback_bind_ports(
    world_size: usize,
    connections_per_rank: usize,
    starting_port: u16,
) -> Result<Vec<Vec<u16>>> {
    anyhow::ensure!(world_size > 0, "world_size must be > 0");
    anyhow::ensure!(connections_per_rank > 0, "connections_per_rank must be > 0");

    let mut next_port = starting_port;
    let mut ranks = Vec::with_capacity(world_size);
    for _ in 0..world_size {
        let mut ports = Vec::with_capacity(connections_per_rank);
        for _ in 0..connections_per_rank {
            anyhow::ensure!(next_port != 0, "port allocation overflow");
            ports.push(next_port);
            next_port = next_port
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("port allocation overflow"))?;
        }
        ranks.push(ports);
    }
    Ok(ranks)
}

#[allow(dead_code)]
pub fn build_mlx_direct_ring_plans(
    rank_endpoints: &[Vec<String>],
) -> Result<Vec<MlxDistributedRankPlan>> {
    validate_rank_endpoint_matrix(rank_endpoints)?;
    let world_size = rank_endpoints.len();
    let hostfile_json =
        serde_json::to_string(rank_endpoints).context("failed to encode MLX hostfile JSON")?;

    Ok(rank_endpoints
        .iter()
        .enumerate()
        .map(|(rank, addrs)| MlxDistributedRankPlan {
            rank,
            world_size,
            bind_addrs: addrs.clone(),
            hostfile_json: hostfile_json.clone(),
            shim_routes: vec![],
            serves_http: rank == 0,
        })
        .collect())
}

#[allow(dead_code)]
pub fn build_mlx_quic_ring_plans(
    bind_ports: &[Vec<u16>],
    shim_start_port: u16,
) -> Result<Vec<MlxDistributedRankPlan>> {
    anyhow::ensure!(!bind_ports.is_empty(), "bind_ports must not be empty");
    let connections_per_rank = bind_ports[0].len();
    anyhow::ensure!(
        connections_per_rank > 0,
        "bind_ports must contain at least one port per rank"
    );
    for (rank, ports) in bind_ports.iter().enumerate() {
        anyhow::ensure!(
            ports.len() == connections_per_rank,
            "rank {rank} has {} bind ports, expected {connections_per_rank}",
            ports.len()
        );
    }

    let bind_addrs: Vec<Vec<String>> = bind_ports
        .iter()
        .map(|ports| {
            ports
                .iter()
                .map(|port| format!("127.0.0.1:{port}"))
                .collect::<Vec<_>>()
        })
        .collect();
    validate_rank_endpoint_matrix(&bind_addrs)?;

    let mut used_ports: HashSet<u16> = bind_ports.iter().flatten().copied().collect();
    let world_size = bind_ports.len();
    let mut next_shim_port = shim_start_port;
    let mut shim_addrs: Vec<Vec<Vec<String>>> = vec![vec![vec![]; world_size]; world_size];
    let mut shim_routes: Vec<Vec<MlxQuicShimRoute>> = vec![vec![]; world_size];

    for rank in 0..world_size {
        for remote_rank in 0..world_size {
            if remote_rank == rank {
                continue;
            }
            for connection_idx in 0..connections_per_rank {
                while used_ports.contains(&next_shim_port) {
                    next_shim_port = next_shim_port
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("shim port allocation overflow"))?;
                }
                let local_listen_addr = format!("127.0.0.1:{next_shim_port}");
                used_ports.insert(next_shim_port);
                shim_addrs[rank][remote_rank].push(local_listen_addr.clone());
                shim_routes[rank].push(MlxQuicShimRoute {
                    local_listen_addr,
                    remote_rank,
                    remote_target_addr: bind_addrs[remote_rank][connection_idx].clone(),
                });
                next_shim_port = next_shim_port
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("shim port allocation overflow"))?;
            }
        }
    }

    let mut plans = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let hostfile_entries: Vec<Vec<String>> = (0..world_size)
            .map(|target_rank| {
                if target_rank == rank {
                    bind_addrs[target_rank].clone()
                } else {
                    shim_addrs[rank][target_rank].clone()
                }
            })
            .collect();

        let hostfile_json = serde_json::to_string(&hostfile_entries)
            .context("failed to encode MLX QUIC hostfile JSON")?;

        plans.push(MlxDistributedRankPlan {
            rank,
            world_size,
            bind_addrs: bind_addrs[rank].clone(),
            hostfile_json,
            shim_routes: shim_routes[rank].clone(),
            serves_http: rank == 0,
        });
    }

    Ok(plans)
}

fn mlx_model_overrides() -> &'static Mutex<HashMap<u16, String>> {
    static OVERRIDES: OnceLock<Mutex<HashMap<u16, String>>> = OnceLock::new();
    OVERRIDES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mlx_hostfiles() -> &'static Mutex<Vec<PathBuf>> {
    static HOSTFILES: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    HOSTFILES.get_or_init(|| Mutex::new(Vec::new()))
}

fn register_mlx_model_override(port: u16, model: &Path) {
    if let Ok(mut overrides) = mlx_model_overrides().lock() {
        overrides.insert(port, model.to_string_lossy().to_string());
    }
}

fn unregister_mlx_model_override(port: u16) {
    if let Ok(mut overrides) = mlx_model_overrides().lock() {
        overrides.remove(&port);
    }
}

pub fn mlx_model_override_for_port(port: u16) -> Option<String> {
    mlx_model_overrides()
        .lock()
        .ok()
        .and_then(|overrides| overrides.get(&port).cloned())
}

pub(crate) fn temp_log_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn log_tail(path: &Path, max_lines: usize) -> String {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return String::new();
    };

    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn parse_available_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_devices = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed == "available devices:" {
            in_devices = true;
            continue;
        }
        if !in_devices || trimmed.is_empty() {
            continue;
        }
        let Some((name, _rest)) = trimmed.split_once(':') else {
            continue;
        };
        if !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }
        devices.push(name.to_string());
    }

    devices
}

fn probe_available_devices(binary: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new(binary)
        .args(["-d", "__mesh_llm_probe_invalid__", "-p", "0"])
        .output()
    else {
        return Vec::new();
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !combined.is_empty() && !output.stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_available_devices(&combined)
}

fn preferred_device(available: &[String], flavor: Option<BinaryFlavor>) -> Option<String> {
    let candidates: &[&str] = if let Some(flavor) = flavor {
        flavor.preferred_devices()
    } else {
        &["MTL0", "CUDA0", "HIP0", "Vulkan0", "CPU"]
    };

    for candidate in candidates {
        if available.iter().any(|device| device == candidate) {
            return Some((*candidate).to_string());
        }
    }
    available.first().cloned()
}

fn resolve_device_for_binary(
    binary: &Path,
    flavor: Option<BinaryFlavor>,
    requested: Option<&str>,
) -> Result<String> {
    let available = probe_available_devices(binary);

    if let Some(device) = requested {
        if !available.is_empty() && !available.iter().any(|candidate| candidate == device) {
            anyhow::bail!(
                "requested device {device} is not supported by {}. Available devices: {}",
                binary.display(),
                available.join(", ")
            );
        }
        return Ok(device.to_string());
    }

    if let Some(selected) = preferred_device(&available, flavor) {
        return Ok(selected);
    }

    if let Some(flavor) = flavor {
        return Ok(flavor.primary_device().to_string());
    }

    Ok(detect_device())
}

fn command_has_output(command: &str, args: &[&str]) -> bool {
    let Ok(output) = std::process::Command::new(command).args(args).output() else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| !line.trim().is_empty())
}

fn tcp_addr_is_open(addr: &str) -> bool {
    let Ok(addr) = addr.parse::<std::net::SocketAddr>() else {
        return false;
    };
    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(200)).is_ok()
}

pub async fn start_model_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    spec: ModelLaunchSpec<'_>,
) -> Result<tokio::sync::oneshot::Receiver<()>> {
    match spec.backend {
        backend::BackendKind::Llama => {
            start_llama_server(
                bin_dir,
                binary_flavor,
                spec.model,
                spec.http_port,
                spec.tunnel_ports,
                spec.tensor_split,
                spec.draft,
                spec.draft_max,
                spec.model_bytes,
                spec.my_vram,
                spec.mmproj,
                spec.ctx_size_override,
                spec.total_group_vram,
            )
            .await
        }
        backend::BackendKind::Mlx => start_mlx_server(spec).await,
    }
}

pub async fn kill_server_processes(kind: backend::BackendKind) {
    match kind {
        backend::BackendKind::Llama => kill_llama_server().await,
        backend::BackendKind::Mlx => kill_mlx_server().await,
    }
}

pub async fn kill_all_server_processes() {
    for kind in backend::BackendKind::ALL {
        kill_server_processes(kind).await;
    }
}

/// Start a local rpc-server and return the port it's listening on.
/// Picks an available port automatically.
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    device: Option<&str>,
    gguf_path: Option<&Path>,
) -> Result<u16> {
    let rpc_server = resolve_binary_path(bin_dir, "rpc-server", binary_flavor)?;

    // Find a free port
    let port = find_free_port().await?;

    let device = resolve_device_for_binary(&rpc_server.path, rpc_server.flavor, device)?;
    let startup_timeout = if device.starts_with("Vulkan") {
        std::time::Duration::from_secs(90)
    } else {
        std::time::Duration::from_secs(15)
    };
    let startup_polls = (startup_timeout.as_millis() / 500) as usize;

    tracing::info!("Starting rpc-server on :{port} (device: {device})");

    let rpc_log = temp_log_path(&format!("mesh-llm-rpc-{port}.log"));
    let rpc_log_file = std::fs::File::create(&rpc_log)
        .with_context(|| format!("Failed to create rpc-server log file {}", rpc_log.display()))?;
    let rpc_log_file2 = rpc_log_file.try_clone()?;

    let mut args = vec![
        "-d".to_string(),
        device.clone(),
        "-p".to_string(),
        port.to_string(),
    ];
    if let Some(path) = gguf_path {
        args.push("--gguf".to_string());
        args.push(path.to_string_lossy().to_string());
        tracing::info!(
            "rpc-server will load weights from local GGUF: {}",
            path.display()
        );
    }

    let mut child = Command::new(&rpc_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start rpc-server at {}",
                rpc_server.path.display()
            )
        })?;

    // Wait for it to be listening
    for _ in 0..startup_polls {
        if is_port_open(port).await {
            // Detach — let it run in the background
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(port);
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll rpc-server status for {}",
                rpc_server.path.display()
            )
        })? {
            let tail = log_tail(&rpc_log, 40);
            let tail_msg = if tail.is_empty() {
                format!("See {}", rpc_log.display())
            } else {
                format!("See {}:\n{}", rpc_log.display(), tail)
            };
            anyhow::bail!(
                "rpc-server exited before listening on port {port} (device: {device}, status: {status}). {tail_msg}"
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let tail = log_tail(&rpc_log, 40);
    let tail_msg = if tail.is_empty() {
        format!("See {}", rpc_log.display())
    } else {
        format!("See {}:\n{}", rpc_log.display(), tail)
    };
    anyhow::bail!(
        "rpc-server failed to start on port {port} within {}s (device: {device}). {tail_msg}",
        startup_timeout.as_secs()
    );
}

/// Kill orphan rpc-server processes from previous mesh-llm runs.
/// Only kills rpc-servers with PPID 1 (parent died, adopted by init).
/// Safe to call while a live mesh-llm has its own rpc-server child.
pub async fn kill_orphan_rpc_servers() {
    #[cfg(windows)]
    {
        return;
    }

    #[cfg(not(windows))]
    if let Ok(output) = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,comm"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut killed = 0;
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[2].contains("rpc-server") && parts[1] == "1" {
                if let Ok(pid) = parts[0].parse::<u32>() {
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .status();
                    killed += 1;
                }
            }
        }
        if killed > 0 {
            eprintln!("🧹 Cleaned up {killed} orphan rpc-server process(es)");
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }
}

#[derive(Clone, Debug)]
struct ResolvedMlxRuntime {
    executable: String,
    prefix_args: Vec<String>,
}

fn command_on_path(command: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path).any(|dir| {
                let candidate = dir.join(command);
                candidate.exists() && candidate.is_file()
            })
        })
        .unwrap_or(false)
}

fn python_has_module(command: &str, module: &str) -> bool {
    std::process::Command::new(command)
        .args([
            "-c",
            &format!(
                "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module:?}) else 1)"
            ),
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn resolve_mlx_runtime() -> Result<ResolvedMlxRuntime> {
    if let Some(path) = std::env::var_os("MESH_LLM_MLX_SERVER_BIN") {
        let path = PathBuf::from(path);
        anyhow::ensure!(
            path.exists(),
            "MESH_LLM_MLX_SERVER_BIN points to missing path {}",
            path.display()
        );
        return Ok(ResolvedMlxRuntime {
            executable: path.to_string_lossy().to_string(),
            prefix_args: vec![],
        });
    }

    if command_on_path("mlx_lm.server") {
        return Ok(ResolvedMlxRuntime {
            executable: "mlx_lm.server".to_string(),
            prefix_args: vec![],
        });
    }

    for python in ["python3", "python"] {
        if command_on_path(python) && python_has_module(python, "mlx_lm.server") {
            return Ok(ResolvedMlxRuntime {
                executable: python.to_string(),
                prefix_args: vec!["-m".to_string(), "mlx_lm.server".to_string()],
            });
        }
    }

    anyhow::bail!("MLX server runtime not found. Install mlx-lm or set MESH_LLM_MLX_SERVER_BIN");
}

fn validate_mlx_model(model: &Path) -> Result<()> {
    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());
    anyhow::ensure!(
        model.is_dir(),
        "mlx backend expects a local model directory, got {}",
        model.display()
    );
    anyhow::ensure!(
        model.join("config.json").exists(),
        "mlx backend requires config.json in {}",
        model.display()
    );
    anyhow::ensure!(
        model.join("tokenizer_config.json").exists() || model.join("tokenizer.json").exists(),
        "mlx backend requires tokenizer metadata in {}",
        model.display()
    );
    anyhow::ensure!(
        model.join("model.safetensors").exists()
            || model.join("model.safetensors.index.json").exists(),
        "mlx backend requires model.safetensors or model.safetensors.index.json in {}",
        model.display()
    );
    Ok(())
}

async fn kill_mlx_server() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "mlx_lm.server"])
        .status();
    if let Ok(mut hostfiles) = mlx_hostfiles().lock() {
        for path in hostfiles.drain(..) {
            let _ = std::fs::remove_file(path);
        }
    }
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
}

/// Kill all running llama-server processes.
pub async fn kill_llama_server() {
    let _ = terminate_process_by_name("llama-server");
    // Wait for the process to actually exit and release the port
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        if !is_process_running("llama-server") {
            return;
        }
    }
    // Force kill if still alive after 5s
    let _ = force_kill_process_by_name("llama-server");
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

async fn start_mlx_server(spec: ModelLaunchSpec<'_>) -> Result<tokio::sync::oneshot::Receiver<()>> {
    let distributed = spec.mlx_hostfile_json.is_some();
    if !distributed {
        anyhow::ensure!(
            spec.tunnel_ports.is_empty(),
            "mlx backend does not support rpc split workers yet"
        );
        anyhow::ensure!(
            spec.tensor_split.is_none(),
            "mlx backend does not support tensor split yet"
        );
    }
    anyhow::ensure!(
        spec.draft.is_none(),
        "mlx backend does not support draft/speculative mode yet"
    );
    anyhow::ensure!(
        spec.mmproj.is_none(),
        "mlx backend does not support llama.cpp mmproj launch args"
    );
    if let Some(ctx_size) = spec.ctx_size_override {
        tracing::warn!(
            "Ignoring ctx-size override {} for mlx backend; mlx_lm.server does not expose an equivalent server flag",
            ctx_size
        );
    }

    validate_mlx_model(spec.model)?;
    let runtime = resolve_mlx_runtime()?;
    let log_path = temp_log_path("mesh-llm-mlx-server.log");
    let log_file = std::fs::File::create(&log_path).with_context(|| {
        format!(
            "Failed to create mlx server log file {}",
            log_path.display()
        )
    })?;
    let log_file2 = log_file.try_clone()?;

    let mut args = runtime.prefix_args.clone();
    args.extend_from_slice(&[
        "--model".to_string(),
        spec.model.to_string_lossy().to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        spec.http_port.to_string(),
    ]);
    if let Some(mode) = spec.mlx_mode {
        args.extend(mode.extra_server_args().iter().map(|arg| arg.to_string()));
    }

    let hostfile_path = if let Some(hostfile_json) = spec.mlx_hostfile_json {
        let rank = spec.mlx_rank.unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "mesh-llm-mlx-hostfile-rank-{rank}-{}.json",
            spec.http_port
        ));
        std::fs::write(&path, hostfile_json)
            .with_context(|| format!("failed to write MLX hostfile {}", path.display()))?;
        if let Ok(mut hostfiles) = mlx_hostfiles().lock() {
            hostfiles.push(path.clone());
        }
        Some(path)
    } else {
        None
    };

    tracing::info!(
        "Starting mlx_lm.server on :{} with model {} (rank {:?}, serves_http={}, bind_addrs={:?}, hostfile={})",
        spec.http_port,
        spec.model.display(),
        spec.mlx_rank,
        spec.mlx_serves_http,
        spec.mlx_bind_addrs,
        hostfile_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".to_string())
    );

    if spec.mlx_serves_http {
        register_mlx_model_override(spec.http_port, spec.model);
    }
    let mut cmd = Command::new(&runtime.executable);
    cmd.args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2));
    if let Some(rank) = spec.mlx_rank {
        cmd.env("MLX_RANK", rank.to_string());
    }
    if let Some(ref path) = hostfile_path {
        cmd.env("MLX_HOSTFILE", path);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            if spec.mlx_serves_http {
                unregister_mlx_model_override(spec.http_port);
            }
            if let Some(path) = hostfile_path {
                let _ = std::fs::remove_file(path);
            }
            return Err(err).with_context(|| {
                format!("Failed to start mlx_lm.server via {}", runtime.executable)
            });
        }
    };

    let url = format!("http://localhost:{}/health", spec.http_port);
    for _ in 0..600 {
        let ready = if spec.mlx_serves_http {
            reqwest_health_check(&url).await
        } else {
            !spec.mlx_bind_addrs.is_empty()
                && spec
                    .mlx_bind_addrs
                    .iter()
                    .all(|addr| tcp_addr_is_open(addr))
        };
        if ready {
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            let port = spec.http_port;
            let hostfile_path = hostfile_path.clone();
            let serves_http = spec.mlx_serves_http;
            tokio::spawn(async move {
                let _ = child.wait().await;
                if serves_http {
                    unregister_mlx_model_override(port);
                }
                if let Some(path) = hostfile_path {
                    let _ = std::fs::remove_file(path);
                }
                eprintln!("⚠️  mlx_lm.server process exited unexpectedly");
                let _ = death_tx.send(());
            });
            return Ok(death_rx);
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll mlx_lm.server status for {}",
                runtime.executable
            )
        })? {
            if spec.mlx_serves_http {
                unregister_mlx_model_override(spec.http_port);
            }
            if let Some(path) = hostfile_path {
                let _ = std::fs::remove_file(path);
            }
            anyhow::bail!(
                "mlx_lm.server exited before becoming ready (status: {status}). See {}",
                log_path.display()
            );
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    if spec.mlx_serves_http {
        unregister_mlx_model_override(spec.http_port);
    }
    if let Some(path) = hostfile_path {
        let _ = std::fs::remove_file(path);
    }
    anyhow::bail!(
        "mlx_lm.server failed to become healthy within 600s. See {}",
        log_path.display()
    );
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
/// `model_bytes` is the total GGUF file size, used to select KV cache quantization:
///   - < 5GB: FP16 (default) — small models, KV cache is tiny
///   - 5-50GB: Q8_0 — no measurable quality loss, saves ~50% KV memory
///   - > 50GB: Q4_0 — slight long-context degradation, but these models need every byte
/// Start llama-server. Returns a oneshot receiver that fires when the process exits.
pub async fn start_llama_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    model: &Path,
    http_port: u16,
    tunnel_ports: &[u16],
    tensor_split: Option<&str>,
    draft: Option<&Path>,
    draft_max: u16,
    model_bytes: u64,
    my_vram: u64,
    mmproj: Option<&Path>,
    ctx_size_override: Option<u32>,
    total_group_vram: Option<u64>,
) -> Result<tokio::sync::oneshot::Receiver<()>> {
    let llama_server = resolve_binary_path(bin_dir, "llama-server", binary_flavor)?;

    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());

    // Build --rpc argument: all tunnel ports as localhost endpoints
    let rpc_endpoints: Vec<String> = tunnel_ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect();
    let rpc_arg = rpc_endpoints.join(",");

    tracing::info!(
        "Starting llama-server on :{http_port} with model {} and --rpc {}",
        model.display(),
        rpc_arg
    );

    let llama_log = temp_log_path("mesh-llm-llama-server.log");
    let log_file = std::fs::File::create(&llama_log).with_context(|| {
        format!(
            "Failed to create llama-server log file {}",
            llama_log.display()
        )
    })?;
    let log_file2 = log_file.try_clone()?;

    // llama-server uses --rpc only for remote workers.
    // Context size: scale to available VRAM on the host node.
    // In split mode (pipeline parallel), each node holds a range of layers
    // and the KV cache for those layers is allocated on the same device.
    // So both weights and KV are distributed. The host only needs VRAM for
    // its share of weights + its share of KV. We estimate the host's weight
    // share proportionally and let llama-server pick the largest -c that fits.
    const GB: u64 = 1_000_000_000;
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        // Split mode: host holds its share of the weights
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        // Local mode: host holds all weights
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size: u32 = if let Some(override_ctx) = ctx_size_override {
        override_ctx
    } else if vram_after_model >= 30 * GB {
        65536 // 30GB+ free: full 64K context
    } else if vram_after_model >= 12 * GB {
        32768 // 12-30GB free: 32K
    } else if vram_after_model >= 6 * GB {
        16384 // 6-12GB free: 16K
    } else if vram_after_model >= 3 * GB {
        8192 // 3-6GB free: 8K
    } else {
        4096 // <3GB free: minimal
    };
    tracing::info!(
        "Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB VRAM, {:.1}GB free{})",
        model_bytes as f64 / GB as f64,
        host_model_bytes as f64 / GB as f64,
        my_vram as f64 / GB as f64,
        vram_after_model as f64 / GB as f64,
        if total_group_vram.is_some() {
            " [split]"
        } else {
            ""
        }
    );

    let mut args = vec!["-m".to_string(), model.to_string_lossy().to_string()];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    args.extend_from_slice(&[
        "-ngl".to_string(),
        "99".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-fit".to_string(),
        "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        http_port.to_string(),
        "-c".to_string(),
        ctx_size.to_string(),
        // Use deepseek format: thinking goes into reasoning_content field.
        // Goose/OpenAI clients parse this correctly. "none" leaks raw <think>
        // tags into content which is worse.
        "--reasoning-format".to_string(),
        "deepseek".to_string(),
        // Disable thinking by default. Thinking models (Qwen3, MiniMax) burn
        // 15-80s on hidden reasoning for no quality gain on most tasks, and
        // Qwen3.5-9B is completely broken (reasoning consumes all max_tokens).
        // API users can opt-in per-request with:
        //   "chat_template_kwargs": {"enable_thinking": true}
        "--reasoning-budget".to_string(),
        "0".to_string(),
    ]);
    // KV cache quantization based on model size:
    //   < 5GB: leave default (FP16) — small models, KV cache is negligible
    //   5-50GB: Q8_0 — essentially lossless, halves KV memory
    //   > 50GB: Q4_0 — slight long-context quality trade, but critical memory savings
    if model_bytes >= 50 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            "q4_0".to_string(),
            "--cache-type-v".to_string(),
            "q4_0".to_string(),
        ]);
        tracing::info!("KV cache: Q4_0 (model > 50GB)");
    } else if model_bytes >= 5 * GB {
        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            "q8_0".to_string(),
            "--cache-type-v".to_string(),
            "q8_0".to_string(),
        ]);
        tracing::info!("KV cache: Q8_0 (model 5-50GB)");
    }
    if let Some(ts) = tensor_split {
        args.push("--tensor-split".to_string());
        args.push(ts.to_string());
    }
    let local_device = resolve_device_for_binary(&llama_server.path, llama_server.flavor, None)?;
    if let Some(draft_path) = draft {
        if draft_path.exists() {
            if local_device != "CPU" {
                args.push("-md".to_string());
                args.push(draft_path.to_string_lossy().to_string());
                args.push("-ngld".to_string());
                args.push("99".to_string());
                args.push("--device-draft".to_string());
                args.push(local_device.clone());
                args.push("--draft-max".to_string());
                args.push(draft_max.to_string());
                tracing::info!(
                    "Speculative decoding: draft={}, draft-max={}, device={}",
                    draft_path.display(),
                    draft_max,
                    local_device
                );
            } else {
                tracing::warn!(
                    "Draft model present at {} but no GPU backend detected, skipping speculative decoding",
                    draft_path.display()
                );
            }
        } else {
            tracing::warn!(
                "Draft model not found at {}, skipping speculative decoding",
                draft_path.display()
            );
        }
    }
    if let Some(proj) = mmproj {
        if proj.exists() {
            args.push("--mmproj".to_string());
            args.push(proj.to_string_lossy().to_string());
            // Vision images can produce large token batches — need ubatch >= 2048
            args.push("--ubatch-size".to_string());
            args.push("2048".to_string());
            tracing::info!("Vision: mmproj={}", proj.display());
        } else {
            tracing::warn!("mmproj not found at {}, skipping vision", proj.display());
        }
    }
    let mut child = Command::new(&llama_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start llama-server at {}",
                llama_server.path.display()
            )
        })?;

    // Wait for health check
    let url = format!("http://localhost:{http_port}/health");
    for i in 0..600 {
        if i > 0 && i % 10 == 0 {
            let bytes = crate::tunnel::bytes_transferred();
            let kb = bytes as f64 / 1024.0;
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let transferred = if gb >= 1.0 {
                format!("{gb:.1} GB")
            } else if mb >= 1.0 {
                format!("{mb:.1} MB")
            } else {
                format!("{kb:.0} KB")
            };
            tracing::info!(
                "Still waiting for llama-server to load model... ({i}s, {transferred} transferred)"
            );
        }
        if reqwest_health_check(&url).await {
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                eprintln!("⚠️  llama-server process exited unexpectedly");
                let _ = death_tx.send(());
            });
            return Ok(death_rx);
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!("llama-server failed to become healthy within 600s");
}

/// Find an available TCP port
async fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Check if a port is accepting connections
async fn is_port_open(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

pub fn terminate_process_by_name(name: &str) -> bool {
    kill_process_by_name(name, false)
}

pub fn force_kill_process_by_name(name: &str) -> bool {
    kill_process_by_name(name, true)
}

fn kill_process_by_name(name: &str, force: bool) -> bool {
    #[cfg(windows)]
    {
        let image = platform_bin_name(name);
        let mut cmd = std::process::Command::new("taskkill");
        if force {
            cmd.arg("/F");
        }
        cmd.args(["/IM", &image]);
        cmd.status().is_ok_and(|status| status.success())
    }

    #[cfg(not(windows))]
    {
        let mut cmd = std::process::Command::new("pkill");
        if force {
            cmd.arg("-9");
        }
        cmd.args(["-f", name]);
        cmd.status().is_ok_and(|status| status.success())
    }
}

fn is_process_running(name: &str) -> bool {
    #[cfg(windows)]
    {
        let image = platform_bin_name(name);
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("IMAGENAME eq {image}")])
            .output()
            .map(|output| {
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout)
                        .to_ascii_lowercase()
                        .contains(&image.to_ascii_lowercase())
            })
            .unwrap_or(false)
    }

    #[cfg(not(windows))]
    {
        std::process::Command::new("pgrep")
            .args(["-f", name])
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false)
    }
}

/// Detect the best available compute device
fn detect_device() -> String {
    if cfg!(target_os = "macos") {
        return "MTL0".to_string();
    }

    // Linux: check for NVIDIA CUDA
    if command_has_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        return "CUDA0".to_string();
    }

    // Linux: check for NVIDIA Tegra/Jetson (tegrastats — Jetson AGX/NX devices support CUDA)
    // nvidia-smi is absent on Tegra; tegrastats is the canonical hardware stats tool.
    if let Ok(mut child) = std::process::Command::new("tegrastats")
        .args(["--interval", "1"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        let _ = child.kill();
        let _ = child.wait();
        return "CUDA0".to_string();
    }

    // ROCm/HIP
    if has_rocm_backend() {
        return "HIP0".to_string();
    }

    // Vulkan
    if command_succeeds("vulkaninfo", &["--summary"]) {
        return "Vulkan0".to_string();
    }

    "CPU".to_string()
}

fn has_rocm_backend() -> bool {
    #[cfg(windows)]
    {
        if std::env::var_os("ROCM_PATH").is_some() || std::env::var_os("HIP_PATH").is_some() {
            return true;
        }
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            let base = PathBuf::from(program_files).join("AMD");
            if base.join("ROCm").exists() || base.join("HIP").exists() {
                return true;
            }
        }
        command_has_output("hipInfo", &[]) || command_has_output("hipconfig", &[])
    }

    #[cfg(not(windows))]
    {
        command_has_output("rocm-smi", &["--showproductname"])
            || command_has_output("rocminfo", &[])
    }
}

fn command_succeeds(command: &str, args: &[&str]) -> bool {
    std::process::Command::new(command)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Simple HTTP health check (avoid adding reqwest as a dep — just use TCP + raw HTTP)
async fn reqwest_health_check(url: &str) -> bool {
    // Parse host:port from URL
    let url = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = url.split_once('/').unwrap_or((url, ""));
    let path = format!("/{path}");

    let Ok(mut stream) = tokio::net::TcpStream::connect(host_port).await else {
        return false;
    };

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }

    let mut response = vec![0u8; 1024];
    let Ok(n) = stream.read(&mut response).await else {
        return false;
    };

    let response = String::from_utf8_lossy(&response[..n]);
    response.contains("200 OK")
}

use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(test)]
mod tests {
    use super::{
        allocate_mlx_loopback_bind_ports, build_mlx_direct_ring_plans, build_mlx_quic_ring_plans,
        parse_available_devices, preferred_device, BinaryFlavor, MlxDistributedMode,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn parse_available_devices_ignores_non_device_lines() {
        let output = r#"
error: unknown device: HIP0
available devices:
No devices found
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 13737 MiB free)
  CPU: AMD Ryzen 7 7800X3D 8-Core Processor (192857 MiB, 192857 MiB free)
"#;

        assert_eq!(
            parse_available_devices(output),
            vec!["Vulkan0".to_string(), "CPU".to_string()]
        );
    }

    #[test]
    fn preferred_device_picks_vulkan_when_that_is_all_binary_supports() {
        let available = vec!["Vulkan0".to_string(), "CPU".to_string()];
        assert_eq!(
            preferred_device(&available, Some(BinaryFlavor::Vulkan)),
            Some("Vulkan0".to_string())
        );
    }

    #[test]
    fn infer_binary_flavor_from_filename() {
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan")),
            Some(BinaryFlavor::Vulkan)
        );
        #[cfg(windows)]
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan.exe")),
            Some(BinaryFlavor::Vulkan)
        );
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn platform_bin_name_preserves_existing_exe_suffix_case_insensitively() {
        assert_eq!(super::platform_bin_name("rpc-server.EXE"), "rpc-server.EXE");
    }

    #[test]
    fn allocate_mlx_loopback_bind_ports_is_dense() {
        assert_eq!(
            allocate_mlx_loopback_bind_ports(2, 2, 47000).unwrap(),
            vec![vec![47000, 47001], vec![47002, 47003]]
        );
    }

    #[test]
    fn build_mlx_direct_ring_plans_share_hostfile() {
        let plans = build_mlx_direct_ring_plans(&[
            vec!["10.0.0.1:47000".to_string()],
            vec!["10.0.0.2:47000".to_string()],
        ])
        .unwrap();

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].hostfile_json, plans[1].hostfile_json);
        assert!(plans[0].shim_routes.is_empty());
        assert!(plans[1].shim_routes.is_empty());
        assert!(plans[0].serves_http);
        assert!(!plans[1].serves_http);
    }

    #[test]
    fn build_mlx_quic_ring_plans_use_loopback_shims_for_remote_ranks() {
        let bind_ports = vec![vec![47000, 47001], vec![47002, 47003]];
        let plans = build_mlx_quic_ring_plans(&bind_ports, 48000).unwrap();

        assert_eq!(plans.len(), 2);
        assert_eq!(
            plans[0].bind_addrs,
            vec!["127.0.0.1:47000".to_string(), "127.0.0.1:47001".to_string()]
        );
        assert_eq!(plans[0].shim_routes.len(), 2);
        assert_eq!(plans[0].shim_routes[0].remote_rank, 1);
        assert_eq!(
            plans[0].shim_routes[0].remote_target_addr,
            "127.0.0.1:47002".to_string()
        );

        let hostfile0: Vec<Vec<String>> = serde_json::from_str(&plans[0].hostfile_json).unwrap();
        assert_eq!(hostfile0[0], plans[0].bind_addrs);
        assert_eq!(
            hostfile0[1],
            vec!["127.0.0.1:48000".to_string(), "127.0.0.1:48001".to_string()]
        );

        let hostfile1: Vec<Vec<String>> = serde_json::from_str(&plans[1].hostfile_json).unwrap();
        assert_eq!(
            hostfile1[0],
            vec!["127.0.0.1:48002".to_string(), "127.0.0.1:48003".to_string()]
        );
        assert_eq!(hostfile1[1], plans[1].bind_addrs);
    }

    #[test]
    fn mlx_rank_plan_builds_env_and_args() {
        let plans = build_mlx_direct_ring_plans(&[
            vec!["10.0.0.1:47000".to_string()],
            vec!["10.0.0.2:47000".to_string()],
        ])
        .unwrap();
        let env = plans[1].env(Path::new("/tmp/rank-1.hosts.json"));
        assert_eq!(env.get("MLX_RANK"), Some(&"1".to_string()));
        assert_eq!(
            env.get("MLX_HOSTFILE"),
            Some(&"/tmp/rank-1.hosts.json".to_string())
        );

        let args = plans[0].server_args(
            &PathBuf::from("/models/Qwen2.5-1.5B-Instruct-4bit"),
            "0.0.0.0",
            18080,
            MlxDistributedMode::Pipeline,
        );
        assert_eq!(
            args,
            vec![
                "--model".to_string(),
                "/models/Qwen2.5-1.5B-Instruct-4bit".to_string(),
                "--host".to_string(),
                "0.0.0.0".to_string(),
                "--port".to_string(),
                "18080".to_string(),
                "--pipeline".to_string(),
            ]
        );
    }
}
