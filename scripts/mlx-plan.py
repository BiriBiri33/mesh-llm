#!/usr/bin/env python3
import argparse
import json
import math


def parse_candidate(raw: str):
    name, vram = raw.split(":", 1)
    return {"id": name, "vram_gb": float(vram)}


def sort_candidates(candidates):
    return sorted(candidates, key=lambda c: (c["vram_gb"], c["id"]), reverse=True)


def select_cohort(candidates, model_gb: float, force_split: bool):
    need_gb = model_gb * 1.1
    min_nodes = 2 if force_split else 1
    selected = []
    total = 0.0
    for candidate in sort_candidates(candidates):
        if total >= need_gb and len(selected) >= min_nodes:
            break
        selected.append(candidate)
        total += candidate["vram_gb"]
    return selected, need_gb, total


def shim_base_port(model_name: str) -> int:
    h = 0
    for ch in model_name.encode():
        h = (h * 31 + ch) & 0xFFFFFFFFFFFFFFFF
    return 40000 + (h % 200) * 70


def allocate_bind_ports(world_size: int, connections_per_rank: int, starting_port: int):
    next_port = starting_port
    ranks = []
    for _ in range(world_size):
        ports = []
        for _ in range(connections_per_rank):
            ports.append(next_port)
            next_port += 1
        ranks.append(ports)
    return ranks


def build_quic_plan(model_name: str, selected, connections_per_rank: int):
    world_size = len(selected)
    bind_ports = allocate_bind_ports(
        world_size, connections_per_rank, shim_base_port(model_name)
    )
    bind_addrs = [
        [f"127.0.0.1:{port}" for port in rank_ports] for rank_ports in bind_ports
    ]

    next_shim_port = shim_base_port(model_name) + 32
    used_ports = {port for rank in bind_ports for port in rank}
    ranks = []
    for rank in range(world_size):
        hostfile = []
        shim_routes = []
        for target_rank in range(world_size):
            if target_rank == rank:
                hostfile.append(bind_addrs[target_rank])
                continue
            row = []
            for connection_idx in range(connections_per_rank):
                while next_shim_port in used_ports:
                    next_shim_port += 1
                local = f"127.0.0.1:{next_shim_port}"
                used_ports.add(next_shim_port)
                next_shim_port += 1
                row.append(local)
                shim_routes.append(
                    {
                        "local_listen_addr": local,
                        "remote_rank": target_rank,
                        "remote_peer_id": selected[target_rank]["id"],
                        "remote_target_addr": bind_addrs[target_rank][connection_idx],
                    }
                )
            hostfile.append(row)
        ranks.append(
            {
                "rank": rank,
                "peer_id": selected[rank]["id"],
                "vram_gb": selected[rank]["vram_gb"],
                "bind_addrs": bind_addrs[rank],
                "serves_http": rank == 0,
                "env": {
                    "MLX_RANK": str(rank),
                    "MLX_HOSTFILE": f"/tmp/mesh-llm-mlx-hostfile-rank-{rank}.json",
                },
                "hostfile_json": json.dumps(hostfile),
                "shim_routes": shim_routes,
            }
        )
    return bind_ports, ranks


def main():
    parser = argparse.ArgumentParser(description="Print the deterministic MLX mesh plan.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-gb", type=float, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--connections-per-rank", type=int, default=1)
    parser.add_argument("--force-split", action="store_true")
    args = parser.parse_args()

    candidates = [parse_candidate(raw) for raw in args.candidate]
    selected, need_gb, total_gb = select_cohort(
        candidates, args.model_gb, args.force_split
    )
    bind_ports, ranks = build_quic_plan(
        args.model_name, selected, args.connections_per_rank
    )

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "model_gb": args.model_gb,
                "required_gb": round(need_gb, 3),
                "connections_per_rank": args.connections_per_rank,
                "force_split": args.force_split,
                "selected_peer_ids": [entry["id"] for entry in selected],
                "selected_total_vram_gb": round(total_gb, 3),
                "shim_base_port": shim_base_port(args.model_name),
                "bind_ports": bind_ports,
                "ranks": ranks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
