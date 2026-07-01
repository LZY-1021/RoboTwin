#!/usr/bin/env python3
import argparse
from pathlib import Path

from compare_trace_cosine import (
    action_trajectory,
    compare_action,
    compare_kv,
    compare_mlp,
    compare_obs,
    cosine,
    episode_dir,
    fmt,
    infer_dirs,
    print_section,
    read_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare adjacent PI05 trace inferences within one episode with cosine similarity."
    )
    parser.add_argument("trace_dir", type=Path, help="Path to a traces directory.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to compare.")
    parser.add_argument("--start", type=int, default=0, help="First inference index to compare.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end inference index.")
    parser.add_argument("--max-layers", type=int, default=None, help="Limit printed per-layer KV/MLP rows.")
    args = parser.parse_args()

    ep_dir = episode_dir(args.trace_dir, args.episode)
    infers = infer_dirs(ep_dir)
    end = len(infers) if args.end is None else min(args.end, len(infers))
    start = max(0, args.start)

    print(f"Trace: {args.trace_dir}")
    print(f"Episode: {ep_dir.name} {read_json(ep_dir / 'meta.json')}")
    print(f"Adjacent comparisons: {max(0, end - start - 1)}")
    print()

    for idx in range(start, max(start, end - 1)):
        infer_a = infers[idx]
        infer_b = infers[idx + 1]
        print(f"=== infer_{idx:04d}_vs_{idx + 1:04d} ===")
        print_section("obs", compare_obs(infer_a, infer_b), args.max_layers)
        print_section("action", compare_action(infer_a, infer_b), args.max_layers)
        print_section("kv", compare_kv(infer_a, infer_b), args.max_layers)
        print_section("mlp", compare_mlp(infer_a, infer_b), args.max_layers)
        print()

    print("=== episode adjacent action trajectory ===")
    for key in ("action_executed", "action_chunk_full"):
        traj_a = action_trajectory(infers[start : max(start, end - 1)], key)
        traj_b = action_trajectory(infers[start + 1 : end], key)
        print(f"  adjacent_trajectory.{key}: {fmt(cosine(traj_a, traj_b))}")


if __name__ == "__main__":
    main()
