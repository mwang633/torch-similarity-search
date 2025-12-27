#!/usr/bin/env python3
"""Benchmark IVFPQ: PyTorch vs FAISS on GPU.

Measures recall and latency for both implementations across various configurations.

Usage:
    python benchmarks/benchmark_ivfpq.py
    python benchmarks/benchmark_ivfpq.py --n-vectors 100000 --n-queries 1000
    python benchmarks/benchmark_ivfpq.py --device cpu  # CPU-only benchmark
"""

import argparse
import time
from typing import Tuple

import numpy as np
import torch

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: pip install faiss-gpu-cu12")
    exit(1)

from torch_similarity_search import from_faiss


def generate_data(
    n_vectors: int, n_queries: int, dim: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random vectors for benchmarking."""
    np.random.seed(seed)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    return vectors, queries


def compute_ground_truth(
    vectors: np.ndarray, queries: np.ndarray, k: int, device: str, metric: str = "l2"
) -> np.ndarray:
    """Compute ground truth using FAISS exact search (handles large scale)."""
    dim = vectors.shape[1]
    # Use FAISS Flat for ground truth - more memory efficient for large datasets
    if metric == "ip":
        gt_index = faiss.IndexFlatIP(dim)
    else:
        gt_index = faiss.IndexFlatL2(dim)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        gt_index = faiss.index_cpu_to_gpu(res, 0, gt_index)
    gt_index.add(vectors)
    _, gt_indices = gt_index.search(queries, k)
    return gt_indices


def compute_recall(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    recall = 0
    for i in range(pred.shape[0]):
        recall += len(set(pred[i].tolist()) & set(gt[i].tolist()))
    return recall / (pred.shape[0] * k)


def benchmark_faiss(
    faiss_index,
    queries: np.ndarray,
    k: int,
    n_warmup: int = 5,
    n_runs: int = 20,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Benchmark FAISS search latency."""
    if use_gpu:
        # Move index to GPU (keep res reference to prevent GC)
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        _ = res  # Keep reference alive

    # Warmup
    for _ in range(n_warmup):
        faiss_index.search(queries, k)

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        if use_gpu:
            torch.cuda.synchronize()  # Use PyTorch for GPU sync
        start = time.perf_counter()
        _, indices = faiss_index.search(queries, k)
        if use_gpu:
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return indices, np.mean(latencies), np.std(latencies)


def benchmark_torch(
    torch_index,
    queries_tensor: torch.Tensor,
    k: int,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Tuple[np.ndarray, float, float]:
    """Benchmark PyTorch search latency."""
    device = queries_tensor.device

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            torch_index.search(queries_tensor, k)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _, indices = torch_index.search(queries_tensor, k)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    return indices.cpu().numpy(), np.mean(latencies), np.std(latencies)


def run_benchmark(
    n_vectors: int,
    n_queries: int,
    dim: int,
    nlist: int,
    M: int,
    nbits: int,
    nprobe: int,
    k: int,
    device: str,
    metric: str,
    n_warmup: int,
    n_runs: int,
) -> dict:
    """Run a single benchmark configuration."""
    print(f"\n{'=' * 60}")
    print(f"Config: {n_vectors:,} vectors, {n_queries} queries, dim={dim}")
    print(f"IVFPQ: nlist={nlist}, M={M}, nbits={nbits}, nprobe={nprobe}, k={k}")
    print(f"Metric: {metric}, Device: {device}, Compiled: Yes")
    print("=" * 60)

    # Generate data
    print("\nGenerating data...")
    vectors, queries = generate_data(n_vectors, n_queries, dim)

    # Compute ground truth
    print("Computing ground truth (exact search)...")
    gt_indices = compute_ground_truth(vectors, queries, k, device, metric)

    # Build FAISS index
    print("Building FAISS IVFPQ index...")
    if metric == "ip":
        quantizer = faiss.IndexFlatIP(dim)
        faiss_index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT
        )
    else:
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, nbits)
    faiss_index.train(vectors)
    faiss_index.add(vectors)
    faiss_index.nprobe = nprobe

    # Convert to PyTorch
    print("Converting to PyTorch...")
    torch_index = from_faiss(faiss_index)
    torch_index.nprobe = nprobe

    use_gpu = device == "cuda"
    if use_gpu:
        torch_index = torch_index.cuda()
        queries_tensor = torch.from_numpy(queries).cuda()
    else:
        queries_tensor = torch.from_numpy(queries)

    # Compile model for optimized performance
    print("Compiling PyTorch model")
    torch_index = torch.compile(torch_index)

    # Benchmark FAISS
    print(f"\nBenchmarking FAISS ({n_warmup} warmup, {n_runs} runs)...")
    faiss_indices, faiss_mean, faiss_std = benchmark_faiss(
        faiss_index, queries, k, n_warmup, n_runs, use_gpu
    )
    faiss_recall = compute_recall(faiss_indices, gt_indices, k)

    # Benchmark PyTorch
    print(f"Benchmarking PyTorch ({n_warmup} warmup, {n_runs} runs)...")
    torch_indices, torch_mean, torch_std = benchmark_torch(
        torch_index, queries_tensor, k, n_warmup, n_runs
    )
    torch_recall = compute_recall(torch_indices, gt_indices, k)

    # Results
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"{'Metric':<20} {'FAISS':<20} {'PyTorch':<20}")
    print("-" * 60)
    print(f"{'Recall@' + str(k):<20} {faiss_recall:<20.4f} {torch_recall:<20.4f}")
    print(
        f"{'Latency (ms)':<20} {faiss_mean:<10.2f} +/- {faiss_std:<6.2f} {torch_mean:<10.2f} +/- {torch_std:<6.2f}"
    )
    print(
        f"{'QPS':<20} {n_queries / faiss_mean * 1000:<20.1f} {n_queries / torch_mean * 1000:<20.1f}"
    )
    print("-" * 60)

    speedup = faiss_mean / torch_mean
    if speedup > 1:
        print(f"PyTorch is {speedup:.2f}x faster than FAISS")
    else:
        print(f"FAISS is {1 / speedup:.2f}x faster than PyTorch")

    recall_diff = torch_recall - faiss_recall
    if abs(recall_diff) < 0.001:
        print("Recall is identical")
    elif recall_diff > 0:
        print(f"PyTorch has {recall_diff:.4f} higher recall")
    else:
        print(f"FAISS has {-recall_diff:.4f} higher recall")

    return {
        "n_vectors": n_vectors,
        "n_queries": n_queries,
        "dim": dim,
        "nlist": nlist,
        "M": M,
        "nbits": nbits,
        "nprobe": nprobe,
        "k": k,
        "device": device,
        "faiss_recall": faiss_recall,
        "torch_recall": torch_recall,
        "faiss_latency_ms": faiss_mean,
        "faiss_latency_std": faiss_std,
        "torch_latency_ms": torch_mean,
        "torch_latency_std": torch_std,
        "metric": metric,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark IVFPQ: PyTorch vs FAISS")
    parser.add_argument(
        "--n-vectors", type=int, default=1_000_000, help="Number of vectors to index"
    )
    parser.add_argument(
        "--n-queries", type=int, default=1000, help="Number of query vectors"
    )
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument(
        "--nlist", type=int, default=1000, help="Number of IVF clusters"
    )
    parser.add_argument("--M", type=int, default=8, help="Number of PQ subquantizers")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per subquantizer")
    parser.add_argument(
        "--nprobe", type=int, default=32, help="Number of clusters to probe"
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "ip"],
        help="Distance metric: l2 (Euclidean) or ip (inner product)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--n-runs", type=int, default=20, help="Benchmark iterations")
    parser.add_argument(
        "--sweep-nprobe",
        action="store_true",
        help="Sweep nprobe values [1, 5, 10, 20, 50, 100]",
    )
    args = parser.parse_args()

    print("IVFPQ Benchmark: PyTorch vs FAISS")
    print("=" * 60)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    if args.sweep_nprobe:
        # Sweep nprobe values
        nprobe_values = [1, 5, 10, 20, 50, min(100, args.nlist)]
        results = []
        for nprobe in nprobe_values:
            result = run_benchmark(
                n_vectors=args.n_vectors,
                n_queries=args.n_queries,
                dim=args.dim,
                nlist=args.nlist,
                M=args.M,
                nbits=args.nbits,
                nprobe=nprobe,
                k=args.k,
                device=args.device,
                metric=args.metric,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
            )
            results.append(result)

        # Summary table
        print("\n" + "=" * 80)
        print(f"Summary: nprobe sweep (metric={args.metric})")
        print("=" * 80)
        print(
            f"{'nprobe':<8} {'FAISS Recall':<14} {'Torch Recall':<14} {'FAISS ms':<12} {'Torch ms':<12} {'Ratio':<10}"
        )
        print("-" * 80)
        for r in results:
            ratio = r["faiss_latency_ms"] / r["torch_latency_ms"]
            ratio_str = f"{ratio:.2f}x" if ratio >= 1 else f"{1 / ratio:.2f}x"
            print(
                f"{r['nprobe']:<8} {r['faiss_recall']:<14.4f} {r['torch_recall']:<14.4f} "
                f"{r['faiss_latency_ms']:<12.2f} {r['torch_latency_ms']:<12.2f} {ratio_str:<10}"
            )
    else:
        # Single run
        run_benchmark(
            n_vectors=args.n_vectors,
            n_queries=args.n_queries,
            dim=args.dim,
            nlist=args.nlist,
            M=args.M,
            nbits=args.nbits,
            nprobe=args.nprobe,
            k=args.k,
            device=args.device,
            metric=args.metric,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )


if __name__ == "__main__":
    main()
