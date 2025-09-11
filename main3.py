import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.logging import configure_logging
from utils.qdrant_utils import setup_qdrant
import psutil  # For memory usage analysis
import time  # For query time evaluation

configure_logging()

# Argument Parsing
args = ArgumentParser()
args.add_argument("--dataset", type=str, default="poj")
args.add_argument("--qdrant_host_url", type=str, default="localhost")
args.add_argument("--qdrant_port", type=int, default=6333)
args.add_argument("--embedding_model", type=str, default="ncoop57/codeformer-java")
args.add_argument("--num_samples", type=int, default=500)
args.add_argument("--k", type=int, default=100)
args.add_argument("--scalability_test", action="store_true", help="Enable scalability testing")
args = args.parse_args()

# Initialize Variables
path = os.getcwd()
model_name = args.embedding_model.split("/")[-1]
collection_name = f"{args.dataset}_{model_name}_not-normalized"
encoder = SentenceTransformer(args.embedding_model, trust_remote_code=True, device="cuda:0")
driver = QdrantClient(host=args.qdrant_host_url, port=args.qdrant_port)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Scalability Test Function
def scalability_test(driver, encoder, collection_name, dataset_sizes):
    """
    Perform scalability analysis: memory usage and query time evaluation.
    """
    memory_usage_results = []
    query_time_results = []

    for size in dataset_sizes:
        print(f"Running scalability test for dataset size: {size} records")

        # Simulate a subset of the dataset (e.g., first `size` records)
        if args.dataset == "bcb":
            from bcb_utils import create_batch_dfs, ingest_data
            clones_df = pd.read_csv("C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\BigCloneBench\\CLONES.csv")
            functions_df = pd.read_csv("C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\FUNCTIONS_CLEANED.csv")
            mappings = create_clone_mappings(clones_df)
            final = create_batch_dfs(mappings, functions_df)[:size]  # Use only `size` records
        elif args.dataset == "poj":
            from poj_utils import create_batches, read_batch, get_dataset_structure
            dataset_path = os.path.join(os.getcwd(), "datasets", "poj104")
            dataset = get_dataset_structure(dataset_path)
            batches = create_batches(dataset, batch_size=128)[:size]  # Use only `size` batches
            final = [read_batch(batch) for batch in batches]

        # Measure Memory Usage During Ingestion
        memory_before_ingestion = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
        if not driver.collection_exists(collection_name):
            setup_qdrant(
                qdrant_host_url=args.qdrant_host_url,
                qdrant_port=args.qdrant_port,
                collection_name=collection_name,
                embedding_model=args.embedding_model,
            )
            for batch_df in tqdm(final, desc="Ingesting data"):
                ingest_data(driver, batch_df, encoder, collection_name)
        memory_after_ingestion = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_usage_results.append({"size": size, "memory_usage_mb": memory_after_ingestion - memory_before_ingestion})

        # Measure Query Time
        query_times = []
        for _ in range(10):  # Run 10 queries for each size to average the results
            sample_query_code = encoder.encode([final[0]["code_snippet"]])  # Use an actual code snippet as the query
            start_time = time.time()
            driver.search(
                collection_name=collection_name,
                query_vector=sample_query_code,
                limit=args.k
            )
            end_time = time.time()
            query_times.append(end_time - start_time)
        avg_query_time = sum(query_times) / len(query_times)
        query_time_results.append({"size": size, "avg_query_time_seconds": avg_query_time})

    # Save Results with Unique File Names
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    pd.DataFrame(memory_usage_results).to_csv(results_dir / f"{collection_name}_memory_usage_{timestamp}.csv", index=False)
    pd.DataFrame(query_time_results).to_csv(results_dir / f"{collection_name}_query_time_{timestamp}.csv", index=False)

# Main Execution
if args.dataset == "bcb" or args.dataset == "poj":
    if args.scalability_test:
        # Define dataset sizes for scalability testing
        dataset_sizes = [1000, 5000, 10000, 20000, 50000]  # Larger dataset sizes for better results
        scalability_test(driver, encoder, collection_name, dataset_sizes)
    else:
        # Normal execution (as per your original script)
        if args.dataset == "bcb":
            from bcb_utils import benchmark, create_batch_dfs, create_clone_mappings, ingest_data
            clones_df = pd.read_csv("C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\BigCloneBench\\CLONES.csv")
            functions_df = pd.read_csv("C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\FUNCTIONS_CLEANED.csv")
            mappings = create_clone_mappings(clones_df)
            final = create_batch_dfs(mappings, functions_df)
            if not driver.collection_exists(collection_name):
                setup_qdrant(
                    qdrant_host_url=args.qdrant_host_url,
                    qdrant_port=args.qdrant_port,
                    collection_name=collection_name,
                    embedding_model=args.embedding_model,
                )
                ingest_data(driver, final, encoder, collection_name, 16)
            (
                success_rate,
                mean_precision_at_k,
                mrr,
                map_at_k,
                results_df,
                query_stats_df,
            ) = benchmark(
                driver,
                encoder,
                collection_name,
                final,
                num_samples=args.num_samples,
                k=args.k,
            )
            results_df.to_csv(results_dir / f"{collection_name}_detailed_results.csv", index=False)
            query_stats_df.to_csv(results_dir / f"{collection_name}_query_stats.csv", index=False)
        elif args.dataset == "poj":
            from poj_utils import benchmark, create_batches, get_dataset_structure, ingest_data, read_batch
            dataset_path = os.path.join(os.getcwd(), "datasets", "poj104")
            dataset = get_dataset_structure(dataset_path)
            batches = create_batches(dataset, 128)
            batch_dfs = [read_batch(batch) for batch in batches]
            if not driver.collection_exists(collection_name):
                setup_qdrant(
                    qdrant_host_url=args.qdrant_host_url,
                    qdrant_port=args.qdrant_port,
                    collection_name=collection_name,
                    embedding_model=args.embedding_model,
                )
                for index, batch_df in tqdm(
                    enumerate(batch_dfs), total=len(batch_dfs), desc="Ingesting data"
                ):
                    ingest_data(driver, batch_df, encoder, collection_name)
            final_df = pd.concat(batch_dfs)
            (
                success_rate,
                mean_precision_at_k,
                mrr,
                map_at_k,
                results_df,
                query_stats_df,
            ) = benchmark(
                driver,
                encoder,
                collection_name,
                final_df,
                num_samples=args.num_samples,
                k=args.k,
            )
            results_df.to_csv(results_dir / f"{collection_name}_detailed_results.csv", index=False)
            query_stats_df.to_csv(results_dir / f"{collection_name}_query_stats.csv", index=False)