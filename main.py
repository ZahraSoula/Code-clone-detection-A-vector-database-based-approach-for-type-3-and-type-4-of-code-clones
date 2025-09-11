import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logging import configure_logging
from utils.qdrant_utils import setup_qdrant

configure_logging()

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="poj")
args.add_argument("--qdrant_host_url", type=str, default="localhost")
args.add_argument("--qdrant_port", type=int, default=6333)
args.add_argument("--normalized", type=bool, default=False)
args.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
args.add_argument("--num_samples", type=int, default=500)
args.add_argument("--k", type=int, default=100)
args = args.parse_args()

path = os.getcwd()

# Extract model name from path
model_name = args.embedding_model.split("/")[-1]

collection_name = (
    args.dataset + "_" + model_name + "_" + "not-normalized"
    if not args.normalized
    else "normalized"
)

encoder = SentenceTransformer(args.embedding_model, trust_remote_code=True, device="cuda:0")
driver = QdrantClient(host=args.qdrant_host_url, port=args.qdrant_port)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

if args.dataset == "bcb":
    from bcb_utils import (
        benchmark,
        create_batch_dfs,
        create_clone_mappings,
        ingest_data,
    )

    clones_df = pd.read_csv(
        "C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\BigCloneBench\\CLONES.csv"
    )
    functions_df = pd.read_csv(
        "C:\\Users\\MSI\\Desktop\\Replika\\datasets\\BigCloneBench\\FUNCTIONS_CLEANED.csv"
    )
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

    results_df.to_csv(
        results_dir / f"{collection_name}_detailed_results.csv", index=False
    )
    query_stats_df.to_csv(
        results_dir / f"{collection_name}_query_stats.csv", index=False
    )

elif args.dataset == "poj":
    from poj_utils import (
        benchmark,
        create_batches,
        get_dataset_structure,
        ingest_data,
        read_batch,
    )

    dataset_path = os.path.join(
        os.getcwd(),
        "datasets",
        "poj104",
    )
    collection_name = (
        "poj" + "_" + model_name + "_" + "not-normalized"
        if not args.normalized
        else "normalized"
    )
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

    results_df.to_csv(
        results_dir / f"{collection_name}_detailed_results.csv", index=False
    )
    query_stats_df.to_csv(
        results_dir / f"{collection_name}_query_stats.csv", index=False
    )
