import os
import uuid
from pathlib import Path

import chardet
import pandas as pd
from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.qdrant_utils import search


def get_dataset_structure(directory: str):
    """
    Get the structure of the dataset.

    Args:
        directory: The directory to search.
    """
    files = []
    for path, dir_list, file_list in os.walk(directory):
        for file_name in file_list:
            if file_name.endswith(".txt"):
                files.append(os.path.join(path, file_name))

    return files


def create_batches(file_list: list, batch_size: int = 128):
    """
    Create batches of the file list

    Args:
        file_list (list): the list of files
        batch_size (int, optional): the size of the batch. Defaults to 128.
    """
    batches = []
    for i in range(0, len(file_list), batch_size):
        batches.append(file_list[i : i + batch_size])

    return batches


def read_batch(batch: list[str]):
    """
    Read the rows of the mapping, opens them, creates the UUIDs, stores them in a new dataframe
    uses multithreading to read the files

    Args:
        mapping (pd.DataFrame): the mapping dataframe
    """

    try:
        results = []
        for row in tqdm(batch, total=len(batch), desc="Reading batch", leave=False):
            path_obj = Path(row)
            clone_type = path_obj.parent.name
            code_name = path_obj.name

            code_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, clone_type + code_name)

            with open(f"{row}", "rb") as file:
                raw_data = file.read()
                # Detect the encoding of the file
                result = chardet.detect(raw_data)
                encoding = result["encoding"]
                # Decode the raw data using the detected encoding
                code_contents = raw_data.decode(encoding, errors="ignore")

            results.append([clone_type, code_name, code_uuid, code_contents])

        batch_df = pd.DataFrame(
            results,
            columns=[
                "clone_type",
                "code_name",
                "code_uuid",
                "code_contents",
            ],
        )

        return batch_df

    except Exception as e:
        logger.opt(exception=True).error(f"Failed to read batch: {e}")
        raise e


def ingest_data(
    driver: QdrantClient,
    batch_df: pd.DataFrame,
    encoder: SentenceTransformer,
    collection_name: str,
) -> None:
    """
    Ingests data into the specified collection using the provided QdrantClient driver.

    Args:
        driver (QdrantClient): The QdrantClient driver used to interact with the Qdrant service.
        mapping (pd.DataFrame): The DataFrame containing the data to be ingested.
        encoder (SentenceTransformer): The SentenceTransformer model used to encode the code contents.
        collection_name (str): The name of the collection where the data will be ingested.
        batch_size (int, optional): The size of each batch for processing the data. Defaults to 128.
    """

    embeddings = encoder.encode(batch_df["code_contents"].tolist())

    points = [
        models.PointStruct(
            id=str(row["code_uuid"]),
            vector=embeddings[index].tolist(),
            payload={
                "clone_id": row["clone_type"],
                "contents": row["code_contents"],
                "name": row["code_name"],
            },
        )
        for index, row in batch_df.iterrows()
    ]

    driver.upload_points(
        collection_name=collection_name,
        points=points,
        max_retries=3,
    )


def benchmark(
    driver: QdrantClient,
    encoder: SentenceTransformer,
    collection_name: str,
    dataset: pd.DataFrame,
    num_samples: int = 10,
    k: int = 10,
):
    """
    Benchmarks the performance of a search system using various metrics.

    Parameters:
    driver (QdrantClient): The search driver used to perform the search.
    encoder (SentenceTransformer): The encoder used to encode the search queries.
    collection_name (str): The name of the collection to search within.
    dataset (pandas.DataFrame): The dataset containing the samples to benchmark.
    num_samples (int, optional): The number of samples to use for benchmarking. Default is 10.
    k (int, optional): The number of top results to consider for each query. Default is 10.

    Returns:
    tuple: A tuple containing the following metrics:
        - success_rate (float): The success rate at k.
        - mean_precision_at_k (float): The mean precision at k.
        - mrr (float): The mean reciprocal rank (MRR).
        - map_at_k (float): The mean average precision at k (MAP@k).
    """
    samples = dataset.sample(num_samples).reset_index(drop=True)

    total_success = 0
    total_precision = 0
    reciprocal_ranks = []
    average_precisions = []

    # New: Store statistics
    query_stats = []
    detailed_results = []

    for index, sample in tqdm(
        samples.iterrows(), total=num_samples, desc=f"Benchmarking -> {collection_name}"
    ):
        code_to_search = sample["code_contents"]
        code_id = str(sample["code_uuid"])
        clone_project = sample["clone_type"]

        results = search(
            driver=driver,
            encoder=encoder,
            collection_name=collection_name,
            query=code_to_search,
            limit=k,
        )
        results = [result for result in results if result["id"] != code_id]

        # Collect scores for same type and different type
        same_type_scores = []
        diff_type_scores = []

        for result in results:
            is_same_type = result["payload"]["clone_id"] == clone_project
            score = result["score"]

            if is_same_type:
                same_type_scores.append(score)
            else:
                diff_type_scores.append(score)

            detailed_results.append(
                {
                    "query_id": code_id,
                    "result_id": result["id"],
                    "result_type": result["payload"]["clone_id"],
                    "similarity_score": score,
                    "is_same_type": is_same_type,
                }
            )

        # Calculate average scores for this query
        avg_same_type_score = (
            sum(same_type_scores) / len(same_type_scores) if same_type_scores else 0
        )
        avg_diff_type_score = (
            sum(diff_type_scores) / len(diff_type_scores) if diff_type_scores else 0
        )

        query_stats.append(
            {
                "query_id": code_id,
                "query_type": clone_project,
                "avg_same_type_score": avg_same_type_score,
                "avg_diff_type_score": avg_diff_type_score,
                "num_same_type_found": len(same_type_scores),
                "num_diff_type_found": len(diff_type_scores),
            }
        )

        # Success Rate Calculation
        num_retrieved_clones = sum(
            [
                1 if result["payload"]["clone_id"] == clone_project else 0
                for result in results
            ]
        )
        if num_retrieved_clones > 0:
            total_success += 1

        # Precision at k
        precision = num_retrieved_clones / k
        total_precision += precision

        # Reciprocal Rank
        try:
            first_relevant_rank = [
                result["payload"]["clone_id"] for result in results
            ].index(clone_project) + 1
            reciprocal_rank = 1 / first_relevant_rank
        except ValueError:
            reciprocal_rank = 0
        reciprocal_ranks.append(reciprocal_rank)

        # Average Precision at k
        cumulative_precision = 0
        relevant_items = 0
        precisions_at_relevant = []
        for i, result in enumerate(results, start=1):
            if result["payload"]["clone_id"] == clone_project:
                relevant_items += 1
                cumulative_precision = relevant_items / i
                precisions_at_relevant.append(cumulative_precision)
        if precisions_at_relevant:
            average_precision = sum(precisions_at_relevant) / k
        else:
            average_precision = 0
        average_precisions.append(average_precision)

    success_rate = total_success / num_samples
    mean_precision_at_k = total_precision / num_samples
    mrr = sum(reciprocal_ranks) / num_samples
    map_at_k = sum(average_precisions) / num_samples

    # Convert results to DataFrames
    results_df = pd.DataFrame(detailed_results)
    query_stats_df = pd.DataFrame(query_stats)

    # Calculate overall averages
    overall_stats = {
        "global_avg_same_type_score": query_stats_df["avg_same_type_score"].mean(),
        "global_avg_diff_type_score": query_stats_df["avg_diff_type_score"].mean(),
    }

    print(f"Success Rate at {k}: {success_rate:.4f}")
    print(f"Mean Precision at {k}: {mean_precision_at_k:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Mean Average Precision at {k} (MAP@{k}): {map_at_k:.4f}")
    print(
        f"Global Average Same-Type Score: {overall_stats['global_avg_same_type_score']:.4f}"
    )
    print(
        f"Global Average Different-Type Score: {overall_stats['global_avg_diff_type_score']:.4f}"
    )

    return success_rate, mean_precision_at_k, mrr, map_at_k, results_df, query_stats_df
