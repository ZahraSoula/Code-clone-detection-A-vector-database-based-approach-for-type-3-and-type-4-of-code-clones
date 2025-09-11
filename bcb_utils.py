import uuid
from collections import defaultdict

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.qdrant_utils import search


def create_clone_mappings(clones_df: pd.DataFrame) -> dict:
    """
    The CLONES.csv file has the entire mapping between clones.
    Anything outside of FUNCTION_ID_ONE and FUNCTION_ID_TWO is not needed.
    this function reads the clones_df and creates a mapping between the two functions.
    the dictionary returned contains {"FUNCTION_ID_ONE": [list of all functions that are clones of FUNCTION_ID_ONE]}
    Args:
        clones_df (pd.DataFrame): _description_

    Returns:
        dict: _description_
    """
    mapping = defaultdict(set)
    for func_id_one, func_id_two in tqdm(
        clones_df[["FUNCTION_ID_ONE", "FUNCTION_ID_TWO"]].itertuples(index=False),
        total=len(clones_df),
        desc="Creating mappings",
    ):
        mapping[func_id_one].add(func_id_two)
        mapping[func_id_two].add(func_id_one)

    return dict(mapping)


def create_batch_dfs(clone_mappings: dict, function_mappings: pd.DataFrame):
    """
    Creates a batch DataFrame containing code snippets and their corresponding clones.

    Args:
        clone_mappings (dict): A dictionary where keys are function IDs and values are lists of clone IDs.
        function_mappings (pd.DataFrame): A DataFrame containing function metadata with columns "ID", "STARTLINE", "ENDLINE", and "NAME".

    Returns:
        pd.DataFrame: A DataFrame with columns "code_uuid", "func_id", "code_contents", and "clones".
    """
    results = []
    for func_id, clones in tqdm(
        clone_mappings.items(),
        desc="Creating batch dfs",
        total=len(clone_mappings),
    ):
        try:
            code_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(func_id))
            code_contents = function_mappings[function_mappings["ID"] == func_id]

            code_contents_line_start = code_contents["STARTLINE"].values[0]
            code_contents_line_end = code_contents["ENDLINE"].values[0]
            code_contents_path = code_contents["NAME"].values[0]

            code_contents = "\n".join(
                open(code_contents_path, "r").readlines()[
                    code_contents_line_start:code_contents_line_end
                ]
            )
            clones = list(clone_mappings[func_id])
            results.append([code_uuid, func_id, code_contents, clones])
        except Exception:
            pass
    df = pd.DataFrame(
        results, columns=["code_uuid", "func_id", "code_contents", "clones"]
    )
    return df


def ingest_data(
    driver: QdrantClient,
    data_df: pd.DataFrame,
    encoder: SentenceTransformer,
    collection_name: str,
    batch_size: int = 128,
):
    """
    Ingests data into the Qdrant index.

    Args:
        driver (QdrantClient): The Qdrant client used for uploading points.
        mapping (pd.DataFrame): The mapping dataframe containing the data to be ingested.
        functions_mappings (pd.DataFrame): The functions mappings dataframe.
        encoder (SentenceTransformer): The encoder used for encoding the code contents.
        collection_name (str): The name of the collection in the Qdrant index.
        batch_size (int, optional): The batch size for processing the data. Defaults to 128.
    """

    batches_df = [
        data_df[i : i + batch_size] for i in range(0, len(data_df), batch_size)
    ]
    for batch_df in tqdm(batches_df, desc="Processing batches"):
        batch_df = batch_df.reset_index(drop=True)

        code_embeddings_batch = encoder.encode(batch_df["code_contents"].tolist())

        points = [
            models.PointStruct(
                id=str(row["code_uuid"]),
                vector=code_embeddings_batch[index].tolist(),
                payload={
                    "clones": row["clones"],
                    "contents": row["code_contents"],
                    "func_id": row["func_id"],
                    "is_base64": False,
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
    driver (QdrantClient): The Qdrant client used to interact with the search system.
    encoder (SentenceTransformer): The encoder used to transform code snippets into embeddings.
    collection_name (str): The name of the collection to search within.
    dataset (pd.DataFrame): The dataset containing code snippets and their metadata.
    num_samples (int, optional): The number of samples to use for benchmarking. Default is 10.
    k (int, optional): The number of top results to consider for each query. Default is 10.

    Returns:
    tuple: A tuple containing the following metrics:
        - success_rate (float): The proportion of queries that retrieved at least one relevant result.
        - mean_precision_at_k (float): The average precision at k across all queries.
        - mrr (float): The mean reciprocal rank of the first relevant result.
        - map_at_k (float): The mean average precision at k across all queries.
    """
    samples = dataset.sample(num_samples).reset_index(drop=True)

    total_success = 0
    total_precision = 0
    reciprocal_ranks = []
    average_precisions = []

    # New: Store per-query statistics
    query_stats = []
    detailed_results = []

    for index, sample in tqdm(
        samples.iterrows(), total=num_samples, desc=f"Benchmarking -> {collection_name}"
    ):
        code_to_search = sample["code_contents"]
        code_id = str(sample["code_uuid"])
        clones = set(sample["clones"])

        results = search(
            driver=driver,
            encoder=encoder,
            collection_name=collection_name,
            query=code_to_search,
            limit=k,
        )
        # Exclude the query code itself
        results = [result for result in results if result["id"] != code_id]

        # Collect scores for clones and non-clones
        clone_scores = []
        non_clone_scores = []

        for result in results:
            is_clone = result["payload"]["func_id"] in clones
            score = result["score"]

            if is_clone:
                clone_scores.append(score)
            else:
                non_clone_scores.append(score)

            detailed_results.append(
                {
                    "query_id": code_id,
                    "result_id": result["id"],
                    "result_func_id": result["payload"]["func_id"],
                    "similarity_score": score,
                    "is_clone": is_clone,
                }
            )

        # Calculate average scores for this query
        avg_clone_score = sum(clone_scores) / len(clone_scores) if clone_scores else 0
        avg_non_clone_score = (
            sum(non_clone_scores) / len(non_clone_scores) if non_clone_scores else 0
        )

        query_stats.append(
            {
                "query_id": code_id,
                "avg_clone_score": avg_clone_score,
                "avg_non_clone_score": avg_non_clone_score,
                "num_clones_found": len(clone_scores),
                "num_non_clones_found": len(non_clone_scores),
            }
        )

        retrieved_func_ids = [result["payload"]["func_id"] for result in results]
        retrieved_clones = [
            1 if func_id in clones else 0 for func_id in retrieved_func_ids
        ]

        num_retrieved_clones = sum(retrieved_clones)

        # Success Rate Calculation
        if num_retrieved_clones > 0:
            total_success += 1

        # Precision at k
        precision = num_retrieved_clones / k
        total_precision += precision

        # Reciprocal Rank
        try:
            first_relevant_rank = retrieved_clones.index(1) + 1  # ranks start at 1
            reciprocal_rank = 1 / first_relevant_rank
        except ValueError:
            reciprocal_rank = 0  # no relevant item found
        reciprocal_ranks.append(reciprocal_rank)

        # Average Precision at k
        cumulative_precision = 0
        relevant_items = 0
        precisions_at_relevant = []
        for i, is_relevant in enumerate(retrieved_clones, start=1):
            if is_relevant:
                relevant_items += 1
                cumulative_precision = relevant_items / i
                precisions_at_relevant.append(cumulative_precision)
        if precisions_at_relevant:
            average_precision = sum(precisions_at_relevant) / min(len(clones), k)
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
        "global_avg_clone_score": query_stats_df["avg_clone_score"].mean(),
        "global_avg_non_clone_score": query_stats_df["avg_non_clone_score"].mean(),
    }

    print(f"Success Rate at {k}: {success_rate:.4f}")
    print(f"Mean Precision at {k}: {mean_precision_at_k:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Mean Average Precision at {k} (MAP@{k}): {map_at_k:.4f}")
    print(f"Global Average Clone Score: {overall_stats['global_avg_clone_score']:.4f}")
    print(
        f"Global Average Non-Clone Score: {overall_stats['global_avg_non_clone_score']:.4f}"
    )

    return success_rate, mean_precision_at_k, mrr, map_at_k, results_df, query_stats_df
