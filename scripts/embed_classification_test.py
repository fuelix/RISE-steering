"""
Embed the negation classification test set (1,919 sentences) with text-embedding-3-large.

This produces the embedded version of the test set used in Table 9 / Appendix G
of the ICLR 2026 paper for downstream negation classification.

Usage:
    python scripts/embed_classification_test.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load .env from helm or bkup-helm if OPENAI_API_KEY not already set
for env_path in [Path.home() / "c/helm/.env", Path.home() / "c/bkup-helm/.env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


def batch_embed(client: OpenAI, texts: list[str], model: str, batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches using the OpenAI API."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model})"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    return all_embeddings


def main():
    data_dir = Path("data/negation_classification_test")
    input_path = data_dir / "negation_test_sentences.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    positive = data["positive_sentences"]  # 958 sentences without negation
    negative = data["negative_sentences"]  # 961 sentences with negation

    print(f"Loaded {len(positive)} positive + {len(negative)} negative = {len(positive) + len(negative)} sentences")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "text-embedding-3-large"

    # Embed all sentences in one pass
    all_sentences = positive + negative
    all_labels = [0] * len(positive) + [1] * len(negative)

    print(f"\nEmbedding {len(all_sentences)} sentences with {model}...")
    all_embeddings = batch_embed(client, all_sentences, model)

    print(f"Got {len(all_embeddings)} embeddings of dimension {len(all_embeddings[0])}")

    # Write as JSONL: one record per sentence
    output_path = data_dir / "negation_test_embedded.jsonl"
    with open(output_path, "w") as f:
        for sentence, embedding, label in zip(all_sentences, all_embeddings, all_labels):
            record = {
                "text": sentence,
                "embedding": embedding,
                "label": label,  # 0 = no negation, 1 = has negation
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(all_embeddings)} embedded sentences to {output_path}")
    print(f"  label=0 (no negation): {sum(1 for l in all_labels if l == 0)}")
    print(f"  label=1 (has negation): {sum(1 for l in all_labels if l == 1)}")


if __name__ == "__main__":
    main()
