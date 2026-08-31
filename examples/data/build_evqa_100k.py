import argparse
import glob
import hashlib
import io
import json
import os
import random

import ijson
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 100k E-VQA knowledge base with locally stored images."
    )
    parser.add_argument("--wiki-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output-dir", default="evqa_data_100k")
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--candidate-pool-size", type=int, default=5_000_000)
    parser.add_argument("--sampling-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_wiki_content(titles, texts):
    return "\n\n".join(
        f"{title.strip()}\n{text.strip()}" for title, text in zip(titles, texts)
    )


def get_temp_filename(url):
    digest = hashlib.md5(url.encode()).hexdigest()
    return f"temp_{digest}.jpg"


def load_test_entries(test_json_path):
    with open(test_json_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    entries = {}
    for item in test_data:
        for retrieval_item in item.get("retrieval", []):
            image_urls = retrieval_item.get("image_urls", [])
            if image_urls and image_urls[0] not in entries:
                entry = retrieval_item.copy()
                entry["temp_path"] = None
                entries[image_urls[0]] = entry
    return test_data, entries


def sample_wiki_entries(
    wiki_json_path,
    test_entries,
    candidate_pool_size,
    sampling_rate,
):
    candidates = {}
    with open(wiki_json_path, "rb") as file:
        for key, value in tqdm(ijson.kvitems(file, ""), desc="Scanning Wiki"):
            if len(candidates) + len(test_entries) >= candidate_pool_size:
                break
            if random.random() > sampling_rate:
                continue

            image_urls = value.get("image_urls", [])
            if not image_urls:
                continue

            image_url = image_urls[0]
            if image_url in test_entries or image_url in candidates:
                continue

            value["original_url_key"] = key
            value["temp_path"] = None
            candidates[image_url] = value
    return candidates

def save_matching_images(parquet_dir, images_dir, test_entries, wiki_candidates):
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    random.shuffle(parquet_files)

    test_urls = set(test_entries)
    wiki_urls = set(wiki_candidates)
    valid_wiki_urls = []

    for parquet_file in tqdm(parquet_files, desc="Scanning Parquet files"):
        try:
            frame = pd.read_parquet(
                parquet_file, columns=["image_url", "image"], engine="pyarrow"
            )
        except Exception:
            continue

        matched = frame[
            frame["image_url"].isin(test_urls)
            | frame["image_url"].isin(wiki_urls)
        ]
        for _, row in matched.iterrows():
            image_url = row["image_url"]
            image_data = row["image"]
            blob = (
                image_data
                if isinstance(image_data, bytes)
                else image_data.get("bytes")
                if isinstance(image_data, dict)
                else None
            )
            if not blob:
                continue

            target = test_entries if image_url in test_entries else wiki_candidates
            if target[image_url]["temp_path"] is not None:
                continue

            temp_name = get_temp_filename(image_url)
            temp_path = os.path.join(images_dir, temp_name)
            try:
                image = Image.open(io.BytesIO(blob))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(temp_path, format="JPEG")
            except Exception:
                continue

            target[image_url]["temp_path"] = temp_name
            if image_url in wiki_candidates:
                valid_wiki_urls.append(image_url)

    return valid_wiki_urls


def build_dataset(test_entries, wiki_candidates, valid_wiki_urls, target_count):
    candidates = [
        {"url": url, "data": data} for url, data in test_entries.items()
    ]

    remaining = max(0, target_count - len(candidates))
    valid_wiki_urls.sort()
    random.shuffle(valid_wiki_urls)
    candidates.extend(
        {"url": url, "data": wiki_candidates[url]}
        for url in valid_wiki_urls[:remaining]
    )
    random.shuffle(candidates)
    return candidates


def finalize_dataset(candidates, images_dir):
    dataset = {}
    url_to_id = {}

    for index, item in enumerate(tqdm(candidates, desc="Finalizing dataset")):
        unique_id = f"Q{index:06d}"
        image_url = item["url"]
        data = item["data"]
        temp_filename = data.get("temp_path")

        if temp_filename:
            old_path = os.path.join(images_dir, temp_filename)
            new_path = os.path.join(images_dir, f"{unique_id}.jpg")
            try:
                os.rename(old_path, new_path)
            except OSError:
                pass

        titles = data.get("section_titles", [])
        texts = data.get("section_texts", [])
        summary = f"{titles[0]}\n {texts[0]}" if titles and texts else texts[0] if texts else ""
        dataset[unique_id] = {
            "wikidata_id": unique_id,
            "wikipedia_title": data.get("title", ""),
            "wikipedia_content": format_wiki_content(titles, texts),
            "wikipedia_image_url": image_url,
            "wikipedia_summary": summary,
        }
        url_to_id[image_url] = unique_id

    return dataset, url_to_id


def format_test_data(test_data, url_to_id):
    formatted = []
    for item in test_data:
        new_item = item.copy()
        new_item["retrieval"] = [
            url_to_id[retrieval["image_urls"][0]]
            for retrieval in item.get("retrieval", [])
            if retrieval.get("image_urls")
            and retrieval["image_urls"][0] in url_to_id
        ]
        formatted.append(new_item)
    return formatted


def main():
    args = parse_args()
    random.seed(args.seed)

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    test_data, test_entries = load_test_entries(args.test_json)
    wiki_candidates = sample_wiki_entries(
        args.wiki_json,
        test_entries,
        args.candidate_pool_size,
        args.sampling_rate,
    )
    valid_wiki_urls = save_matching_images(
        args.parquet_dir,
        images_dir,
        test_entries,
        wiki_candidates,
    )
    candidates = build_dataset(
        test_entries,
        wiki_candidates,
        valid_wiki_urls,
        args.target_count,
    )
    dataset, url_to_id = finalize_dataset(candidates, images_dir)

    for temp_file in glob.glob(os.path.join(images_dir, "temp_*.jpg")):
        os.remove(temp_file)

    dataset_path = os.path.join(args.output_dir, "evqa_wiki_100k_data.json")
    formatted_test_path = os.path.join(args.output_dir, "evqa_test.json")
    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
    with open(formatted_test_path, "w", encoding="utf-8") as file:
        json.dump(format_test_data(test_data, url_to_id), file, indent=4, ensure_ascii=False)

    print(f"Saved {len(dataset)} entries to {args.output_dir}")


if __name__ == "__main__":
    main()
