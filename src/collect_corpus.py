import json
import os
import sys

import arxiv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def build_query():
    cat_clause = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    return f"({cat_clause}) AND {config.ARXIV_QUERY}"


def collect():
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=build_query(),
        max_results=config.MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records = []
    seen_ids = set()
    for result in client.results(search):
        doc_id = result.get_short_id()
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        record = {
            "doc_id": doc_id,
            "title": result.title.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip().replace("\n", " "),
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d"),
            "primary_category": result.primary_category,
            "categories": result.categories,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
        }
        records.append(record)

    print(f"Collected {len(records)} papers")

    with open(config.RAW_DATA_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved to {config.RAW_DATA_PATH}")


if __name__ == "__main__":
    collect()
