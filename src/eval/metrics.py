def reciprocal_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    retrieved_set = set(retrieved_ids)
    hit = len(gold_set & retrieved_set)
    return hit / len(gold_set)


def hit_at_k(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    retrieved_set = set(retrieved_ids)
    return 1.0 if gold_set & retrieved_set else 0.0


def aggregate(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)
