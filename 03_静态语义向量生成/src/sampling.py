"""按详细设计构造语义、实例和最终表示三类三元组。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripletSets:
    semantic: list[tuple[str, str, str]]
    instance: list[tuple[int, int, int]]
    final: list[tuple[int, int, int]]


def _known_conflict(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> bool:
    """只有双方字段都有值且不同，才算已知冲突。"""
    return bool(left[field] and right[field] and left[field] != right[field])


def build_semantic_triplets(
    records: pd.DataFrame,
    seed: int,
    strong_positive_ratio: float,
    hard_negative_ratio: float,
) -> list[tuple[str, str, str]]:
    groups = (
        records.loc[records["is_semantic_supervised"]]
        .drop_duplicates("semantic_group_id")
        .sort_values("semantic_group_id")
        .reset_index(drop=True)
    )
    rows = {row.semantic_group_id: row._asdict() for row in groups.itertuples(index=False)}
    group_ids = sorted(rows)

    # 三种索引分别服务于强正、弱正和困难负样本查找。
    by_product_instrument_level: dict[tuple[str, str, str], list[str]] = {}
    by_product_instrument: dict[tuple[str, str], list[str]] = {}
    by_single_field: dict[tuple[str, str], set[str]] = {}
    for group_id, row in rows.items():
        if row["product"] and row["instrument"]:
            by_product_instrument.setdefault((row["product"], row["instrument"]), []).append(group_id)
        if row["product"] and row["instrument"] and row["level"]:
            key = (row["product"], row["instrument"], row["level"])
            by_product_instrument_level.setdefault(key, []).append(group_id)
        for field in ("product", "instrument", "level"):
            if row[field]:
                by_single_field.setdefault((field, row[field]), set()).add(group_id)

    rng = np.random.default_rng(seed)
    triplets: list[tuple[str, str, str]] = []
    for anchor_id in group_ids:
        anchor = rows[anchor_id]
        strong_key = (anchor["product"], anchor["instrument"], anchor["level"])
        strong = [value for value in by_product_instrument_level.get(strong_key, []) if value != anchor_id]
        broad = [
            value
            for value in by_product_instrument.get((anchor["product"], anchor["instrument"]), [])
            if value != anchor_id and not _known_conflict(anchor, rows[value], "level")
        ]
        weak = [value for value in broad if value not in strong]

        # 先按 70/30 选择强弱类型；指定类型为空时用另一类补足。
        prefer_strong = rng.random() < strong_positive_ratio
        positives = strong if prefer_strong else weak
        if not positives:
            positives = weak if prefer_strong else strong
        if not positives:
            continue
        positive_id = str(rng.choice(sorted(positives)))

        near_ids: set[str] = set()
        for field in ("product", "instrument", "level"):
            if anchor[field]:
                near_ids.update(by_single_field.get((field, anchor[field]), set()))
        hard = sorted(
            candidate
            for candidate in near_ids
            if candidate != anchor_id
            and candidate not in broad
            and (
                _known_conflict(anchor, rows[candidate], "product")
                or _known_conflict(anchor, rows[candidate], "instrument")
            )
        )
        random_negatives = [candidate for candidate in group_ids if candidate != anchor_id and candidate not in broad]
        prefer_hard = rng.random() < hard_negative_ratio
        negatives = hard if prefer_hard else random_negatives
        if not negatives:
            negatives = random_negatives if prefer_hard else hard
        if not negatives:
            continue
        triplets.append((anchor_id, positive_id, str(rng.choice(negatives))))
    return triplets


def build_instance_triplets(
    records: pd.DataFrame,
    count: int,
    seed: int,
    hard_negative_ratio: float,
) -> list[tuple[int, int, int]]:
    rows = {int(row.path_index): row._asdict() for row in records.itertuples(index=False)}
    path_ids = sorted(rows)
    by_group: dict[str, list[int]] = {}
    for path_index, row in rows.items():
        by_group.setdefault(row["semantic_group_id"], []).append(path_index)

    rng = np.random.default_rng(seed)
    anchors = rng.choice(path_ids, size=count, replace=count > len(path_ids))
    triplets: list[tuple[int, int, int]] = []
    for anchor_value in anchors:
        anchor_id = int(anchor_value)
        anchor = rows[anchor_id]
        # 困难负样本与锚点同语义组，但日期或来源至少一项不同。
        hard = [
            candidate
            for candidate in by_group[anchor["semantic_group_id"]]
            if candidate != anchor_id
            and (
                rows[candidate]["date_ordinal"] != anchor["date_ordinal"]
                or rows[candidate]["archive_root"] != anchor["archive_root"]
                or rows[candidate]["subsystem"] != anchor["subsystem"]
            )
        ]
        random_negatives = [
            candidate for candidate in path_ids if rows[candidate]["semantic_group_id"] != anchor["semantic_group_id"]
        ]
        prefer_hard = rng.random() < hard_negative_ratio
        negatives = hard if prefer_hard else random_negatives
        if not negatives:
            negatives = random_negatives if prefer_hard else hard
        if negatives:
            triplets.append((anchor_id, anchor_id, int(rng.choice(negatives))))
    return triplets


def build_final_triplets(
    records: pd.DataFrame,
    semantic_triplets: Sequence[tuple[str, str, str]],
    seed: int,
) -> list[tuple[int, int, int]]:
    reliable = records.loc[records["is_semantic_supervised"]]
    paths_by_group = {
        group_id: sorted(group["path_index"].astype(int).tolist())
        for group_id, group in reliable.groupby("semantic_group_id")
    }
    rng = np.random.default_rng(seed)
    return [
        tuple(int(rng.choice(paths_by_group[group_id])) for group_id in group_triplet)
        for group_triplet in semantic_triplets
    ]


def build_triplet_sets(records: pd.DataFrame, config: Mapping[str, Any], seed: int) -> TripletSets:
    sampling = config["sampling"]
    semantic = build_semantic_triplets(
        records,
        seed=seed,
        strong_positive_ratio=float(sampling["semantic_strong_positive_ratio"]),
        hard_negative_ratio=float(sampling["semantic_hard_negative_ratio"]),
    )
    # 每轮以语义锚点数为基准，从数量更大的目录集合抽取等量实例。
    instance = build_instance_triplets(
        records,
        count=len(semantic),
        seed=seed + 1,
        hard_negative_ratio=float(sampling["instance_hard_negative_ratio"]),
    )
    final = build_final_triplets(records, semantic, seed=seed + 2)
    count = min(len(semantic), len(instance), len(final))
    return TripletSets(semantic=semantic[:count], instance=instance[:count], final=final[:count])


def iter_triplet_batches(triplets: TripletSets, batch_size: int, seed: int | None = None):
    order = np.arange(len(triplets.semantic))
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        yield (
            [triplets.semantic[index] for index in indices],
            [triplets.instance[index] for index in indices],
            [triplets.final[index] for index in indices],
        )
