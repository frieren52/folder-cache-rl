from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


LAYER_BOUNDARIES = (0, 5, 30, 300, 3600)


def demand_layer(first_delta_seconds: int) -> int:
    value = int(first_delta_seconds)
    if value < 0 or value >= 3600:
        raise ValueError(f"首次需求提前量不在[0,3600)：{value}")
    return int(np.searchsorted(np.asarray(LAYER_BOUNDARIES), value, side="right") - 1)


def _largest_remainder(capacities: Sequence[int], total: int, weights: Sequence[float]) -> list[int]:
    capacities = [max(0, int(value)) for value in capacities]
    allocation = [0] * len(capacities)
    remaining = min(int(total), sum(capacities))
    while remaining > 0:
        available = [capacity - used for capacity, used in zip(capacities, allocation)]
        active = [index for index, value in enumerate(available) if value > 0]
        if not active:
            break
        active_weight = sum(float(weights[index]) for index in active)
        if active_weight <= 0:
            shares = {index: remaining / len(active) for index in active}
        else:
            shares = {index: remaining * float(weights[index]) / active_weight for index in active}
        base = {index: min(available[index], int(np.floor(shares[index]))) for index in active}
        assigned = sum(base.values())
        for index, count in base.items():
            allocation[index] += count
        remaining -= assigned
        if remaining <= 0:
            break
        order = sorted(
            active,
            key=lambda index: (-(shares[index] - np.floor(shares[index])), index),
        )
        progressed = False
        for index in order:
            if remaining <= 0:
                break
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


@dataclass(frozen=True)
class PositiveSample:
    all_ids: np.ndarray
    sampled_ids: np.ndarray
    layers: np.ndarray
    weights: np.ndarray


def sample_positives(
    first_delta_by_id: Mapping[int, int],
    future_count_by_id: Mapping[int, int],
    seed: int,
    max_objects: int = 96,
    per_layer_target: int = 24,
    deterministic_hot: int = 8,
) -> PositiveSample:
    all_ids = np.asarray(sorted(int(value) for value in first_delta_by_id), dtype=np.int64)
    if len(all_ids) <= max_objects:
        layers = np.asarray([demand_layer(first_delta_by_id[int(value)]) for value in all_ids], dtype=np.int8)
        order = np.lexsort((all_ids, np.asarray([first_delta_by_id[int(value)] for value in all_ids])))
        return PositiveSample(all_ids, all_ids[order], layers[order], np.ones(len(all_ids), dtype=np.float32))
    layer_ids = [
        [int(value) for value in all_ids if demand_layer(first_delta_by_id[int(value)]) == layer]
        for layer in range(4)
    ]
    initial = [min(per_layer_target, len(values)) for values in layer_ids]
    remaining = max_objects - sum(initial)
    extra_capacity = [len(values) - used for values, used in zip(layer_ids, initial)]
    extra = _largest_remainder(extra_capacity, remaining, extra_capacity)
    quotas = [used + added for used, added in zip(initial, extra)]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    selected_ids: list[int] = []
    selected_layers: list[int] = []
    selected_weights: list[float] = []
    for layer, (values, quota) in enumerate(zip(layer_ids, quotas)):
        ranked = sorted(values, key=lambda path_index: (-int(future_count_by_id[path_index]), path_index))
        hot_count = min(deterministic_hot, quota, len(ranked))
        hot = ranked[:hot_count]
        rest = ranked[hot_count:]
        random_count = quota - hot_count
        chosen_random = (
            rng.choice(np.asarray(rest, dtype=np.int64), size=random_count, replace=False).tolist()
            if random_count
            else []
        )
        random_weight = (len(rest) / random_count) if random_count else 1.0
        selected_ids.extend(hot)
        selected_layers.extend([layer] * len(hot))
        selected_weights.extend([1.0] * len(hot))
        selected_ids.extend(int(value) for value in chosen_random)
        selected_layers.extend([layer] * len(chosen_random))
        selected_weights.extend([float(random_weight)] * len(chosen_random))
    order = sorted(
        range(len(selected_ids)),
        key=lambda pos: (first_delta_by_id[selected_ids[pos]], selected_ids[pos]),
    )
    return PositiveSample(
        all_ids,
        np.asarray([selected_ids[pos] for pos in order], dtype=np.int64),
        np.asarray([selected_layers[pos] for pos in order], dtype=np.int8),
        np.asarray([selected_weights[pos] for pos in order], dtype=np.float32),
    )


@dataclass(frozen=True)
class PairSample:
    positive_ids: np.ndarray
    negative_ids: np.ndarray
    positive_layers: np.ndarray
    weights: np.ndarray
    negative_sources: np.ndarray
    pool_sizes: np.ndarray


def sample_pairs(
    positives: PositiveSample,
    negative_pools: Sequence[Sequence[int]],
    ratios: Sequence[float],
    negatives_per_positive: int,
    seed: int,
) -> PairSample:
    positive_set = set(int(value) for value in positives.all_ids)
    pools = [sorted(set(int(value) for value in pool) - positive_set) for pool in negative_pools]
    target = len(positives.sampled_ids) * int(negatives_per_positive)
    quotas = _largest_remainder([len(pool) for pool in pools], target, ratios)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    selected: list[tuple[int, int]] = []
    for source, (pool, quota) in enumerate(zip(pools, quotas)):
        if quota:
            values = rng.choice(np.asarray(pool, dtype=np.int64), size=quota, replace=False)
            selected.extend((int(value), source) for value in values)
    rng.shuffle(selected)
    pair_positive: list[int] = []
    pair_negative: list[int] = []
    pair_layer: list[int] = []
    pair_weight: list[float] = []
    sources: list[int] = []
    per_positive = np.zeros(len(positives.sampled_ids), dtype=np.int64)
    cursor = 0
    for negative_id, source in selected:
        attempts = 0
        while attempts < len(per_positive) and per_positive[cursor] >= negatives_per_positive:
            cursor = (cursor + 1) % len(per_positive)
            attempts += 1
        if attempts == len(per_positive):
            break
        pair_positive.append(int(positives.sampled_ids[cursor]))
        pair_negative.append(negative_id)
        pair_layer.append(int(positives.layers[cursor]))
        pair_weight.append(float(positives.weights[cursor]))
        sources.append(source)
        per_positive[cursor] += 1
        cursor = (cursor + 1) % len(per_positive)
    return PairSample(
        np.asarray(pair_positive, dtype=np.int64),
        np.asarray(pair_negative, dtype=np.int64),
        np.asarray(pair_layer, dtype=np.int8),
        np.asarray(pair_weight, dtype=np.float32),
        np.asarray(sources, dtype=np.int8),
        np.asarray([len(pool) for pool in pools], dtype=np.int32),
    )
