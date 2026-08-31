from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def load_token_classifier():
    module_path = Path(__file__).with_name("02_classify_path_tokens.py")
    spec = importlib.util.spec_from_file_location("path_token_classifier", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载Token分类器: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOKEN = load_token_classifier()


SCALAR_FIELDS = [
    "archive_root",
    "subsystem",
    "data_domain",
    "data_family",
    "platform",
    "satellite",
    "instrument",
    "data_level",
    "data_sublevel",
    "product",
    "aggregation_period",
    "projection",
    "resolution",
    "orbit_direction",
    "observe_year",
    "observe_date",
    "temporal_phase",
    "time_slot",
    "format_hint",
]

LIST_FIELDS = [
    "satellite_qualifiers",
    "region_types",
    "spatial_subregions",
    "processing_stages",
    "product_variants",
    "instance_ids",
    "extra_tokens",
]

PROCESSING_TOKENS = {
    "TEMPWORK",
    "MIPS",
    "ENGIN",
    "TAPE_RETRIEVE",
    "ARCHIVE",
    "EXCHANGE",
    "DSSCACHE",
    "PRODUCT",
    "MATCH",
}

TEMPORAL_PHASES = {"DAY", "NIGHT"}
SPATIAL_SUBREGIONS = {"SPOL", "NPOL"}
PRODUCT_VARIANTS = {"COMB", "ScSND", "GR", "SG", "ORBITIMG", "ARCOrbit", "DESCOrbit"}


@dataclass
class FieldValue:
    value: str
    raw_value: str
    source_depth: int
    confidence: float
    evidence: list[str]


@dataclass
class Conflict:
    field: str
    kept_value: str
    competing_value: str
    kept_confidence: float
    competing_confidence: float
    source_depth: int
    evidence: str


@dataclass
class ListValue:
    value: str
    raw_value: str
    source_depth: int
    confidence: float
    evidence: str


@dataclass
class ParseResult:
    path_index: int
    path: str
    access_count: int
    total_size_bytes: int
    parser_route: str
    route_supported: bool
    fields: dict[str, FieldValue] = field(default_factory=dict)
    lists: dict[str, list[ListValue]] = field(default_factory=lambda: defaultdict(list))
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assigned_depths: set[int] = field(default_factory=set)
    token_count: int = 0
    parse_status: str = "pending"

    def assign(
        self,
        field_name: str,
        raw_value: str,
        source_depth: int,
        confidence: float,
        evidence: str,
        normalized_value: str | None = None,
    ) -> None:
        value = normalized_value if normalized_value is not None else raw_value
        incoming = FieldValue(
            value=value,
            raw_value=raw_value,
            source_depth=source_depth,
            confidence=confidence,
            evidence=[evidence],
        )
        current = self.fields.get(field_name)
        if current is None:
            self.fields[field_name] = incoming
            self.assigned_depths.add(source_depth)
            return

        if current.value == incoming.value:
            current.confidence = max(current.confidence, incoming.confidence)
            if evidence not in current.evidence:
                current.evidence.append(evidence)
            self.assigned_depths.add(source_depth)
            return

        keep_incoming = incoming.confidence > current.confidence
        kept = incoming if keep_incoming else current
        competing = current if keep_incoming else incoming
        self.conflicts.append(
            Conflict(
                field=field_name,
                kept_value=kept.value,
                competing_value=competing.value,
                kept_confidence=kept.confidence,
                competing_confidence=competing.confidence,
                source_depth=source_depth,
                evidence=evidence,
            )
        )
        if keep_incoming:
            self.fields[field_name] = incoming
        self.assigned_depths.add(source_depth)

    def add_list(
        self,
        field_name: str,
        raw_value: str,
        source_depth: int,
        confidence: float,
        evidence: str,
        normalized_value: str | None = None,
    ) -> None:
        value = normalized_value if normalized_value is not None else raw_value
        existing = self.lists[field_name]
        if not any(item.value == value and item.source_depth == source_depth for item in existing):
            existing.append(
                ListValue(
                    value=value,
                    raw_value=raw_value,
                    source_depth=source_depth,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
        self.assigned_depths.add(source_depth)


def normalize_satellite(raw: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"(FY[34][A-H])(?:(CHN|IMG|SIMU))?", raw, re.IGNORECASE)
    if not match:
        return raw.upper(), []
    base = match.group(1).upper()
    qualifier = match.group(2)
    return base, [qualifier.upper()] if qualifier else []


def normalize_instrument(raw: str) -> str:
    upper = raw.upper()
    if upper in {"GNOSO", "GNOSX", "GNOSR"}:
        return "GNOS"
    if upper.endswith("SIMU") and len(upper) > 4:
        return upper[:-4]
    return upper


def normalize_level(raw: str) -> str:
    mapping = {"1A": "L1A", "1B": "L1B"}
    return mapping.get(raw.upper(), raw.upper())


def normalize_resolution(raw: str) -> str:
    upper = raw.upper()
    match = re.fullmatch(r"(\d+)(M|KM)", upper)
    if not match:
        return upper
    return f"{int(match.group(1))}{match.group(2)}"


def normalize_direction(raw: str) -> str:
    upper = raw.upper()
    if upper.startswith("ASCEND"):
        return "ASCEND"
    if upper.startswith("DESCEND"):
        return "DESCEND"
    return upper


def route_path(segments: list[str]) -> tuple[str, bool]:
    if len(segments) >= 3 and segments[:3] == ["FYDATAOUTSHARE", "DATAIOT", "FY3"]:
        return "outshare_dataiot_fy3", True
    if len(segments) >= 3 and segments[:3] == ["FYDATAOUTSHARE", "DATA", "FY3"]:
        return "outshare_data_fy3", True
    if len(segments) >= 3 and segments[:3] == ["FYDATAINSHARE", "BAKIOT", "FY3"]:
        return "inshare_bakiot_fy3", True
    if len(segments) >= 3 and segments[:3] == ["FYDATAOUTSHARE", "DATAIOT", "FYSIMU"]:
        return "outshare_dataiot_fysimu", True
    if (
        len(segments) >= 3
        and segments[:2] == ["FYDATAARCH", "DSSCACHE"]
        and re.fullmatch(r"FY3[FGH](?:CHN|IMG)?", segments[2], re.IGNORECASE)
    ):
        return "arch_dsscache_fy3fgh", True
    if (
        len(segments) >= 3
        and segments[:2] == ["FYDATAARCH", "DSSCACHE"]
        and re.fullmatch(r"FY3[DE](?:CHN|IMG)?", segments[2], re.IGNORECASE)
    ):
        return "arch_dsscache_fy3de", True
    if (
        len(segments) >= 3
        and segments[:2] == ["FYDATAARCH", "DSSCACHE"]
        and re.fullmatch(r"FY3[A-H]SIMU", segments[2], re.IGNORECASE)
    ):
        return "arch_dsscache_fy3simu", True
    if (
        len(segments) >= 4
        and segments[:3] == ["FYDATAARCH", "DSSCACHE", "TEMPWORK"]
        and re.fullmatch(r"FY3[A-H]", segments[3], re.IGNORECASE)
    ):
        return "arch_dsscache_tempwork_fy3", True
    if len(segments) >= 3 and segments[:2] == ["FYDATAARCH", "DSSCACHE"]:
        branch_routes = {
            "FY4B": "arch_dsscache_fy4b",
            "JPSS1": "arch_dsscache_jpss1",
            "METOPC": "arch_dsscache_metopc",
            "EXTSAT": "arch_dsscache_extsat",
            "RS": "arch_dsscache_rs",
            "DQ": "arch_dsscache_dq",
            "GF5A": "arch_dsscache_gf5a",
        }
        route = branch_routes.get(segments[2].upper())
        if route:
            return route, True
    root = segments[0] if segments else "empty"
    branch = segments[1] if len(segments) > 1 else "none"
    return f"generic_fallback:{root}/{branch}", False


def assign_archive_and_subsystem(result: ParseResult, segments: list[str]) -> None:
    result.assign("archive_root", segments[0], 0, 1.00, "root_position")
    if len(segments) > 1:
        result.assign("subsystem", segments[1], 1, 0.98, "depth1_partition_under_root")


def assign_satellite(result: ParseResult, raw: str, depth: int, confidence: float, evidence: str) -> None:
    normalized, qualifiers = normalize_satellite(raw)
    result.assign("satellite", raw, depth, confidence, evidence, normalized)
    for qualifier in qualifiers:
        result.add_list(
            "satellite_qualifiers",
            qualifier,
            depth,
            confidence,
            "satellite_suffix_qualifier",
        )


def classify_all(path_index: int, path: str, segments: list[str]) -> list[Any]:
    return [TOKEN.classify_occurrence(path_index, path, segments, depth) for depth in range(len(segments))]


def split_composite_token(
    result: ParseResult,
    raw: str,
    depth: int,
) -> bool:
    # 只处理有明确结构证据的复合Token，其他下划线Token保持原样。
    match = re.fullmatch(r"(ASCEND|DESCEND)_(\d{1,4}M|\d{1,3}KM)", raw, re.IGNORECASE)
    if match:
        result.assign(
            "orbit_direction",
            match.group(1),
            depth,
            0.92,
            "composite_direction_resolution_token",
            normalize_direction(match.group(1)),
        )
        result.assign(
            "resolution",
            match.group(2),
            depth,
            0.92,
            "composite_direction_resolution_token",
            normalize_resolution(match.group(2)),
        )
        return True

    match = re.fullmatch(r"(\d{1,4}M|\d{1,3}KM)_(JPG|PNG|HDF|NC)", raw, re.IGNORECASE)
    if match:
        result.assign(
            "resolution",
            match.group(1),
            depth,
            0.90,
            "composite_resolution_format_token",
            normalize_resolution(match.group(1)),
        )
        result.assign(
            "format_hint",
            match.group(2),
            depth,
            0.88,
            "composite_resolution_format_token",
            match.group(2).upper(),
        )
        return True

    match = re.fullmatch(r"(DAILY|WEEKLY|MONTHLY|\d+DAY)_(\d{1,4}M|\d{1,3}KM)", raw, re.IGNORECASE)
    if match:
        result.assign(
            "aggregation_period",
            match.group(1),
            depth,
            0.94,
            "composite_period_resolution_token",
            match.group(1).upper(),
        )
        result.assign(
            "resolution",
            match.group(2),
            depth,
            0.92,
            "composite_period_resolution_token",
            normalize_resolution(match.group(2)),
        )
        return True

    # 如ASCENDKu、DESCENDDC：方向语义确定，后缀只作通道/产品变体候选。
    match = re.fullmatch(r"(ASCEND|DESCEND)(KU|DC)", raw, re.IGNORECASE)
    if match:
        result.assign(
            "orbit_direction",
            match.group(1),
            depth,
            0.93,
            "composite_direction_channel_token",
            normalize_direction(match.group(1)),
        )
        result.add_list(
            "product_variants",
            match.group(2),
            depth,
            0.65,
            "unverified_channel_suffix_candidate",
            match.group(2).upper(),
        )
        return True

    match = re.fullmatch(r"(ARCORBIT|DESCORBIT)(KU|DC)", raw, re.IGNORECASE)
    if match:
        direction = "ASCEND" if match.group(1).upper() == "ARCORBIT" else "DESCEND"
        result.assign(
            "orbit_direction",
            direction,
            depth,
            0.82,
            "orbit_variant_direction_candidate",
            direction,
        )
        result.add_list(
            "product_variants",
            match.group(2),
            depth,
            0.65,
            "unverified_channel_suffix_candidate",
            match.group(2).upper(),
        )
        return True
    return False


def interpret_safe_extension_token(result: ParseResult, raw: str, depth: int) -> bool:
    """处理不依赖业务中文释义、仅由Token外形即可确认的扩展类型。"""
    upper = raw.upper()
    if re.fullmatch(r"\d{1,4}M|\d{1,3}KM", upper):
        result.assign("resolution", raw, depth, 0.96, "extended_resolution_regex", normalize_resolution(raw))
        return True
    if re.fullmatch(r"\d+DAY", upper):
        result.assign("aggregation_period", raw, depth, 0.90, "numeric_period_regex", upper)
        return True
    if re.fullmatch(r"\d+HOUR", upper):
        result.assign("time_slot", raw, depth, 0.90, "hour_slot_regex", upper)
        return True
    match = re.fullmatch(r"(\d+DAY)_(IMG)", upper)
    if match:
        result.assign("aggregation_period", match.group(1), depth, 0.88, "composite_period_content_token", match.group(1))
        result.assign("format_hint", match.group(2), depth, 0.72, "composite_period_content_token", match.group(2))
        return True
    if re.fullmatch(r"\d+LINE", upper):
        result.add_list("processing_stages", raw, depth, 0.74, "processing_line_count_candidate", upper)
        return True
    if re.fullmatch(r"\d{1,3}", upper):
        result.add_list("instance_ids", raw, depth, 0.70, "short_numeric_instance_candidate", upper)
        return True
    if re.fullmatch(r"\d{2}MVK", upper):
        result.add_list("product_variants", raw, depth, 0.66, "measurement_band_code_candidate", upper)
        return True
    if re.fullmatch(r"WRAD[A-Z]", upper):
        result.add_list("product_variants", raw, depth, 0.70, "instrument_product_variant_suffix", upper)
        return True
    if upper in {"OBC", "OBCCD"}:
        result.add_list("product_variants", raw, depth, 0.66, "stable_product_variant_candidate", upper)
        return True
    return False


def interpret_tail(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
    start_depth: int,
    allow_product: bool = True,
) -> None:
    for depth in range(start_depth, len(segments)):
        token = segments[depth]
        upper = token.upper()
        occurrence = occurrences[depth]

        if split_composite_token(result, token, depth):
            continue
        if interpret_safe_extension_token(result, token, depth):
            continue

        best_type = occurrence.best_type
        confidence = occurrence.confidence
        evidence = "|".join(occurrence.evidence)

        if occurrence.ambiguous:
            result.warnings.append(
                f"ambiguous_token@{depth}:{token}:{TOKEN.serialize_candidates(occurrence.candidates)}"
            )

        if best_type == "observe_date":
            result.assign("observe_date", token, depth, confidence, evidence)
        elif best_type == "year":
            result.assign("observe_year", token, depth, confidence, evidence)
        elif best_type == "resolution":
            result.assign("resolution", token, depth, confidence, evidence, normalize_resolution(token))
        elif best_type == "orbit_direction":
            result.assign("orbit_direction", token, depth, confidence, evidence, normalize_direction(token))
        elif best_type == "region_or_orbit_type":
            result.add_list("region_types", token, depth, confidence, evidence, upper)
        elif best_type == "aggregation_period":
            result.assign("aggregation_period", token, depth, confidence, evidence, upper)
        elif best_type == "projection":
            result.assign("projection", token, depth, confidence, evidence, upper)
        elif best_type == "projection_candidate":
            # 低置信投影候选保留为字段，但明确降低置信度并产生告警。
            result.assign("projection", token, depth, confidence, evidence, upper)
            result.warnings.append(f"unverified_projection_code@{depth}:{token}")
        elif best_type == "format_or_content_type":
            result.assign("format_hint", token, depth, confidence, evidence, upper)
        elif best_type == "numeric_instance_id":
            result.add_list("instance_ids", token, depth, confidence, evidence)
        elif best_type == "processing_stage_or_directory_role":
            result.add_list("processing_stages", token, depth, confidence, evidence)
        elif upper in TEMPORAL_PHASES:
            result.assign("temporal_phase", token, depth, 0.80, "day_night_phase_code", upper)
        elif upper in SPATIAL_SUBREGIONS:
            result.add_list("spatial_subregions", token, depth, 0.78, "polar_subregion_candidate", upper)
        elif token in PRODUCT_VARIANTS or upper in {item.upper() for item in PRODUCT_VARIANTS}:
            result.add_list("product_variants", token, depth, 0.72, "known_product_variant_candidate")
        elif best_type == "product_or_business_category" and allow_product and "product" not in result.fields:
            result.assign("product", token, depth, confidence, evidence, upper)
        elif best_type == "data_level":
            normalized_level = normalize_level(token)
            if "data_level" in result.fields:
                if result.fields["data_level"].value == normalized_level:
                    result.assigned_depths.add(depth)
                else:
                    result.assign(
                        "data_sublevel",
                        token,
                        depth,
                        0.82,
                        "level_like_token_after_primary_level",
                        normalized_level,
                    )
            else:
                result.assign("data_level", token, depth, confidence, evidence, normalized_level)
        else:
            # 如果产品尚未确定，等级之后的第一个无强类型Token可作为低置信产品候选。
            previous_is_level = depth > 0 and segments[depth - 1].upper() in TOKEN.DATA_LEVELS
            if allow_product and "product" not in result.fields and previous_is_level:
                result.assign(
                    "product",
                    token,
                    depth,
                    0.62,
                    "untyped_first_token_after_level",
                    upper,
                )
            else:
                result.add_list(
                    "extra_tokens",
                    token,
                    depth,
                    confidence,
                    "unresolved_tail_token",
                )


def parse_outshare_fy3(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    if len(segments) < 6:
        result.warnings.append("path_too_short_for_outshare_fy3_core")
        return
    result.assign("data_domain", segments[2], 2, 1.00, "validated_outshare_domain_slot")
    assign_satellite(result, segments[3], 3, 1.00, "validated_outshare_satellite_slot")
    result.assign(
        "instrument",
        segments[4],
        4,
        0.98,
        "validated_outshare_instrument_slot",
        normalize_instrument(segments[4]),
    )
    result.assign(
        "data_level",
        segments[5],
        5,
        1.00,
        "validated_outshare_level_slot",
        normalize_level(segments[5]),
    )
    interpret_tail(result, segments, occurrences, 6, allow_product=True)


def parse_inshare_fy3(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    if len(segments) < 6:
        result.warnings.append("path_too_short_for_inshare_fy3_core")
        return
    result.assign("data_domain", segments[2], 2, 0.99, "validated_inshare_domain_slot")
    assign_satellite(result, segments[3], 3, 0.99, "validated_inshare_satellite_slot")
    result.assign(
        "instrument",
        segments[4],
        4,
        0.96,
        "validated_inshare_instrument_slot",
        normalize_instrument(segments[4]),
    )
    result.assign(
        "data_level",
        segments[5],
        5,
        0.99,
        "validated_inshare_level_slot",
        normalize_level(segments[5]),
    )
    interpret_tail(result, segments, occurrences, 6, allow_product=True)


def parse_arch_dsscache_fy3(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    assign_satellite(result, segments[2], 2, 0.99, "dsscache_fy3_satellite_slot")

    cursor = 3
    while cursor < len(segments) and segments[cursor].upper() in {item.upper() for item in PROCESSING_TOKENS}:
        result.add_list(
            "processing_stages",
            segments[cursor],
            cursor,
            0.95,
            "dsscache_processing_prefix_state",
        )
        cursor += 1

    if cursor < len(segments):
        token = segments[cursor]
        next_token = segments[cursor + 1] if cursor + 1 < len(segments) else None
        occurrence = occurrences[cursor]
        looks_like_instrument = (
            occurrence.best_type == "instrument"
            or (next_token is not None and next_token.upper() in TOKEN.DATA_LEVELS)
        )
        if looks_like_instrument:
            result.assign(
                "instrument",
                token,
                cursor,
                max(0.90, occurrence.confidence),
                "dsscache_instrument_state_before_level_or_codebook",
                normalize_instrument(token),
            )
            cursor += 1

    if cursor < len(segments) and segments[cursor].upper() in TOKEN.DATA_LEVELS:
        result.assign(
            "data_level",
            segments[cursor],
            cursor,
            0.99,
            "explicit_level_after_dsscache_instrument_state",
            normalize_level(segments[cursor]),
        )
        cursor += 1

    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_outshare_fysimu(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    """FYSIMU路径没有等级槽位：FYSIMU/模拟平台/模拟仪器/年/日期。"""
    assign_archive_and_subsystem(result, segments)
    if len(segments) < 5:
        result.warnings.append("path_too_short_for_fysimu_core")
        return
    result.assign("data_domain", segments[2], 2, 1.00, "validated_fysimu_domain_slot", "FYSIMU")
    assign_satellite(result, segments[3], 3, 0.98, "validated_fysimu_platform_slot")
    result.assign(
        "instrument",
        segments[4],
        4,
        0.96,
        "validated_fysimu_instrument_slot",
        normalize_instrument(segments[4]),
    )
    interpret_tail(result, segments, occurrences, 5, allow_product=False)


def parse_arch_fy3simu(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    assign_satellite(result, segments[2], 2, 0.97, "validated_dsscache_simulated_platform_slot")
    if len(segments) > 3:
        result.assign(
            "instrument",
            segments[3],
            3,
            0.94,
            "validated_dsscache_simulated_instrument_slot",
            normalize_instrument(segments[3]),
        )
    interpret_tail(result, segments, occurrences, 4, allow_product=False)


def parse_arch_tempwork_fy3(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    result.add_list("processing_stages", segments[2], 2, 0.97, "validated_processing_prefix_before_platform")
    assign_satellite(result, segments[3], 3, 0.97, "validated_platform_after_processing_prefix")
    if len(segments) > 4:
        result.assign(
            "instrument",
            segments[4],
            4,
            0.93,
            "validated_instrument_after_platform",
            normalize_instrument(segments[4]),
        )
    cursor = 5
    if cursor < len(segments) and segments[cursor].upper() in TOKEN.DATA_LEVELS:
        result.assign(
            "data_level", segments[cursor], cursor, 0.95, "explicit_level_after_instrument", normalize_level(segments[cursor])
        )
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_arch_direct_platform(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    """FY4B、JPSS1、METOPC的直接平台分支。"""
    assign_archive_and_subsystem(result, segments)
    assign_satellite(result, segments[2], 2, 0.98, "validated_dsscache_platform_slot")
    cursor = 3
    while cursor < len(segments) and segments[cursor].upper() in {item.upper() for item in PROCESSING_TOKENS}:
        result.add_list("processing_stages", segments[cursor], cursor, 0.95, "platform_processing_prefix_state")
        cursor += 1
    if cursor < len(segments):
        result.assign(
            "instrument",
            segments[cursor],
            cursor,
            0.94,
            "validated_instrument_slot_under_platform",
            normalize_instrument(segments[cursor]),
        )
        cursor += 1
    if cursor < len(segments) and (
        segments[cursor].upper() in TOKEN.DATA_LEVELS
        or re.fullmatch(r"L\d[A-Z]?", segments[cursor], re.IGNORECASE)
    ):
        result.assign(
            "data_level",
            segments[cursor],
            cursor,
            0.96,
            "explicit_level_after_platform_instrument",
            normalize_level(segments[cursor]),
        )
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_arch_rs(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    """RS/平台/仪器/等级/产品/..."""
    assign_archive_and_subsystem(result, segments)
    result.assign("data_domain", segments[2], 2, 0.98, "validated_rs_domain_slot", "RS")
    if len(segments) > 3:
        assign_satellite(result, segments[3], 3, 0.94, "validated_rs_platform_slot")
    if len(segments) > 4:
        result.assign(
            "instrument", segments[4], 4, 0.92, "validated_rs_instrument_slot", normalize_instrument(segments[4])
        )
    cursor = 5
    if cursor < len(segments) and segments[cursor].upper() in TOKEN.DATA_LEVELS:
        result.assign(
            "data_level", segments[cursor], cursor, 0.95, "validated_rs_level_slot", normalize_level(segments[cursor])
        )
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_arch_extsat(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    """EXTSAT下SATE与NAFP结构不同，仅解析能由上下文确认的槽位。"""
    assign_archive_and_subsystem(result, segments)
    result.assign("data_domain", segments[2], 2, 0.98, "validated_extsat_domain_slot", "EXTSAT")
    if len(segments) < 4:
        return
    family = segments[3].upper()
    result.assign("data_family", segments[3], 3, 0.92, "extsat_family_slot", family)
    if len(segments) < 5:
        return
    cursor = 4
    if family == "SATE":
        assign_satellite(result, segments[cursor], cursor, 0.91, "extsat_satellite_platform_slot")
    else:
        result.assign("platform", segments[cursor], cursor, 0.82, "extsat_non_satellite_provider_or_model_slot", segments[cursor].upper())
    cursor += 1

    if cursor < len(segments):
        next_token = segments[cursor + 1] if cursor + 1 < len(segments) else ""
        occurrence = occurrences[cursor]
        is_instrument = occurrence.best_type == "instrument" or next_token.upper() in TOKEN.DATA_LEVELS
        if family == "SATE" and is_instrument:
            result.assign(
                "instrument",
                segments[cursor],
                cursor,
                max(0.88, occurrence.confidence),
                "extsat_instrument_before_level_or_codebook",
                normalize_instrument(segments[cursor]),
            )
            cursor += 1
        elif "product" not in result.fields:
            result.assign("product", segments[cursor], cursor, 0.72, "extsat_direct_product_or_model_family", segments[cursor].upper())
            cursor += 1

    if cursor < len(segments) and segments[cursor].upper() in TOKEN.DATA_LEVELS:
        result.assign(
            "data_level", segments[cursor], cursor, 0.94, "extsat_explicit_level", normalize_level(segments[cursor])
        )
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_arch_dq(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    result.assign("data_domain", segments[2], 2, 0.98, "validated_dq_domain_slot", "DQ")
    if len(segments) > 3:
        result.assign("data_family", segments[3], 3, 0.86, "dq_collection_slot", segments[3].upper())
    if len(segments) > 4:
        result.assign(
            "instrument", segments[4], 4, 0.90, "dq_instrument_slot", normalize_instrument(segments[4])
        )
    cursor = 5
    if cursor < len(segments):
        result.assign(
            "data_level", segments[cursor], cursor, 0.88, "dq_level_slot", normalize_level(segments[cursor])
        )
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=True)


def parse_arch_gf5a(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    assign_satellite(result, segments[2], 2, 0.96, "validated_gf5a_platform_slot")
    if len(segments) > 3:
        result.assign(
            "instrument", segments[3], 3, 0.94, "validated_gf5a_instrument_slot", normalize_instrument(segments[3])
        )
    cursor = 4
    if cursor < len(segments) and occurrences[cursor].best_type != "observe_date":
        result.assign("product", segments[cursor], cursor, 0.78, "gf5a_product_slot", segments[cursor].upper())
        cursor += 1
    interpret_tail(result, segments, occurrences, cursor, allow_product=False)


def parse_generic_fallback(
    result: ParseResult,
    segments: list[str],
    occurrences: list[Any],
) -> None:
    assign_archive_and_subsystem(result, segments)
    for depth in range(2, len(segments)):
        token = segments[depth]
        occurrence = occurrences[depth]
        best_type = occurrence.best_type
        evidence = "|".join(occurrence.evidence)

        if split_composite_token(result, token, depth):
            continue
        if interpret_safe_extension_token(result, token, depth):
            continue
        if best_type == "satellite":
            assign_satellite(result, token, depth, occurrence.confidence, evidence)
        elif best_type == "platform" and "satellite" not in result.fields:
            result.assign("satellite", token, depth, occurrence.confidence, evidence, token.upper())
        elif best_type == "instrument":
            result.assign(
                "instrument",
                token,
                depth,
                occurrence.confidence,
                evidence,
                normalize_instrument(token),
            )
        elif best_type == "data_level":
            normalized_level = normalize_level(token)
            if "data_level" in result.fields:
                if result.fields["data_level"].value == normalized_level:
                    result.assigned_depths.add(depth)
                else:
                    result.assign(
                        "data_sublevel",
                        token,
                        depth,
                        0.78,
                        "generic_hierarchical_level_candidate",
                        normalized_level,
                    )
            else:
                result.assign("data_level", token, depth, occurrence.confidence, evidence, normalized_level)
        elif best_type == "data_domain":
            result.assign("data_domain", token, depth, occurrence.confidence, evidence)
        elif best_type == "observe_date":
            result.assign("observe_date", token, depth, occurrence.confidence, evidence)
        elif best_type == "year":
            result.assign("observe_year", token, depth, occurrence.confidence, evidence)
        elif best_type == "resolution":
            result.assign(
                "resolution",
                token,
                depth,
                occurrence.confidence,
                evidence,
                normalize_resolution(token),
            )
        elif best_type == "orbit_direction":
            result.assign(
                "orbit_direction",
                token,
                depth,
                occurrence.confidence,
                evidence,
                normalize_direction(token),
            )
        elif best_type == "region_or_orbit_type":
            result.add_list("region_types", token, depth, occurrence.confidence, evidence, token.upper())
        elif best_type == "aggregation_period":
            result.assign("aggregation_period", token, depth, occurrence.confidence, evidence, token.upper())
        elif best_type == "projection":
            result.assign("projection", token, depth, occurrence.confidence, evidence, token.upper())
        elif best_type == "format_or_content_type":
            result.assign("format_hint", token, depth, occurrence.confidence, evidence, token.upper())
        elif best_type == "numeric_instance_id":
            result.add_list("instance_ids", token, depth, occurrence.confidence, evidence)
        elif best_type == "processing_stage_or_directory_role":
            result.add_list("processing_stages", token, depth, occurrence.confidence, evidence)
        else:
            result.add_list("extra_tokens", token, depth, occurrence.confidence, "generic_fallback_unresolved")


def finalize_result(result: ParseResult, segments: list[str]) -> None:
    year_field = result.fields.get("observe_year")
    date_field = result.fields.get("observe_date")
    if year_field and date_field and not date_field.value.startswith(year_field.value):
        result.conflicts.append(
            Conflict(
                field="observe_year_vs_date",
                kept_value=date_field.value[:4],
                competing_value=year_field.value,
                kept_confidence=date_field.confidence,
                competing_confidence=year_field.confidence,
                source_depth=year_field.source_depth,
                evidence="year_directory_conflicts_with_observe_date",
            )
        )

    required_by_route = {
        "outshare_dataiot_fy3": {"archive_root", "subsystem", "data_domain", "satellite", "instrument", "data_level"},
        "outshare_data_fy3": {"archive_root", "subsystem", "data_domain", "satellite", "instrument", "data_level"},
        "inshare_bakiot_fy3": {"archive_root", "subsystem", "data_domain", "satellite", "instrument", "data_level"},
        "arch_dsscache_fy3fgh": {"archive_root", "subsystem", "satellite"},
        "arch_dsscache_fy3de": {"archive_root", "subsystem", "satellite"},
        "arch_dsscache_fy3simu": {"archive_root", "subsystem", "satellite", "instrument"},
        "arch_dsscache_tempwork_fy3": {"archive_root", "subsystem", "satellite", "instrument"},
        "outshare_dataiot_fysimu": {"archive_root", "subsystem", "data_domain", "satellite", "instrument"},
        "arch_dsscache_fy4b": {"archive_root", "subsystem", "satellite", "instrument"},
        "arch_dsscache_jpss1": {"archive_root", "subsystem", "satellite", "instrument"},
        "arch_dsscache_metopc": {"archive_root", "subsystem", "satellite", "instrument"},
        "arch_dsscache_extsat": {"archive_root", "subsystem", "data_domain", "data_family"},
        "arch_dsscache_rs": {"archive_root", "subsystem", "data_domain", "satellite", "instrument"},
        "arch_dsscache_dq": {"archive_root", "subsystem", "data_domain", "data_family", "instrument"},
        "arch_dsscache_gf5a": {"archive_root", "subsystem", "satellite", "instrument"},
    }

    if not result.route_supported:
        result.parse_status = "fallback"
    elif result.conflicts:
        result.parse_status = "conflict"
    else:
        missing = required_by_route.get(result.parser_route, set()) - set(result.fields)
        if missing:
            result.parse_status = "partial"
            result.warnings.append("missing_required_fields:" + ",".join(sorted(missing)))
        elif result.lists.get("extra_tokens"):
            result.parse_status = "parsed_with_extras"
        else:
            result.parse_status = "parsed"

    # extra_tokens也算保留，不算已理解；assigned_token_ratio只计算具有语义解释的Token。
    extra_depths = {item.source_depth for item in result.lists.get("extra_tokens", [])}
    result.assigned_depths -= extra_depths
    result.token_count = len(segments)


def parse_row(path_index: int, row: dict[str, Any]) -> ParseResult:
    segments = row["segments"]
    route, supported = route_path(segments)
    result = ParseResult(
        path_index=path_index,
        path=row["path"],
        access_count=row["access_count"],
        total_size_bytes=row["total_size_bytes"],
        parser_route=route,
        route_supported=supported,
    )
    occurrences = classify_all(path_index, row["path"], segments)

    if route in {"outshare_dataiot_fy3", "outshare_data_fy3"}:
        parse_outshare_fy3(result, segments, occurrences)
    elif route == "inshare_bakiot_fy3":
        parse_inshare_fy3(result, segments, occurrences)
    elif route == "outshare_dataiot_fysimu":
        parse_outshare_fysimu(result, segments, occurrences)
    elif route in {"arch_dsscache_fy3fgh", "arch_dsscache_fy3de"}:
        parse_arch_dsscache_fy3(result, segments, occurrences)
    elif route == "arch_dsscache_fy3simu":
        parse_arch_fy3simu(result, segments, occurrences)
    elif route == "arch_dsscache_tempwork_fy3":
        parse_arch_tempwork_fy3(result, segments, occurrences)
    elif route in {"arch_dsscache_fy4b", "arch_dsscache_jpss1", "arch_dsscache_metopc"}:
        parse_arch_direct_platform(result, segments, occurrences)
    elif route == "arch_dsscache_extsat":
        parse_arch_extsat(result, segments, occurrences)
    elif route == "arch_dsscache_rs":
        parse_arch_rs(result, segments, occurrences)
    elif route == "arch_dsscache_dq":
        parse_arch_dq(result, segments, occurrences)
    elif route == "arch_dsscache_gf5a":
        parse_arch_gf5a(result, segments, occurrences)
    else:
        parse_generic_fallback(result, segments, occurrences)

    finalize_result(result, segments)
    return result


def field_value(result: ParseResult, name: str) -> str:
    item = result.fields.get(name)
    return item.value if item else ""


def field_raw(result: ParseResult, name: str) -> str:
    item = result.fields.get(name)
    return item.raw_value if item else ""


def list_values(result: ParseResult, name: str) -> str:
    return "|".join(item.value for item in result.lists.get(name, []))


def result_to_flat_row(result: ParseResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path_index": result.path_index,
        "path": result.path,
        "access_count": result.access_count,
        "total_size_bytes": result.total_size_bytes,
        "parser_route": result.parser_route,
        "route_supported": int(result.route_supported),
        "parse_status": result.parse_status,
        "token_count": result.token_count,
        "semantic_token_count": len(result.assigned_depths),
        "semantic_token_ratio": round(len(result.assigned_depths) / max(1, result.token_count), 4),
    }
    for name in SCALAR_FIELDS:
        row[name] = field_value(result, name)
        row[f"{name}_raw"] = field_raw(result, name)
    for name in LIST_FIELDS:
        row[name] = list_values(result, name)
    row["field_metadata_json"] = json.dumps(
        {name: asdict(value) for name, value in result.fields.items()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row["list_metadata_json"] = json.dumps(
        {name: [asdict(value) for value in values] for name, values in result.lists.items()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row["conflicts_json"] = json.dumps(
        [asdict(item) for item in result.conflicts], ensure_ascii=False, separators=(",", ":")
    )
    row["warnings_json"] = json.dumps(result.warnings, ensure_ascii=False, separators=(",", ":"))
    return row


def run_self_tests() -> None:
    samples = [
        {
            "path": "/FYDATAOUTSHARE/DATAIOT/FY3/FY3H/MERSI/L2L3/OCA/ORBIT/010KM/2026/20260331",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/FY3H/TEMPWORK/MERSI/L2L3/OCA/ORBIT/010KM/20260518",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/FY3H/MIPS/GLL/HDF/0250M/20260518",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAOUTSHARE/DATA/FY3/FY3D/MWRI/L2L3/SIC/ORBIT/ASCEND_6250M/2026/20260520",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAOUTSHARE/DATA/FY3/FY3E/HIRAS/L2L3/L1C/DESCEND/2026/20260522",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAOUTSHARE/DATAIOT/FYSIMU/FY3HSIMU/MWHSSIMU/2026/20260519",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/FY4B/HEPD/L2/HEPE/05MVk/20260516",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/JPSS1/TEMPWORK/VIIRS/L1/PRO/375M/20260522",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/RS/FY4B/AGRI/IMG/DISK/4KM/20260519",
            "access_count": 1,
            "total_size_bytes": 1,
        },
        {
            "path": "/FYDATAARCH/DSSCACHE/EXTSAT/SATE/H09/AHI/L2/PAR/20260517",
            "access_count": 1,
            "total_size_bytes": 1,
        },
    ]
    for sample in samples:
        sample["segments"] = sample["path"].strip("/").split("/")

    parsed = [parse_row(index, sample) for index, sample in enumerate(samples)]
    assert field_value(parsed[0], "satellite") == "FY3H"
    assert field_value(parsed[0], "instrument") == "MERSI"
    assert field_value(parsed[0], "product") == "OCA"
    assert field_value(parsed[0], "resolution") == "10KM"
    assert field_value(parsed[0], "observe_date") == "20260331"

    assert list_values(parsed[1], "processing_stages") == "TEMPWORK"
    assert field_value(parsed[1], "instrument") == "MERSI"
    assert field_value(parsed[1], "data_level") == "L2L3"

    assert "instrument" not in parsed[2].fields
    assert field_value(parsed[2], "projection") == "GLL"
    assert field_value(parsed[2], "format_hint") == "HDF"
    assert field_value(parsed[2], "resolution") == "250M"

    assert field_value(parsed[3], "orbit_direction") == "ASCEND"
    assert field_value(parsed[3], "resolution") == "6250M"

    assert field_value(parsed[4], "data_level") == "L2L3"
    assert field_value(parsed[4], "data_sublevel") == "L1C"
    assert not parsed[4].conflicts

    assert field_value(parsed[5], "satellite") == "FY3H"
    assert list_values(parsed[5], "satellite_qualifiers") == "SIMU"
    assert field_value(parsed[5], "instrument") == "MWHS"

    assert field_value(parsed[6], "satellite") == "FY4B"
    assert field_value(parsed[6], "instrument") == "HEPD"
    assert field_value(parsed[6], "product") == "HEPE"
    assert list_values(parsed[6], "product_variants") == "05MVK"

    assert list_values(parsed[7], "processing_stages") == "TEMPWORK"
    assert field_value(parsed[7], "instrument") == "VIIRS"
    assert field_value(parsed[7], "product") == "PRO"

    assert field_value(parsed[8], "data_domain") == "RS"
    assert field_value(parsed[8], "satellite") == "FY4B"
    assert field_value(parsed[8], "resolution") == "4KM"

    assert field_value(parsed[9], "data_family") == "SATE"
    assert field_value(parsed[9], "satellite") == "H09"
    assert field_value(parsed[9], "instrument") == "AHI"
    assert field_value(parsed[9], "product") == "PAR"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤3：路径分支路由与状态机语义解析")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-route", type=int, default=80)
    args = parser.parse_args()

    run_self_tests()
    rows = TOKEN.parse_input(args.input.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [parse_row(index, row) for index, row in enumerate(rows)]
    flat_rows = [result_to_flat_row(result) for result in results]

    with gzip.open(output_dir / "parsed_paths.csv.gz", "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    route_groups: dict[str, list[ParseResult]] = defaultdict(list)
    for result in results:
        route_groups[result.parser_route].append(result)

    route_summary_rows = []
    for route, group in sorted(route_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        statuses = Counter(item.parse_status for item in group)
        route_summary_rows.append(
            {
                "parser_route": route,
                "paths": len(group),
                "path_percentage": round(100.0 * len(group) / len(results), 4),
                "route_supported": int(group[0].route_supported),
                "parsed": statuses["parsed"],
                "parsed_with_extras": statuses["parsed_with_extras"],
                "partial": statuses["partial"],
                "conflict": statuses["conflict"],
                "fallback": statuses["fallback"],
                "mean_semantic_token_ratio": round(
                    sum(len(item.assigned_depths) / max(1, item.token_count) for item in group) / len(group), 4
                ),
            }
        )
    write_csv(output_dir / "route_summary.csv", route_summary_rows)

    field_coverage_rows = []
    all_fields = SCALAR_FIELDS + LIST_FIELDS
    for route, group in sorted(route_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        for name in all_fields:
            if name in SCALAR_FIELDS:
                count = sum(name in item.fields for item in group)
            else:
                count = sum(bool(item.lists.get(name)) for item in group)
            field_coverage_rows.append(
                {
                    "parser_route": route,
                    "field": name,
                    "paths_with_field": count,
                    "route_paths": len(group),
                    "coverage_percentage": round(100.0 * count / len(group), 4),
                }
            )
    write_csv(output_dir / "field_coverage_by_route.csv", field_coverage_rows)

    field_value_stats: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        for field_name, item in result.fields.items():
            key = (result.parser_route, field_name, item.value, item.raw_value)
            stats = field_value_stats.setdefault(key, {"count": 0, "examples": []})
            stats["count"] += 1
            if len(stats["examples"]) < 3 and result.path not in stats["examples"]:
                stats["examples"].append(result.path)
        for field_name, items in result.lists.items():
            if field_name == "extra_tokens":
                continue
            for item in items:
                key = (result.parser_route, field_name, item.value, item.raw_value)
                stats = field_value_stats.setdefault(key, {"count": 0, "examples": []})
                stats["count"] += 1
                if len(stats["examples"]) < 3 and result.path not in stats["examples"]:
                    stats["examples"].append(result.path)

    field_value_rows = []
    for (route, field_name, value, raw_value), stats in sorted(
        field_value_stats.items(), key=lambda item: (-item[1]["count"], item[0])
    ):
        field_value_rows.append(
            {
                "parser_route": route,
                "field": field_name,
                "value": value,
                "raw_value": raw_value,
                "occurrences": stats["count"],
                "example_paths": " || ".join(stats["examples"]),
            }
        )
    write_csv(output_dir / "field_value_catalog.csv", field_value_rows)

    conflict_rows = []
    for result in results:
        for conflict in result.conflicts:
            conflict_rows.append(
                {
                    "path_index": result.path_index,
                    "path": result.path,
                    "parser_route": result.parser_route,
                    **asdict(conflict),
                }
            )
    write_csv(
        output_dir / "field_conflicts.csv",
        conflict_rows,
        [
            "path_index",
            "path",
            "parser_route",
            "field",
            "kept_value",
            "competing_value",
            "kept_confidence",
            "competing_confidence",
            "source_depth",
            "evidence",
        ],
    )

    extra_stats: dict[str, dict[str, Any]] = {}
    for result in results:
        for item in result.lists.get("extra_tokens", []):
            stats = extra_stats.setdefault(
                item.raw_value,
                {"count": 0, "routes": Counter(), "depths": Counter(), "examples": []},
            )
            stats["count"] += 1
            stats["routes"][result.parser_route] += 1
            stats["depths"][item.source_depth] += 1
            if len(stats["examples"]) < 3 and result.path not in stats["examples"]:
                stats["examples"].append(result.path)

    extra_rows = []
    for token, stats in sorted(extra_stats.items(), key=lambda item: (-item[1]["count"], item[0])):
        extra_rows.append(
            {
                "token": token,
                "occurrences": stats["count"],
                "route_counts_json": json.dumps(stats["routes"], ensure_ascii=False, separators=(",", ":")),
                "depth_counts_json": json.dumps(stats["depths"], ensure_ascii=False, separators=(",", ":")),
                "example_paths": " || ".join(stats["examples"]),
            }
        )
    write_csv(
        output_dir / "unresolved_extra_tokens.csv",
        extra_rows,
        ["token", "occurrences", "route_counts_json", "depth_counts_json", "example_paths"],
    )

    randomizer = random.Random(20260820)
    sample_indices: set[int] = set()
    for route, group in route_groups.items():
        take = min(args.samples_per_route, len(group))
        sample_indices.update(item.path_index for item in randomizer.sample(group, take))
    sample_indices.update(item.path_index for item in results if item.conflicts)
    sample_indices.update(item.path_index for item in results if item.parse_status == "partial")
    sample_indices.update(item.path_index for item in results if item.lists.get("extra_tokens"))
    sample_indices = set(sorted(sample_indices)[:5000])

    result_by_index = {item.path_index: item for item in results}
    with (output_dir / "parsed_path_samples.jsonl").open("w", encoding="utf-8") as handle:
        for path_index in sorted(sample_indices):
            result = result_by_index[path_index]
            payload = {
                "path_index": path_index,
                "path": result.path,
                "parser_route": result.parser_route,
                "route_supported": result.route_supported,
                "parse_status": result.parse_status,
                "fields": {name: asdict(value) for name, value in result.fields.items()},
                "lists": {name: [asdict(value) for value in values] for name, values in result.lists.items()},
                "conflicts": [asdict(item) for item in result.conflicts],
                "warnings": result.warnings,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    status_counts = Counter(result.parse_status for result in results)
    supported_count = sum(result.route_supported for result in results)
    supported_conflicts = sum(result.route_supported and bool(result.conflicts) for result in results)
    supported_extras = sum(result.route_supported and bool(result.lists.get("extra_tokens")) for result in results)
    mean_ratio_supported = sum(
        len(result.assigned_depths) / max(1, result.token_count)
        for result in results
        if result.route_supported
    ) / max(1, supported_count)

    summary = {
        "input_paths": len(results),
        "supported_paths": supported_count,
        "supported_path_percentage": round(100.0 * supported_count / len(results), 4),
        "parse_status_counts": dict(status_counts),
        "supported_paths_with_conflicts": supported_conflicts,
        "supported_conflict_percentage": round(100.0 * supported_conflicts / max(1, supported_count), 4),
        "supported_paths_with_unresolved_extras": supported_extras,
        "supported_unresolved_extra_percentage": round(100.0 * supported_extras / max(1, supported_count), 4),
        "supported_mean_semantic_token_ratio": round(mean_ratio_supported, 4),
        "unique_unresolved_extra_tokens": len(extra_rows),
        "self_tests": "passed",
    }
    (output_dir / "step3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_table = "\n".join(
        f"| {row['parser_route']} | {row['paths']} | {row['path_percentage']}% | "
        f"{row['parsed']} | {row['parsed_with_extras']} | {row['partial']} | "
        f"{row['conflict']} | {row['fallback']} | {row['mean_semantic_token_ratio']} |"
        for row in route_summary_rows
    )
    extra_table = "\n".join(
        f"| {row['token']} | {row['occurrences']} | {row['route_counts_json']} | {row['example_paths'][:150]} |"
        for row in extra_rows[:30]
    ) or "| 无 | 0 | - | - |"
    conflict_table = "\n".join(
        f"| {row['field']} | {row['kept_value']} | {row['competing_value']} | {row['parser_route']} | {row['path'][:140]} |"
        for row in conflict_rows[:30]
    ) or "| 无 | - | - | - | - |"

    report = f"""# 步骤3：路径分支路由与状态机解析报告

## 1. 实现范围

本步骤为以下主流分支建立了专用解析器：

1. FYDATAOUTSHARE/DATAIOT/FY3
2. FYDATAOUTSHARE/DATA/FY3
3. FYDATAARCH/DSSCACHE/FY3F、FY3G、FY3H
4. FYDATAINSHARE/BAKIOT/FY3

其他分支使用保守通用解析器，只提取强正则和代码表字段，状态标记为fallback。

- 输入路径：{len(results)}
- 专用路由覆盖：{supported_count}（{summary['supported_path_percentage']}%）
- 专用路由字段冲突：{supported_conflicts}（{summary['supported_conflict_percentage']}%）
- 专用路由含未解释附加Token：{supported_extras}（{summary['supported_unresolved_extra_percentage']}%）
- 专用路由平均语义Token比例：{summary['supported_mean_semantic_token_ratio']}
- 代表路径自测：通过

## 2. 路由结果

| 解析路由 | 路径数 | 占全部路径 | 完整解析 | 带附加Token | 部分解析 | 字段冲突 | 兜底 | 平均语义Token比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{route_table}

## 3. 状态机原则

### OUTSHARE和INSHARE主分支

稳定主干按槽位解析：根目录、子系统、FY3域、卫星、仪器、等级。等级后的尾部不再使用固定位置，而按Token类型依次识别产品、区域、周期、分辨率、方向、投影、年份和日期。

### ARCH/DSSCACHE/FY3分支

卫星后允许出现零个或多个处理目录，例如TEMPWORK、MIPS、ENGIN。只有Token本身是仪器候选，或者其后紧跟显式等级时，才认定为仪器。这样避免把MIPS后的GLL错误解析成仪器。

### 不确定信息

- MLT、HAM、NIG、NUL只作为低置信投影候选，并产生告警。
- 未知Token保存在extra_tokens，不删除、不强行分类。
- 原始值和规范化值同时保存，例如0250M规范化为250M，GNOSO规范化为GNOS。
- 多个不同值竞争同一字段时输出field_conflicts，不静默覆盖。

## 4. 当前Top未解释附加Token

| Token | 出现次数 | 路由分布 | 示例路径 |
| --- | ---: | --- | --- |
{extra_table}

## 5. 字段冲突样例

| 字段 | 保留值 | 竞争值 | 路由 | 示例路径 |
| --- | --- | --- | --- | --- |
{conflict_table}

## 6. 当前限制

1. 当前正确性是结构一致性验证，不是业务真值准确率；产品缩写的精确中文释义仍需业务代码表。
2. fallback分支没有专用产品语法，不能据此评价产品字段总体覆盖率。
3. extra_tokens中的高频Token将作为步骤4扩展规则的优先队列。
4. 路径只能提供目录级语义，无法恢复目录内具体文件的时间、版本和格式。

## 7. 输出文件

- parsed_paths.csv.gz：全部路径的结构化解析结果和字段元数据。
- route_summary.csv：各解析路由状态统计。
- field_coverage_by_route.csv：每个路由的字段覆盖率。
- field_value_catalog.csv：各路由、字段和值的频次及示例。
- field_conflicts.csv：字段冲突明细。
- unresolved_extra_tokens.csv：未解释Token优先队列。
- parsed_path_samples.jsonl：包含证据、置信度和告警的解析样例。
- step3_summary.json：总体摘要。

## 8. 下一步

步骤4依据未解释Token频次和fallback分支访问规模扩展规则。优先处理能够显著提升覆盖率且语义可从上下文稳定确认的Token，不追求把软件库目录等低价值路径全部解释成卫星字段。
"""
    (output_dir / "步骤3_路径状态机解析报告.md").write_text(report, encoding="utf-8")

    print("STEP3_PATH_PARSER_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
