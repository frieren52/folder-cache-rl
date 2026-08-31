from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


ARCHIVE_ROOTS = {
    "FYDATAOUTSHARE",
    "FYDATAARCH",
    "FYDATATEMP",
    "FYDATAINSHARE",
    "datan_lhhz",
    "smallsat",
    "log",
}

SUBSYSTEMS = {
    "DATAIOT",
    "DATA",
    "DATAIOT_TMP",
    "DATATMP",
    "THEME",
    "DSSCACHE",
    "EXCHANGE",
    "ARSWORK",
    "BAKIOT",
    "TAPE_RETRIEVE",
    "ARCHIVE",
    "YDP",
    "QCSDATA",
    "QCSWORK",
    "software",
}

DATA_DOMAINS = {
    "FY3",
    "FY4",
    "FYSIMU",
    "EVENT",
    "SST_WNP_CASP",
    "XIANTEMP",
    "EXTSAT",
    "RS",
    "DQ",
    "DataSource",
    "GLOBALIMG",
}

# 这里只确认“Token可能是仪器代码”，不把中文释义视为已验证事实。
# 第一组来自项目内标准文档；第二组来自真实路径中稳定的仪器槽位。
DOCUMENTED_INSTRUMENT_CODES = {
    "VIRR",
    "MERSI",
    "MWHS",
    "MWTS",
    "MWRI",
    "IRAS",
    "ERB",
    "GNOS",
    "HIRAS",
    "VASS",
    "MULSS",
    "AGRI",
    "GIIRS",
    "LMI",
    "SEP",
    "AMSUA",
    "AMSUB",
    "ATOVS",
    "AVHRR",
    "HIRS",
    "MODIS",
    "ATMS",
    "CRIS",
    "VIIRS",
    "IMAGE",
    "SOUND",
    "JAMI",
    "AHI",
    "MVIRI",
    "SEVIRI",
    "GOME",
    "IASI",
    "ASCAT",
}

EMPIRICAL_INSTRUMENT_CANDIDATES = {
    "GNOSO",
    "GNOSX",
    "GNOSR",
    "SEM",
    "AHSI",
    "ACDL",
    "PMR",
    "MWRIA",
    "ERM",
    "CPD",
    "OMSN",
    "OMSL",
    "IPM",
    "WRAD",
    "TSHS",
    "GAS",
    "HAOC",
    "GHI",
    "WAILA",
    "WAILI",
    "WAISA",
    "WAISI",
}

DATA_LEVELS = {
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L2L3",
    "1A",
    "1B",
    "L1C",
    "L1D",
    "IMG",
}

ORBIT_DIRECTIONS = {"ASCEND", "DESCEND", "ASCENDC", "DESCENDC"}

REGION_TYPES = {
    "ORBIT",
    "ORBT",
    "ORBN",
    "ORBS",
    "GRAN",
    "GBAL",
    "REG",
    "RNC",
    "RNG",
    "REGC",
    "REGA",
    "REGX",
    "DISK",
    "CHN",
}

AGGREGATION_PERIODS = {
    "DAILY",
    "10DAY",
    "MONTHLY",
    "WEEKLY",
    "POAD",
    "POAM",
    "POWE",
    "PODE",
}

CONFIRMED_PROJECTION_CODES = {"GLL", "NOM", "MCT", "LCC", "PSP", "SIN", "GEO"}
UNCERTAIN_PROJECTION_CODES = {"MLT", "HAM", "NIG", "NUL"}

FORMAT_CODES = {
    "HDF",
    "H5",
    "NC",
    "BIN",
    "L1B",
    "L1C",
    "L1D",
    "SP3",
    "CNT",
    "DAT",
    "AWX",
    "TIF",
    "TAR",
    "JPG",
    "PNG",
    "ZIP",
    "XZ",
    "BZ2",
    "TXT",
    "CSV",
    "XLS",
    "LOG",
    "JLOG",
    "PY",
    "PYC",
    "SO",
    "JAR",
    "INI",
    "XML",
    "SH",
}

PROCESSING_STAGE_CODES = {
    "TEMPWORK",
    "MIPS",
    "ENGIN",
    "TAPE_RETRIEVE",
    "ARCHIVE",
    "EXCHANGE",
    "DSSCACHE",
    "PRODUCT",
    "DataSource",
    "MATCH",
    "software",
    "anaconda3",
    "site-packages",
    "__pycache__",
}

EXTERNAL_PLATFORMS = {
    "GF5A",
    "JPSS1",
    "METOPC",
    "AQUA",
    "TERRA",
    "H09",
    "NAFP",
    "M03",
}

RESOLUTION_RE = re.compile(r"^(?:\d{3,4}M|\d{2,3}KM|GEO1K|GEOQK)$", re.IGNORECASE)
SATELLITE_RE = re.compile(r"^FY[34][A-H](?:CHN|IMG|SIMU)?$", re.IGNORECASE)
TILE_RE = re.compile(r"^(?:T\d{3}|h\d{2}v\d{2})$", re.IGNORECASE)
REGION_ID_RE = re.compile(r"^(?:REG|RNC|RNG)\d+$", re.IGNORECASE)
LONGITUDE_RE = re.compile(r"^\d{4}[EW]$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\d+$")


@dataclass
class Candidate:
    semantic_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class OccurrenceResult:
    path_index: int
    path: str
    root: str
    depth: int
    token: str
    best_type: str
    confidence: float
    evidence: list[str]
    candidates: list[Candidate]
    ambiguous: bool


def is_valid_date_yyyymmdd(token: str) -> bool:
    if not re.fullmatch(r"20\d{6}", token):
        return False
    try:
        datetime.strptime(token, "%Y%m%d")
    except ValueError:
        return False
    return True


def is_year(token: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", token))


def add_candidate(
    candidates: dict[str, Candidate],
    semantic_type: str,
    confidence: float,
    evidence: str,
) -> None:
    current = candidates.get(semantic_type)
    if current is None:
        candidates[semantic_type] = Candidate(semantic_type, confidence, [evidence])
        return
    current.confidence = max(current.confidence, confidence)
    if evidence not in current.evidence:
        current.evidence.append(evidence)


def add_lexical_candidates(token: str, candidates: dict[str, Candidate]) -> None:
    upper = token.upper()

    if token in ARCHIVE_ROOTS:
        add_candidate(candidates, "archive_root", 1.00, "exact_archive_root_code")
    if token in SUBSYSTEMS:
        add_candidate(candidates, "subsystem_or_partition", 0.88, "project_subsystem_codebook")
    if token in DATA_DOMAINS:
        add_candidate(candidates, "data_domain", 0.88, "project_domain_codebook")

    if SATELLITE_RE.fullmatch(token):
        add_candidate(candidates, "satellite", 0.98, "satellite_regex")
    if upper in EXTERNAL_PLATFORMS:
        add_candidate(candidates, "platform", 0.90, "external_platform_codebook_candidate")

    if upper in DOCUMENTED_INSTRUMENT_CODES:
        add_candidate(candidates, "instrument", 0.88, "provided_standard_document_code")
    if upper in EMPIRICAL_INSTRUMENT_CANDIDATES:
        add_candidate(candidates, "instrument", 0.78, "empirical_instrument_candidate_code")

    if upper in DATA_LEVELS:
        add_candidate(candidates, "data_level", 0.99, "exact_data_level_code")
    if RESOLUTION_RE.fullmatch(token):
        add_candidate(candidates, "resolution", 0.99, "resolution_regex")
    if upper in ORBIT_DIRECTIONS:
        add_candidate(candidates, "orbit_direction", 0.99, "exact_orbit_direction_code")
    if upper in REGION_TYPES:
        add_candidate(candidates, "region_or_orbit_type", 0.88, "region_or_orbit_codebook")
    if upper in AGGREGATION_PERIODS:
        add_candidate(candidates, "aggregation_period", 0.92, "aggregation_period_codebook")

    if upper in CONFIRMED_PROJECTION_CODES:
        add_candidate(candidates, "projection", 0.88, "provided_projection_code")
    if upper in UNCERTAIN_PROJECTION_CODES:
        add_candidate(candidates, "projection_candidate", 0.58, "uncertain_projection_code_requires_context")

    if upper in FORMAT_CODES:
        add_candidate(candidates, "format_or_content_type", 0.86, "format_codebook")
    if token in PROCESSING_STAGE_CODES:
        add_candidate(candidates, "processing_stage_or_directory_role", 0.86, "processing_stage_codebook")

    if is_valid_date_yyyymmdd(token):
        add_candidate(candidates, "observe_date", 1.00, "strict_yyyymmdd_date")
    elif is_year(token):
        add_candidate(candidates, "year", 0.99, "four_digit_year")
    elif NUMERIC_RE.fullmatch(token) and len(token) >= 7:
        add_candidate(candidates, "numeric_instance_id", 0.93, "long_numeric_non_date")
    elif NUMERIC_RE.fullmatch(token):
        add_candidate(candidates, "numeric_unknown", 0.45, "short_numeric_token")

    if TILE_RE.fullmatch(token):
        add_candidate(candidates, "tile_or_grid_id", 0.95, "tile_grid_regex")
    if REGION_ID_RE.fullmatch(token):
        add_candidate(candidates, "region_instance_id", 0.94, "region_id_regex")
    if LONGITUDE_RE.fullmatch(token):
        add_candidate(candidates, "sub_point_longitude", 0.95, "longitude_regex")


def add_context_candidates(
    segments: list[str],
    depth: int,
    candidates: dict[str, Candidate],
) -> None:
    token = segments[depth]
    upper = token.upper()
    root = segments[0]
    previous = segments[depth - 1] if depth > 0 else None
    following = segments[depth + 1] if depth + 1 < len(segments) else None

    if depth == 0 and root in ARCHIVE_ROOTS:
        add_candidate(candidates, "archive_root", 1.00, "root_position_and_code_match")

    if depth == 1 and root in {"FYDATAOUTSHARE", "FYDATAARCH", "FYDATAINSHARE", "FYDATATEMP"}:
        add_candidate(candidates, "subsystem_or_partition", 0.98, f"depth1_under_{root}")

    # 两个主流OUTSHARE分支：前三个业务层级经真实数据验证高度稳定。
    if (
        len(segments) >= 6
        and root == "FYDATAOUTSHARE"
        and segments[1] in {"DATA", "DATAIOT"}
        and segments[2] == "FY3"
    ):
        if depth == 2:
            add_candidate(candidates, "data_domain", 1.00, "validated_outshare_fy3_domain_slot")
        elif depth == 3:
            add_candidate(candidates, "satellite", 1.00, "validated_outshare_fy3_satellite_slot")
        elif depth == 4:
            add_candidate(candidates, "instrument", 0.98, "validated_outshare_fy3_instrument_slot")
        elif depth == 5:
            add_candidate(candidates, "data_level", 1.00, "validated_outshare_fy3_level_slot")
        elif depth == 6:
            strong_types = {
                "resolution",
                "orbit_direction",
                "year",
                "region_or_orbit_type",
                "aggregation_period",
            }
            if not strong_types.intersection(candidates):
                add_candidate(candidates, "product_or_business_category", 0.78, "first_untyped_token_after_level")

    # INSHARE主分支与OUTSHARE主干相似。
    if (
        len(segments) >= 6
        and root == "FYDATAINSHARE"
        and segments[1] == "BAKIOT"
        and segments[2] == "FY3"
    ):
        slot_types = {
            2: ("data_domain", 0.98),
            3: ("satellite", 0.98),
            4: ("instrument", 0.96),
            5: ("data_level", 0.98),
        }
        if depth in slot_types:
            semantic_type, confidence = slot_types[depth]
            add_candidate(candidates, semantic_type, confidence, "validated_inshare_fy3_core_slot")

    # DSSCACHE是开放分支：平台之后可能直接是仪器，也可能先经过TEMPWORK/MIPS等处理目录。
    if len(segments) >= 3 and root == "FYDATAARCH" and segments[1] == "DSSCACHE":
        if depth == 2:
            if SATELLITE_RE.fullmatch(token):
                add_candidate(candidates, "satellite", 0.99, "dsscache_platform_slot_satellite_pattern")
            else:
                add_candidate(candidates, "data_domain_or_platform", 0.80, "dsscache_branch_slot")

        if depth == 3:
            if token in PROCESSING_STAGE_CODES or token in {"MIPS", "TEMPWORK", "ENGIN"}:
                add_candidate(candidates, "processing_stage_or_directory_role", 0.96, "dsscache_processing_branch_slot")
            elif following in DATA_LEVELS:
                add_candidate(candidates, "instrument", 0.95, "dsscache_token_followed_by_level")

        if depth >= 4 and previous in {"TEMPWORK", "MIPS", "ENGIN"}:
            if following in DATA_LEVELS or upper in DOCUMENTED_INSTRUMENT_CODES | EMPIRICAL_INSTRUMENT_CANDIDATES:
                add_candidate(candidates, "instrument", 0.88, "instrument_after_processing_directory")

        if upper in DATA_LEVELS:
            add_candidate(candidates, "data_level", 0.99, "explicit_level_inside_dsscache")

        if previous in DATA_LEVELS:
            strong_types = {
                "resolution",
                "orbit_direction",
                "year",
                "observe_date",
                "region_or_orbit_type",
                "aggregation_period",
                "format_or_content_type",
            }
            if not strong_types.intersection(candidates):
                add_candidate(candidates, "product_or_business_category", 0.75, "untyped_token_immediately_after_level")

    # RS/FY4分支在真实数据里具有较明确的目录语法。
    if len(segments) >= 4 and segments[:3] == ["FYDATAARCH", "DSSCACHE", "RS"]:
        fy4_slots = {
            2: ("data_domain", 0.95),
            3: ("satellite", 0.95),
            4: ("instrument", 0.90),
            5: ("data_level", 0.95),
            6: ("product_or_business_category", 0.82),
            7: ("region_or_orbit_type", 0.78),
            8: ("projection", 0.78),
        }
        if depth in fy4_slots:
            semantic_type, confidence = fy4_slots[depth]
            add_candidate(candidates, semantic_type, confidence, "dsscache_rs_fy4_branch_slot")

    # 外部卫星路径：EXTSAT/SATE/{平台}/{产品}/...
    if len(segments) >= 4 and segments[:3] == ["FYDATAARCH", "DSSCACHE", "EXTSAT"]:
        if depth == 3:
            add_candidate(candidates, "processing_stage_or_directory_role", 0.76, "external_satellite_sate_slot")
        elif depth == 4:
            add_candidate(candidates, "platform", 0.92, "external_satellite_platform_slot")
        elif depth == 5:
            add_candidate(candidates, "product_or_business_category", 0.82, "external_satellite_product_slot")

    # 路径末尾的合法8位日期置信度最高；其他长数字更可能是实例ID。
    if depth == len(segments) - 1 and is_valid_date_yyyymmdd(token):
        add_candidate(candidates, "observe_date", 1.00, "valid_date_at_path_tail")


def classify_occurrence(path_index: int, path: str, segments: list[str], depth: int) -> OccurrenceResult:
    candidates: dict[str, Candidate] = {}
    token = segments[depth]
    add_lexical_candidates(token, candidates)
    add_context_candidates(segments, depth, candidates)

    ordered = sorted(candidates.values(), key=lambda item: (-item.confidence, item.semantic_type))
    if not ordered:
        return OccurrenceResult(
            path_index=path_index,
            path=path,
            root=segments[0],
            depth=depth,
            token=token,
            best_type="unknown",
            confidence=0.0,
            evidence=["no_matching_lexical_or_context_rule"],
            candidates=[],
            ambiguous=False,
        )

    best = ordered[0]
    ambiguous = len(ordered) > 1 and (best.confidence - ordered[1].confidence) <= 0.10
    return OccurrenceResult(
        path_index=path_index,
        path=path,
        root=segments[0],
        depth=depth,
        token=token,
        best_type=best.semantic_type,
        confidence=best.confidence,
        evidence=best.evidence,
        candidates=ordered,
        ambiguous=ambiguous,
    )


def parse_input(input_path: Path) -> list[dict]:
    rows: list[dict] = []
    line_re = re.compile(r"^(\d+)\s+(\S+)\s+(\d+)\s*$")
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = line_re.fullmatch(line)
            if not match:
                raise ValueError(f"第{line_number}行无法解析: {line[:200]}")
            path = match.group(2)
            segments = [part for part in path.strip("/").split("/") if part]
            rows.append(
                {
                    "line_number": line_number,
                    "access_count": int(match.group(1)),
                    "path": path,
                    "total_size_bytes": int(match.group(3)),
                    "segments": segments,
                }
            )
    return rows


def confidence_bucket(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "unknown"


def serialize_candidates(candidates: Iterable[Candidate]) -> str:
    return json.dumps(
        [
            {
                "type": item.semantic_type,
                "confidence": round(item.confidence, 4),
                "evidence": item.evidence,
            }
            for item in candidates
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def run_self_tests() -> None:
    cases = [
        (
            "/FYDATAOUTSHARE/DATAIOT/FY3/FY3H/MERSI/L2L3/OCA/ORBIT/010KM/2026/20260331",
            {0: "archive_root", 3: "satellite", 4: "instrument", 5: "data_level", 6: "product_or_business_category", 8: "resolution", 10: "observe_date"},
        ),
        (
            "/FYDATAARCH/DSSCACHE/FY3H/TEMPWORK/MERSI/L2L3/OCA/ORBIT/010KM/20260518",
            {0: "archive_root", 1: "subsystem_or_partition", 2: "satellite", 3: "processing_stage_or_directory_role", 4: "instrument", 5: "data_level", 9: "observe_date"},
        ),
        (
            "/FYDATAARCH/DSSCACHE/FY3H/MIPS/GLL/HDF/0250M/20260518",
            {3: "processing_stage_or_directory_role", 4: "projection", 5: "format_or_content_type", 6: "resolution", 7: "observe_date"},
        ),
    ]

    for path, expected in cases:
        segments = path.strip("/").split("/")
        for depth, expected_type in expected.items():
            result = classify_occurrence(0, path, segments, depth)
            if result.best_type != expected_type:
                raise AssertionError(
                    f"自测失败: {path} depth={depth} token={segments[depth]} "
                    f"expected={expected_type} actual={result.best_type} candidates={serialize_candidates(result.candidates)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤2：路径Token候选类型分类与验证")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    run_self_tests()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_input(input_path)
    occurrences: list[OccurrenceResult] = []
    path_type_sets: list[set[str]] = []
    path_confidences: list[list[float]] = []

    for path_index, row in enumerate(rows):
        type_set: set[str] = set()
        confidences: list[float] = []
        for depth in range(len(row["segments"])):
            result = classify_occurrence(path_index, row["path"], row["segments"], depth)
            occurrences.append(result)
            if result.best_type != "unknown":
                type_set.add(result.best_type)
            confidences.append(result.confidence)
        path_type_sets.append(type_set)
        path_confidences.append(confidences)

    occurrence_path = output_dir / "token_occurrences.csv.gz"
    with gzip.open(occurrence_path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path_index",
                "path",
                "root",
                "depth",
                "token",
                "best_type",
                "confidence",
                "confidence_bucket",
                "ambiguous",
                "evidence",
                "candidates_json",
            ],
        )
        writer.writeheader()
        for item in occurrences:
            writer.writerow(
                {
                    "path_index": item.path_index,
                    "path": item.path,
                    "root": item.root,
                    "depth": item.depth,
                    "token": item.token,
                    "best_type": item.best_type,
                    "confidence": round(item.confidence, 4),
                    "confidence_bucket": confidence_bucket(item.confidence),
                    "ambiguous": int(item.ambiguous),
                    "evidence": "|".join(item.evidence),
                    "candidates_json": serialize_candidates(item.candidates),
                }
            )

    type_counts = Counter(item.best_type for item in occurrences)
    type_rows: dict[str, set[int]] = defaultdict(set)
    type_tokens: dict[str, set[str]] = defaultdict(set)
    for item in occurrences:
        type_rows[item.best_type].add(item.path_index)
        type_tokens[item.best_type].add(item.token)

    field_type_rows = []
    for semantic_type, count in type_counts.most_common():
        field_type_rows.append(
            {
                "semantic_type": semantic_type,
                "token_occurrences": count,
                "occurrence_percentage": round(100.0 * count / len(occurrences), 4),
                "paths_with_type": len(type_rows[semantic_type]),
                "path_coverage_percentage": round(100.0 * len(type_rows[semantic_type]) / len(rows), 4),
                "unique_tokens": len(type_tokens[semantic_type]),
            }
        )

    with (output_dir / "field_type_coverage.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_type_rows[0]))
        writer.writeheader()
        writer.writerows(field_type_rows)

    token_stats: dict[str, dict] = {}
    for item in occurrences:
        stats = token_stats.setdefault(
            item.token,
            {
                "count": 0,
                "paths": set(),
                "roots": Counter(),
                "depths": Counter(),
                "best_types": Counter(),
                "confidences": [],
                "ambiguous": 0,
                "evidence": Counter(),
                "examples": [],
            },
        )
        stats["count"] += 1
        stats["paths"].add(item.path_index)
        stats["roots"][item.root] += 1
        stats["depths"][item.depth] += 1
        stats["best_types"][item.best_type] += 1
        stats["confidences"].append(item.confidence)
        stats["ambiguous"] += int(item.ambiguous)
        for evidence in item.evidence:
            stats["evidence"][evidence] += 1
        if len(stats["examples"]) < 3 and item.path not in stats["examples"]:
            stats["examples"].append(item.path)

    catalog_rows = []
    for token, stats in token_stats.items():
        dominant_type, dominant_count = stats["best_types"].most_common(1)[0]
        dominant_ratio = dominant_count / stats["count"]
        average_confidence = mean(stats["confidences"])
        nonzero_types = [name for name, count in stats["best_types"].items() if name != "unknown" and count > 0]
        if dominant_type == "unknown":
            status = "unknown"
        elif len(nonzero_types) > 1 and dominant_ratio < 0.95:
            status = "context_dependent"
        elif average_confidence < 0.70:
            status = "low_confidence"
        else:
            status = "stable"

        catalog_rows.append(
            {
                "token": token,
                "occurrences": stats["count"],
                "paths": len(stats["paths"]),
                "dominant_type": dominant_type,
                "dominant_ratio": round(dominant_ratio, 4),
                "mean_confidence": round(average_confidence, 4),
                "max_confidence": round(max(stats["confidences"]), 4),
                "ambiguous_occurrences": stats["ambiguous"],
                "status": status,
                "type_counts_json": json.dumps(stats["best_types"], ensure_ascii=False, separators=(",", ":")),
                "root_counts_json": json.dumps(stats["roots"], ensure_ascii=False, separators=(",", ":")),
                "depth_counts_json": json.dumps(stats["depths"], ensure_ascii=False, separators=(",", ":")),
                "top_evidence": "|".join(name for name, _ in stats["evidence"].most_common(5)),
                "example_paths": " || ".join(stats["examples"]),
            }
        )

    catalog_rows.sort(key=lambda row: (-row["occurrences"], row["token"]))
    with (output_dir / "token_catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(catalog_rows)

    ambiguous_rows = [
        row for row in catalog_rows if row["status"] == "context_dependent" or row["ambiguous_occurrences"] > 0
    ]
    with (output_dir / "ambiguous_tokens.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(ambiguous_rows)

    unknown_rows = [row for row in catalog_rows if row["dominant_type"] == "unknown"]
    with (output_dir / "unknown_tokens.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(unknown_rows)

    expected_fields = [
        "archive_root",
        "subsystem_or_partition",
        "data_domain",
        "satellite",
        "platform",
        "instrument",
        "data_level",
        "product_or_business_category",
        "region_or_orbit_type",
        "aggregation_period",
        "projection",
        "projection_candidate",
        "resolution",
        "orbit_direction",
        "observe_date",
        "year",
        "format_or_content_type",
    ]
    path_field_rows = []
    for semantic_type in expected_fields:
        count = sum(semantic_type in type_set for type_set in path_type_sets)
        path_field_rows.append(
            {
                "semantic_type": semantic_type,
                "paths_with_type": count,
                "path_coverage_percentage": round(100.0 * count / len(rows), 4),
            }
        )
    with (output_dir / "path_field_coverage.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_field_rows[0]))
        writer.writeheader()
        writer.writerows(path_field_rows)

    confidence_counts = Counter(confidence_bucket(item.confidence) for item in occurrences)
    high_or_medium = confidence_counts["high"] + confidence_counts["medium"]
    unknown_occurrences = confidence_counts["unknown"]
    ambiguous_occurrences = sum(item.ambiguous for item in occurrences)

    randomizer = random.Random(20260820)
    sample_indices = set(randomizer.sample(range(len(rows)), min(args.sample_size, len(rows))))
    ambiguous_path_indices = {item.path_index for item in occurrences if item.ambiguous}
    sample_indices.update(sorted(ambiguous_path_indices)[:500])
    occurrence_by_path: dict[int, list[OccurrenceResult]] = defaultdict(list)
    for item in occurrences:
        if item.path_index in sample_indices:
            occurrence_by_path[item.path_index].append(item)

    with (output_dir / "classified_path_samples.jsonl").open("w", encoding="utf-8") as handle:
        for path_index in sorted(sample_indices):
            row = rows[path_index]
            payload = {
                "path_index": path_index,
                "path": row["path"],
                "access_count": row["access_count"],
                "tokens": [
                    {
                        "depth": item.depth,
                        "token": item.token,
                        "best_type": item.best_type,
                        "confidence": round(item.confidence, 4),
                        "ambiguous": item.ambiguous,
                        "evidence": item.evidence,
                        "candidates": [
                            {
                                "type": candidate.semantic_type,
                                "confidence": round(candidate.confidence, 4),
                                "evidence": candidate.evidence,
                            }
                            for candidate in item.candidates
                        ],
                    }
                    for item in occurrence_by_path[path_index]
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "input_file": str(input_path),
        "paths": len(rows),
        "token_occurrences": len(occurrences),
        "unique_tokens": len(token_stats),
        "confidence_counts": dict(confidence_counts),
        "high_or_medium_occurrence_coverage_percentage": round(100.0 * high_or_medium / len(occurrences), 4),
        "unknown_occurrences": unknown_occurrences,
        "unknown_occurrence_percentage": round(100.0 * unknown_occurrences / len(occurrences), 4),
        "ambiguous_occurrences": ambiguous_occurrences,
        "ambiguous_occurrence_percentage": round(100.0 * ambiguous_occurrences / len(occurrences), 4),
        "unknown_unique_tokens": len(unknown_rows),
        "ambiguous_or_context_dependent_unique_tokens": len(ambiguous_rows),
        "self_tests": "passed",
    }
    (output_dir / "step2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_unknown = unknown_rows[:25]
    top_ambiguous = ambiguous_rows[:25]
    top_types = field_type_rows[:25]

    type_table = "\n".join(
        f"| {row['semantic_type']} | {row['token_occurrences']} | {row['paths_with_type']} | "
        f"{row['path_coverage_percentage']}% | {row['unique_tokens']} |"
        for row in top_types
    )
    unknown_table = "\n".join(
        f"| {row['token']} | {row['occurrences']} | {row['root_counts_json']} | {row['example_paths'][:160]} |"
        for row in top_unknown
    ) or "| 无 | 0 | - | - |"
    ambiguous_table = "\n".join(
        f"| {row['token']} | {row['occurrences']} | {row['type_counts_json']} | {row['status']} |"
        for row in top_ambiguous
    ) or "| 无 | 0 | - | - |"

    report = f"""# 步骤2：路径Token类型识别与验证报告

## 1. 本步骤目标

本步骤只回答“路径中的每个Token可能属于什么类型”，暂不把整条路径强制组装成唯一Schema。

- 输入路径：{len(rows)}
- Token出现次数：{len(occurrences)}
- 唯一Token：{len(token_stats)}
- 内置代表路径自测：通过

## 2. 分类结果概览

| 置信度 | Token出现次数 | 含义 |
| --- | ---: | --- |
| high（>=0.90） | {confidence_counts['high']} | 强正则或已验证分支槽位 |
| medium（0.70～0.90） | {confidence_counts['medium']} | 代码表候选或上下文推断 |
| low（0～0.70） | {confidence_counts['low']} | 含义待确认，只能作为候选 |
| unknown | {confidence_counts['unknown']} | 当前无规则，不强行解释 |

中高置信Token覆盖率为 {summary['high_or_medium_occurrence_coverage_percentage']}%。未知Token出现占比为 {summary['unknown_occurrence_percentage']}%。

多候选分数接近的Token出现 {ambiguous_occurrences} 次，占 {summary['ambiguous_occurrence_percentage']}%。这些Token必须交由下一步的目录状态机消歧。

## 3. 识别出的主要类型

| 类型 | Token出现次数 | 覆盖路径数 | 路径覆盖率 | 唯一Token数 |
| --- | ---: | ---: | ---: | ---: |
{type_table}

## 4. 当前Top未知Token

| Token | 出现次数 | 根目录分布 | 示例路径 |
| --- | ---: | --- | --- |
{unknown_table}

未知不表示无价值。未知Token会原样保留，后续根据前后文、共现和业务确认逐步加入代码表。

## 5. 当前Top多义或上下文依赖Token

| Token | 出现次数 | 不同上下文下的最佳类型 | 状态 |
| --- | ---: | --- | --- |
{ambiguous_table}

同一个Token在不同分支中可以有不同角色，因此分类结果按“Token出现位置”保存，而不是为每个字符串永久指定唯一类型。

## 6. 置信度来源

- 1.00～0.98：严格日期、明确等级/分辨率/方向正则，或经真实数据验证稳定的OUTSHARE主干槽位。
- 0.95～0.90：较稳定分支位置或明确平台、区域、周期代码。
- 0.89～0.70：项目内代码表候选和上下文推断，仍需目录状态机与业务样本确认。
- 低于0.70：MLT/HAM/NIG/NUL等含义尚未验证的候选，不直接写入最终字段。
- 0：没有足够证据，输出unknown。

## 7. 本步骤边界

1. 项目内AI生成文档只作为候选代码来源，不作为权威真值。
2. 仪器代码可以通过稳定槽位验证其“字段类型”，但精确中文释义仍需正式代码表或业务人员确认。
3. product是开放集合，主要依靠“等级后的首个未类型化Token”生成中置信候选。
4. 本步骤不会把unknown删除；原始Token、深度、根目录和示例路径全部保留。
5. 最终字段需要步骤3的分支路由和状态机完成。

## 8. 输出文件

- step2_summary.json：总体结果。
- token_occurrences.csv.gz：全部Token逐次分类结果。
- token_catalog.csv：按Token聚合的类型、置信度和上下文。
- field_type_coverage.csv：识别类型覆盖情况。
- path_field_coverage.csv：每种字段候选在路径级的覆盖率。
- ambiguous_tokens.csv：多义和上下文依赖Token。
- unknown_tokens.csv：当前无法解释的Token。
- classified_path_samples.jsonl：分层抽样的完整分类示例。

## 9. 下一步建议

步骤3根据根目录和前三层分支建立路由，再用状态机把候选Token组装成统一字段。第一批优先处理：

1. FYDATAOUTSHARE/DATAIOT/FY3
2. FYDATAOUTSHARE/DATA/FY3
3. FYDATAARCH/DSSCACHE/FY3H、FY3F、FY3G
4. FYDATAINSHARE/BAKIOT/FY3

解析器必须输出字段值、来源深度、证据、置信度和冲突，不追求所有字段100%填满。
"""
    (output_dir / "步骤2_Token类型识别报告.md").write_text(report, encoding="utf-8")

    print("STEP2_TOKEN_CLASSIFICATION_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
