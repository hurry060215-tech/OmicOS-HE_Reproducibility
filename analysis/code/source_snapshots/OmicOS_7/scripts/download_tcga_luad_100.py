#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


API = "https://api.gdc.cancer.gov"
SOURCE = Path("/data/hulei/OmicOS/OmicOS_7/inputs/tcga_luad")
OUT = Path("/data/hulei/OmicOS/OmicOS_7/inputs/tcga_luad_100")
LOG = Path("/data/hulei/OmicOS/OmicOS_7/outputs/logs")


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def patient_id(item: dict[str, Any]) -> str:
    return item["cases"][0]["submitter_id"]


def case_id(item: dict[str, Any]) -> str:
    return item["cases"][0].get("case_id", "")


def sample_type(item: dict[str, Any]) -> str:
    samples = item.get("cases", [{}])[0].get("samples") or []
    if not samples:
        return ""
    return samples[0].get("sample_type") or ""


def slide_code(file_name: str) -> str:
    match = re.search(r"-(DX|TS|BS)(\d+)\.", file_name)
    if not match:
        return "NA"
    return f"{match.group(1)}{match.group(2)}"


def slide_rank(item: dict[str, Any]) -> tuple[int, int, int, int, str]:
    code = slide_code(item.get("file_name", ""))
    match = re.match(r"(DX|TS|BS)(\d+)$", code)
    type_priority = {"DX": 0, "TS": 1, "BS": 2}
    if match:
        family = match.group(1)
        number = int(match.group(2))
        family_rank = type_priority.get(family, 9)
    else:
        number = 99
        family_rank = 9
    sample_rank = 0 if sample_type(item) == "Primary Tumor" else 1
    expected = int(item.get("file_size") or 0)
    return (sample_rank, family_rank, number, expected, item.get("file_name", ""))


def survival_record(case: dict[str, Any]) -> dict[str, Any] | None:
    demographic = case.get("demographic") or {}
    diagnoses = case.get("diagnoses") or []
    diagnosis = diagnoses[0] if diagnoses else {}
    vital_status = demographic.get("vital_status")
    days_to_death = demographic.get("days_to_death") or diagnosis.get("days_to_death")
    days_to_follow = demographic.get("days_to_last_follow_up") or diagnosis.get("days_to_last_follow_up")
    if vital_status not in {"Alive", "Dead"}:
        return None
    if vital_status == "Dead":
        survival_days = days_to_death if days_to_death is not None else days_to_follow
    else:
        survival_days = days_to_follow
    if survival_days is None:
        return None
    stage = diagnosis.get("ajcc_pathologic_stage") or "NA"
    return {
        "vital_status": vital_status,
        "event": 1 if vital_status == "Dead" else 0,
        "survival_days": float(survival_days),
        "stage": stage,
        "stage_group": stage_group(stage),
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
        "gender": demographic.get("gender"),
    }


def stage_group(stage: str) -> str:
    text = (stage or "").upper()
    if "IV" in text:
        return "IV"
    if "III" in text:
        return "III"
    if "II" in text:
        return "II"
    if "I" in text:
        return "I"
    return "NA"


def proportional_stage_sample(
    candidates: list[dict[str, Any]],
    target_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target_n <= 0:
        return []
    if len(candidates) <= target_n:
        return list(candidates)

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[item["stage_group"]].append(item)
    for items in groups.values():
        rng.shuffle(items)

    total = len(candidates)
    raw = {k: target_n * len(v) / total for k, v in groups.items()}
    quota = {k: min(len(groups[k]), int(raw[k])) for k in groups}
    remaining = target_n - sum(quota.values())
    ranked = sorted(groups, key=lambda k: (raw[k] - int(raw[k]), len(groups[k])), reverse=True)
    while remaining > 0:
        changed = False
        for key in ranked:
            if remaining <= 0:
                break
            if quota[key] < len(groups[key]):
                quota[key] += 1
                remaining -= 1
                changed = True
        if not changed:
            break

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        selected.extend(groups[key][: quota[key]])
    rng.shuffle(selected)
    return selected[:target_n]


def choose_patients(
    slides: list[dict[str, Any]],
    rna_files: list[dict[str, Any]],
    clinical_cases: list[dict[str, Any]],
    n_patients: int,
    target_events: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slides_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in slides:
        slides_by_patient[patient_id(item)].append(item)

    rna_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rna_files:
        if sample_type(item) == "Primary Tumor":
            rna_by_patient[patient_id(item)].append(item)

    clinical_by_patient = {case["submitter_id"]: case for case in clinical_cases}
    survival_by_patient = {
        pid: record
        for pid, case in clinical_by_patient.items()
        if (record := survival_record(case)) is not None
    }

    candidates: list[dict[str, Any]] = []
    skip_reasons = Counter()
    all_patients = sorted(set(slides_by_patient) | set(rna_by_patient) | set(clinical_by_patient))
    for pid in all_patients:
        patient_slides = slides_by_patient.get(pid, [])
        primary_slides = [item for item in patient_slides if sample_type(item) == "Primary Tumor"]
        analysis_slides = [
            item for item in primary_slides if slide_code(item.get("file_name", "")).startswith(("DX", "TS"))
        ]
        if not primary_slides:
            skip_reasons["no_primary_tumor_slide"] += 1
            continue
        if not analysis_slides:
            skip_reasons["no_dx_or_ts_primary_tumor_slide"] += 1
            continue
        if pid not in rna_by_patient:
            skip_reasons["no_primary_tumor_star_counts"] += 1
            continue
        if pid not in survival_by_patient:
            skip_reasons["no_usable_survival"] += 1
            continue
        slide = sorted(analysis_slides, key=slide_rank)[0]
        rna = sorted(rna_by_patient[pid], key=lambda x: (int(x.get("file_size") or 0), x.get("file_name", "")))[0]
        record = {
            "patient_id": pid,
            "case_id": case_id(slide),
            **survival_by_patient[pid],
            "slide": slide,
            "rna": rna,
            "slide_code": slide_code(slide.get("file_name", "")),
        }
        candidates.append(record)

    dead = [item for item in candidates if item["event"] == 1]
    alive = [item for item in candidates if item["event"] == 0]
    dead_n = min(target_events, len(dead), n_patients)
    alive_n = min(n_patients - dead_n, len(alive))
    selected = proportional_stage_sample(dead, dead_n, seed) + proportional_stage_sample(alive, alive_n, seed + 1)
    if len(selected) < n_patients:
        selected_ids = {item["patient_id"] for item in selected}
        remainder = [item for item in candidates if item["patient_id"] not in selected_ids]
        selected.extend(proportional_stage_sample(remainder, n_patients - len(selected), seed + 2))

    selected = sorted(selected, key=lambda x: x["patient_id"])
    summary = {
        "eligible_patients": len(candidates),
        "requested_patients": n_patients,
        "selected_patients": len(selected),
        "requested_events": target_events,
        "selected_events": int(sum(item["event"] for item in selected)),
        "skip_reasons": dict(skip_reasons),
        "selected_stage_groups": dict(Counter(item["stage_group"] for item in selected)),
        "selected_vital_status": dict(Counter(item["vital_status"] for item in selected)),
        "selected_slide_codes": dict(Counter(item["slide_code"] for item in selected)),
    }
    if len(selected) < n_patients:
        summary["warning"] = "Fewer eligible patients than requested."
    return selected, summary


def manifest_row(item: dict[str, Any]) -> list[Any]:
    return [
        item["file_id"],
        item.get("file_name", ""),
        item.get("md5sum", ""),
        item.get("file_size", ""),
        item.get("state", ""),
    ]


def write_manifest(items: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as fh:
        fh.write("id\tfilename\tmd5\tsize\tstate\n")
        for item in items:
            fh.write("\t".join(str(x) for x in manifest_row(item)) + "\n")


def write_selection(selected: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "patient_id",
        "case_id",
        "vital_status",
        "event",
        "survival_days",
        "stage",
        "stage_group",
        "age_at_diagnosis",
        "gender",
        "slide_code",
        "slide_file_id",
        "slide_file_name",
        "slide_file_size",
        "rna_file_id",
        "rna_file_name",
        "rna_file_size",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "patient_id": item["patient_id"],
                    "case_id": item["case_id"],
                    "vital_status": item["vital_status"],
                    "event": item["event"],
                    "survival_days": item["survival_days"],
                    "stage": item["stage"],
                    "stage_group": item["stage_group"],
                    "age_at_diagnosis": item["age_at_diagnosis"],
                    "gender": item["gender"],
                    "slide_code": item["slide_code"],
                    "slide_file_id": item["slide"]["file_id"],
                    "slide_file_name": item["slide"].get("file_name", ""),
                    "slide_file_size": item["slide"].get("file_size", ""),
                    "rna_file_id": item["rna"]["file_id"],
                    "rna_file_name": item["rna"].get("file_name", ""),
                    "rna_file_size": item["rna"].get("file_size", ""),
                }
            )


def safe_name(item: dict[str, Any]) -> str:
    return (item.get("file_name") or f"{item['file_id']}.dat").replace("/", "_")


def link_existing(item: dict[str, Any], out: Path, search_dirs: list[Path]) -> bool:
    expected = int(item.get("file_size") or 0)
    if not expected:
        return False
    for directory in search_dirs:
        source = directory / safe_name(item)
        if not source.exists() or source.stat().st_size != expected:
            continue
        try:
            os.link(source, out)
        except OSError:
            shutil.copy2(source, out)
        return True
    return False


def download_file(item: dict[str, Any], outdir: Path, search_dirs: list[Path], retries: int = 3) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / safe_name(item)
    expected = int(item.get("file_size") or 0)
    if out.exists() and expected and out.stat().st_size == expected:
        return {"file_id": item["file_id"], "file_name": out.name, "status": "exists", "size": out.stat().st_size}
    if not out.exists() and link_existing(item, out, search_dirs):
        return {"file_id": item["file_id"], "file_name": out.name, "status": "linked_existing", "size": out.stat().st_size}

    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{API}/data/{item['file_id']}"
    for attempt in range(1, retries + 1):
        try:
            resume_from = tmp.stat().st_size if tmp.exists() else 0
            headers = {}
            if expected and 0 < resume_from < expected:
                headers["Range"] = f"bytes={resume_from}-"
            with requests.get(url, stream=True, timeout=180, headers=headers) as response:
                response.raise_for_status()
                mode = "ab" if headers and response.status_code == 206 else "wb"
                with tmp.open(mode) as fh:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(out)
            observed = out.stat().st_size
            if expected and observed != expected:
                raise RuntimeError(f"size mismatch: expected {expected}, observed {observed}")
            return {"file_id": item["file_id"], "file_name": out.name, "status": "downloaded", "size": observed}
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                return {"file_id": item["file_id"], "file_name": out.name, "status": "failed", "error": repr(exc)}
            time.sleep(15 * attempt)
    return {"file_id": item["file_id"], "file_name": out.name, "status": "failed", "error": "unreachable"}


def download_many(
    items: list[dict[str, Any]],
    outdir: Path,
    search_dirs: list[Path],
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(download_file, item, outdir, search_dirs) for item in items]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(json.dumps({"label": label, "done": index, "total": len(items), **result}, ensure_ascii=False), flush=True)
    return results


def existing_cohort_roots(out_root: Path) -> list[Path]:
    roots = []
    for root in sorted(out_root.parent.glob("tcga_luad_*")):
        if root == out_root:
            continue
        if (root / "slides").exists() or (root / "rna").exists():
            roots.append(root)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=100)
    parser.add_argument("--target-events", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--skip-slides", action="store_true")
    parser.add_argument("--skip-rna", action="store_true")
    parser.add_argument("--slide-workers", type=int, default=3)
    parser.add_argument("--rna-workers", type=int, default=4)
    args = parser.parse_args()

    out_root = args.out_root
    cohort_label = out_root.name
    manifest_dir = out_root / "manifest"
    slide_dir = out_root / "slides"
    rna_dir = out_root / "rna"
    clinical_dir = out_root / "clinical"
    table_dir = out_root / "tables"
    for path in [manifest_dir, slide_dir, rna_dir, clinical_dir, table_dir, LOG]:
        path.mkdir(parents=True, exist_ok=True)

    slides = load_json(SOURCE / "manifest" / "gdc_metadata_tcga_luad_slides.json")
    rna_files = load_json(SOURCE / "manifest" / "gdc_metadata_tcga_luad_star_counts.json")
    clinical_cases = load_json(SOURCE / "clinical" / "gdc_tcga_luad_clinical_cases.json")

    selected, selection_summary = choose_patients(
        slides=slides,
        rna_files=rna_files,
        clinical_cases=clinical_cases,
        n_patients=args.n_patients,
        target_events=args.target_events,
        seed=args.seed,
    )
    selected_slides = [item["slide"] for item in selected]
    selected_rna = [item["rna"] for item in selected]
    selected_patients = {item["patient_id"] for item in selected}
    selected_clinical = [case for case in clinical_cases if case["submitter_id"] in selected_patients]

    write_selection(selected, table_dir / f"{cohort_label}_patient_selection.csv")
    write_manifest(selected_slides, manifest_dir / f"gdc_manifest_{cohort_label}_slides.txt")
    write_manifest(selected_rna, manifest_dir / f"gdc_manifest_{cohort_label}_star_counts.txt")
    (manifest_dir / f"gdc_metadata_{cohort_label}_slides.json").write_text(json.dumps(selected_slides, indent=2), encoding="utf-8")
    (manifest_dir / f"gdc_metadata_{cohort_label}_star_counts.json").write_text(json.dumps(selected_rna, indent=2), encoding="utf-8")
    (clinical_dir / f"gdc_{cohort_label}_clinical_subset.json").write_text(json.dumps(selected_clinical, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "project": "TCGA-LUAD",
        "output_root": str(out_root),
        "selection": selection_summary,
        "slide_manifest": str(manifest_dir / f"gdc_manifest_{cohort_label}_slides.txt"),
        "rna_manifest": str(manifest_dir / f"gdc_manifest_{cohort_label}_star_counts.txt"),
        "patient_selection": str(table_dir / f"{cohort_label}_patient_selection.csv"),
        "slide_total_bytes": int(sum(int(item.get("file_size") or 0) for item in selected_slides)),
        "rna_total_bytes": int(sum(int(item.get("file_size") or 0) for item in selected_rna)),
        "slide_downloads": [],
        "rna_downloads": [],
    }

    if not args.select_only:
        reuse_roots = existing_cohort_roots(out_root)
        if not args.skip_slides:
            summary["slide_downloads"] = download_many(
                selected_slides,
                slide_dir,
                [SOURCE / "slides"] + [root / "slides" for root in reuse_roots],
                workers=args.slide_workers,
                label="slide",
            )
        if not args.skip_rna:
            summary["rna_downloads"] = download_many(
                selected_rna,
                rna_dir,
                [SOURCE / "rna"] + [root / "rna" for root in reuse_roots],
                workers=args.rna_workers,
                label="rna",
            )

    out = LOG / f"{cohort_label}_download_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
