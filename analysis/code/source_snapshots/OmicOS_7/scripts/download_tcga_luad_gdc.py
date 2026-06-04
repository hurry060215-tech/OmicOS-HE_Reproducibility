#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


API = "https://api.gdc.cancer.gov"
OUT = Path("/data/hulei/OmicOS/OmicOS_7/inputs/tcga_luad")
LOG = Path("/data/hulei/OmicOS/OmicOS_7/outputs/logs")


def gdc_filter(conditions: list[dict]) -> dict:
    return {"op": "and", "content": conditions}


def eq(field: str, value: str) -> dict:
    return {"op": "=", "content": {"field": field, "value": value}}


def query_files(filters: dict, fields: list[str], size: int = 10000) -> list[dict]:
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
    }
    r = requests.get(f"{API}/files", params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["data"]
    return data.get("hits", [])


def write_manifest(files: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        fh.write("id\tfilename\tmd5\tsize\tstate\n")
        for item in files:
            fh.write(
                f"{item['file_id']}\t{item.get('file_name','')}\t{item.get('md5sum','')}\t{item.get('file_size','')}\t{item.get('state','')}\n"
            )


def safe_name(item: dict) -> str:
    name = item.get("file_name") or f"{item['file_id']}.dat"
    return name.replace("/", "_")


def download_file(item: dict, outdir: Path, retries: int = 3) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / safe_name(item)
    expected = int(item.get("file_size") or 0)
    if out.exists() and expected and out.stat().st_size == expected:
        return {"file_id": item["file_id"], "file_name": out.name, "status": "exists", "size": out.stat().st_size}
    url = f"{API}/data/{item['file_id']}"
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(out)
            return {"file_id": item["file_id"], "file_name": out.name, "status": "downloaded", "size": out.stat().st_size}
        except Exception as exc:
            if attempt == retries:
                return {"file_id": item["file_id"], "file_name": out.name, "status": "failed", "error": repr(exc)}
            time.sleep(10 * attempt)
    return {"file_id": item["file_id"], "file_name": out.name, "status": "failed", "error": "unreachable"}


def query_clinical_cases(path: Path) -> dict:
    fields = [
        "case_id",
        "submitter_id",
        "demographic.gender",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.age_at_diagnosis",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
    ]
    params = {
        "filters": json.dumps(eq("project.project_id", "TCGA-LUAD")),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "2000",
    }
    r = requests.get(f"{API}/cases", params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["data"]
    path.write_text(json.dumps(data.get("hits", []), indent=2), encoding="utf-8")
    return {"cases": len(data.get("hits", [])), "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-slides", type=int, default=10, help="number of SVS slide images to download now")
    parser.add_argument("--download-rna", action="store_true", help="download STAR-count RNA files now; otherwise manifest only")
    parser.add_argument("--max-rna", type=int, default=10)
    args = parser.parse_args()

    manifest_dir = OUT / "manifest"
    slide_dir = OUT / "slides"
    rna_dir = OUT / "rna"
    clinical_dir = OUT / "clinical"
    for p in [manifest_dir, slide_dir, rna_dir, clinical_dir, LOG]:
        p.mkdir(parents=True, exist_ok=True)

    fields = [
        "file_id", "file_name", "md5sum", "file_size", "state", "access",
        "data_category", "data_type", "data_format", "experimental_strategy",
        "cases.case_id", "cases.submitter_id", "cases.samples.sample_type", "cases.samples.submitter_id",
    ]
    slide_filters = gdc_filter([
        eq("cases.project.project_id", "TCGA-LUAD"),
        eq("data_type", "Slide Image"),
        eq("data_format", "SVS"),
    ])
    slide_files = query_files(slide_filters, fields)
    slide_files = sorted(slide_files, key=lambda x: (x.get("cases", [{}])[0].get("submitter_id", ""), x.get("file_name", "")))
    (manifest_dir / "gdc_metadata_tcga_luad_slides.json").write_text(json.dumps(slide_files, indent=2), encoding="utf-8")
    write_manifest(slide_files, manifest_dir / "gdc_manifest_tcga_luad_slides.txt")

    rna_filters = gdc_filter([
        eq("cases.project.project_id", "TCGA-LUAD"),
        eq("data_category", "Transcriptome Profiling"),
        eq("data_type", "Gene Expression Quantification"),
        eq("analysis.workflow_type", "STAR - Counts"),
    ])
    rna_files = query_files(rna_filters, fields + ["analysis.workflow_type"])
    rna_files = sorted(rna_files, key=lambda x: (x.get("cases", [{}])[0].get("submitter_id", ""), x.get("file_name", "")))
    (manifest_dir / "gdc_metadata_tcga_luad_star_counts.json").write_text(json.dumps(rna_files, indent=2), encoding="utf-8")
    write_manifest(rna_files, manifest_dir / "gdc_manifest_tcga_luad_star_counts.txt")

    clinical_status = query_clinical_cases(clinical_dir / "gdc_tcga_luad_clinical_cases.json")

    slide_downloads = []
    for item in slide_files[: max(0, args.max_slides)]:
        slide_downloads.append(download_file(item, slide_dir))

    rna_downloads = []
    if args.download_rna:
        for item in rna_files[: max(0, args.max_rna)]:
            rna_downloads.append(download_file(item, rna_dir))

    summary = {
        "project": "TCGA-LUAD",
        "slide_file_count": len(slide_files),
        "slide_total_bytes": int(sum(int(x.get("file_size") or 0) for x in slide_files)),
        "slide_manifest": str(manifest_dir / "gdc_manifest_tcga_luad_slides.txt"),
        "slide_metadata": str(manifest_dir / "gdc_metadata_tcga_luad_slides.json"),
        "slide_download_limit": args.max_slides,
        "slide_downloads": slide_downloads,
        "rna_file_count": len(rna_files),
        "rna_total_bytes": int(sum(int(x.get("file_size") or 0) for x in rna_files)),
        "rna_manifest": str(manifest_dir / "gdc_manifest_tcga_luad_star_counts.txt"),
        "rna_metadata": str(manifest_dir / "gdc_metadata_tcga_luad_star_counts.json"),
        "rna_downloads": rna_downloads,
        "clinical": clinical_status,
    }
    out = LOG / "tcga_luad_gdc_download_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
