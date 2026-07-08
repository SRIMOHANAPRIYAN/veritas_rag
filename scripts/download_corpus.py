"""Download CUAD corpus from HuggingFace.

Downloads the Contract Understanding Attainment Dataset (CUAD) raw text files
from HuggingFace (theatticusproject/cuad) and places them in data/raw/.

This ensures raw data is always re-creatable and not dependent on local copies.
"""

import sys
from pathlib import Path

from loguru import logger


def download_cuad_corpus(output_dir: str = "data/raw") -> dict:
    """Download CUAD contract text files from HuggingFace.

    Args:
        output_dir: Directory to save downloaded text files.

    Returns:
        Dict with download statistics.
    """
    from huggingface_hub import HfApi, hf_hub_download

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_id = "theatticusproject/cuad"

    logger.info(f"Listing files in HuggingFace dataset: {repo_id}")
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Find the text files folder — may be full_contract_txt or full_contracts_txt
    txt_files = [f for f in all_files if "full_contract" in f and f.endswith(".txt")]
    logger.info(f"Found {len(txt_files)} contract text files in repository")

    if not txt_files:
        logger.error("No text files found in CUAD repository!")
        return {"downloaded": 0, "skipped": 0, "failed": 0}

    downloaded = 0
    skipped = 0
    failed = 0

    for fname in txt_files:
        basename = Path(fname).name
        dest = out_dir / basename

        if dest.exists():
            skipped += 1
            continue

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
            )
            # Copy from HF cache to data/raw/
            import shutil

            shutil.copy2(local_path, dest)
            downloaded += 1
        except Exception as e:
            logger.error(f"Failed to download {fname}: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info("CUAD Download Summary:")
    logger.info(f"  Total in repo:   {len(txt_files)}")
    logger.info(f"  Downloaded:      {downloaded}")
    logger.info(f"  Already existed: {skipped}")
    logger.info(f"  Failed:          {failed}")
    logger.info(f"  Output dir:      {out_dir}")
    logger.info("=" * 60)

    return {
        "total_available": len(txt_files),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


def match_against_manifest(
    raw_dir: str = "data/raw",
    manifest_path: str = "data/indexes/manifest.json",
) -> dict:
    """Match downloaded files against manifest SHA-256 hashes.

    Args:
        raw_dir: Directory containing downloaded text files.
        manifest_path: Path to the semantic index manifest.

    Returns:
        Dict with match statistics.
    """
    import json
    import hashlib

    manifest = json.load(open(manifest_path))
    per_file_sha256 = manifest.get("per_file_sha256", {})

    raw_path = Path(raw_dir)
    matched = 0
    unmatched = 0
    manifest_files_found = 0

    # Build a hash-to-manifest-path lookup
    hash_to_path = {}
    for fpath, fhash in per_file_sha256.items():
        hash_to_path[fhash] = fpath

    # Build a filename (stem) to manifest path lookup for name-based matching
    stem_to_manifest = {}
    for fpath in per_file_sha256:
        stem = Path(fpath).stem
        stem_to_manifest[stem] = fpath

    for txt_file in sorted(raw_path.glob("*.txt")):
        sha = hashlib.sha256()
        with open(txt_file, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        file_hash = sha.hexdigest()

        if file_hash in hash_to_path:
            matched += 1
        else:
            unmatched += 1

    logger.info("=" * 60)
    logger.info("SHA-256 Matching Summary:")
    logger.info(f"  Manifest entries:     {len(per_file_sha256)}")
    logger.info(f"  Files in {raw_dir}:  {matched + unmatched}")
    logger.info(f"  Hash matches:         {matched}")
    logger.info(f"  No hash match:        {unmatched}")
    logger.info("=" * 60)

    return {
        "manifest_entries": len(per_file_sha256),
        "files_checked": matched + unmatched,
        "hash_matches": matched,
        "no_match": unmatched,
    }


if __name__ == "__main__":
    logger.info(f"Interpreter: {sys.prefix}")

    result = download_cuad_corpus()
    if result["downloaded"] > 0 or result["skipped"] > 0:
        match_result = match_against_manifest()
