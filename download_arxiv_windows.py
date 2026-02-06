"""
================================================================================
arXiv HTML Downloader - Quantum Computing Research Pipeline-Step 1
================================================================================

Purpose: Downloads HTML versions of arXiv papers for figure caption extraction.

USAGE:
1. Place Sample_papers.csv in SAME FOLDER as this script
2. Run: python download_arxiv.py  
3. Output: downloaded_html/ folder with HTML files

Sample_papers.csv format (one arXiv ID per line):
2301.12345
2205.06789
2103.45678

REQUIREMENTS: Python 3.6+ and 'pip install requests'

Output files are named: 00001_arXiv_2301.12345.html
Supports resuming interrupted downloads.

================================================================================
"""


import os
import time
import json
import random
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# ============ CONFIGURATION ============
# Your Windows path - CHANGE THIS IF NEEDED
OUTPUT_DIR = "downloaded_html"

# Input file - should be in the same folder as this script, or provide full path
IDS_FILE = "Sample_papers.csv"

# Checkpoint file (tracks progress, allows resume)
CHECKPOINT_FILE = "download_checkpoint.json"

# arXiv API
API_URL = "http://export.arxiv.org/api/query"

# Speed settings
DELAY_MIN = 2.0          # Minimum seconds between downloads
DELAY_MAX = 4.0          # Maximum seconds between downloads  
BATCH_SIZE = 50          # Pause every N downloads
BATCH_PAUSE = 60         # Seconds to pause between batches
RATE_LIMIT_PAUSE = 180   # Seconds to wait if rate limited

# ============ END CONFIGURATION ============

# Rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
]


def print_banner():
    print()
    print("=" * 70)
    print("  arXiv HTML Downloader")
    print("  Downloading to:", OUTPUT_DIR)
    print("=" * 70)
    print()


def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }


def normalize_id(raw_id):
    """Clean arXiv ID."""
    raw_id = raw_id.strip().lstrip('\ufeff')
    if raw_id.lower().startswith("arxiv:"):
        raw_id = raw_id.split(":", 1)[1]
    return raw_id


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"downloaded": [], "failed": [], "not_available": []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def get_already_downloaded():
    """Get set of already downloaded arXiv IDs."""
    downloaded = set()
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith(".html"):
                parts = fname.replace(".html", "").split("_arXiv_")
                if len(parts) == 2:
                    downloaded.add(parts[1])
    return downloaded


def get_versioned_id(arxiv_id, session):
    """Get versioned ID from arXiv API."""
    params = {"id_list": arxiv_id, "max_results": 1}
    resp = session.get(API_URL, params=params, headers=get_headers(), timeout=30)
    
    if resp.status_code == 429:
        raise Exception("RATE_LIMITED")
    
    resp.raise_for_status()
    
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", ns)
    
    if entry is None:
        raise ValueError(f"No entry for {arxiv_id}")
    
    title = entry.findtext("atom:title", default="", namespaces=ns)
    entry_id = entry.findtext("atom:id", default="", namespaces=ns)
    
    if title == "Error" or "api/errors#" in entry_id:
        raise ValueError(f"API error for {arxiv_id}")
    
    full_id_url = entry.findtext("atom:id", default="", namespaces=ns)
    return full_id_url.rsplit("/", 1)[-1] if full_id_url else arxiv_id


def download_paper(idx, arxiv_id, session, checkpoint):
    """Download a single paper's HTML."""
    norm_id = normalize_id(arxiv_id)
    
    try:
        # Get versioned ID from API
        versioned_id = get_versioned_id(norm_id, session)
        time.sleep(0.5)  # Small delay between API and HTML
        
        # Download HTML
        html_url = f"https://arxiv.org/html/{versioned_id}"
        resp = session.get(html_url, headers=get_headers(), timeout=60)
        
        if resp.status_code == 429:
            return "rate_limited", "Rate limited by arXiv"
        
        if resp.status_code == 404:
            checkpoint["not_available"].append(norm_id)
            return "not_available", "HTML version not available"
        
        resp.raise_for_status()
        
        # Save file
        filename = f"{idx:05d}_arXiv_{norm_id}.html"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(resp.text)
        
        checkpoint["downloaded"].append(norm_id)
        return "success", filename
        
    except Exception as e:
        error_msg = str(e)
        if "RATE_LIMITED" in error_msg or "429" in error_msg:
            return "rate_limited", error_msg
        
        if norm_id not in checkpoint["failed"]:
            checkpoint["failed"].append(norm_id)
        return "failed", error_msg


def main():
    print_banner()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output folder: {OUTPUT_DIR}")
    
    # Check if input file exists
    if not os.path.exists(IDS_FILE):
        print(f"\nERROR: Cannot find '{IDS_FILE}'")
        print(f"Please make sure Sample_papers.csv is in the same folder as this script.")
        print(f"Current folder: {os.getcwd()}")
        input("\nPress Enter to exit...")
        return
    
    # Load paper IDs
    with open(IDS_FILE, "r", encoding="utf-8") as f:
        raw_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Total papers in list: {len(raw_ids)}")
    
    # Get already downloaded
    already_downloaded = get_already_downloaded()
    print(f"Already downloaded: {len(already_downloaded)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Build list of papers to download
    to_download = []
    for idx, raw_id in enumerate(raw_ids, start=1):
        norm_id = normalize_id(raw_id)
        if norm_id not in already_downloaded:
            to_download.append((idx, raw_id))
    
    print(f"Remaining to download: {len(to_download)}")
    
    if not to_download:
        print("\nAll papers already downloaded!")
        input("\nPress Enter to exit...")
        return
    
    print(f"\nStarting download...")
    print(f"Delay: {DELAY_MIN}-{DELAY_MAX} seconds between files")
    print(f"Will pause {BATCH_PAUSE}s every {BATCH_SIZE} files")
    print("-" * 60)
    
    # Create session
    session = requests.Session()
    
    # Statistics
    success_count = 0
    fail_count = 0
    not_available_count = 0
    start_time = datetime.now()
    consecutive_success = 0
    
    for i, (idx, raw_id) in enumerate(to_download):
        norm_id = normalize_id(raw_id)
        
        # Random delay
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        
        # Faster if many consecutive successes
        if consecutive_success > 20:
            delay = random.uniform(1.5, 2.5)
        
        time.sleep(delay)
        
        # Download
        status, message = download_paper(idx, raw_id, session, checkpoint)
        
        if status == "success":
            success_count += 1
            consecutive_success += 1
            print(f"[{idx:04d}] OK: {norm_id} -> {message}")
            
        elif status == "rate_limited":
            print(f"\n[{idx:04d}] RATE LIMITED! Waiting {RATE_LIMIT_PAUSE} seconds...")
            consecutive_success = 0
            save_checkpoint(checkpoint)
            time.sleep(RATE_LIMIT_PAUSE)
            # Retry this one
            status2, message2 = download_paper(idx, raw_id, session, checkpoint)
            if status2 == "success":
                success_count += 1
                print(f"[{idx:04d}] OK (retry): {norm_id}")
            else:
                fail_count += 1
                print(f"[{idx:04d}] FAILED (retry): {norm_id} - {message2}")
                
        elif status == "not_available":
            not_available_count += 1
            consecutive_success += 1  # Don't reset, this is not an error
            print(f"[{idx:04d}] 404: {norm_id} - HTML not available")
            
        else:
            fail_count += 1
            consecutive_success = 0
            print(f"[{idx:04d}] FAILED: {norm_id} - {message}")
        
        # Save checkpoint and take break every BATCH_SIZE files
        if (success_count + not_available_count) > 0 and (success_count + not_available_count) % BATCH_SIZE == 0:
            save_checkpoint(checkpoint)
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (success_count + not_available_count) / (elapsed / 60)
            print(f"\n--- Batch complete: {success_count} downloaded, {not_available_count} not available, {fail_count} failed ---")
            print(f"--- Speed: {rate:.1f} papers/minute ---")
            print(f"--- Pausing {BATCH_PAUSE} seconds... ---\n")
            time.sleep(BATCH_PAUSE)
    
    # Final save
    save_checkpoint(checkpoint)
    session.close()
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Not available (404): {not_available_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total in folder: {len(already_downloaded) + success_count}")
    print()
    print(f"  Files saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
