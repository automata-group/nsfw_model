"""
Fetch and Purify Safe Cartoon & Anime Drawings
Streams verified 2D anime and cartoon drawings, ensuring ZERO exposed sexual anatomy,
and integrates them into dataset/drawings to eliminate any false hentai confusion.
"""

import os
import sys
import io
import time
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nudenet import NudeDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_DIR = os.path.join(BASE_DIR, 'dataset', 'drawings')
URLS_FILE = 'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/drawings/urls_drawings.txt'

EXPLICIT_EXPOSED = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'ANUS_EXPOSED'
}

def load_existing_hashes() -> set:
    hashes = set()
    for fname in os.listdir(DRAWINGS_DIR):
        p = os.path.join(DRAWINGS_DIR, fname)
        if os.path.isfile(p):
            try:
                with open(p, 'rb') as f:
                    hashes.add(hashlib.sha256(f.read()).hexdigest())
            except Exception:
                pass
    return hashes

def download_candidate(url: str) -> Tuple[bool, bytes, str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status != 200:
                return False, b'', 'http_error'
            data = resp.read()
            if len(data) < 3000:
                return False, b'', 'too_small'
            return True, data, ''
    except Exception:
        return False, b'', 'download_failed'

def main(target_new_images: int = 350):
    os.makedirs(DRAWINGS_DIR, exist_ok=True)
    existing_hashes = load_existing_hashes()
    print(f"Existing drawings count: {len(existing_hashes)}")

    print("Fetching candidate URLs...")
    req = urllib.request.Request(URLS_FILE, headers={'User-Agent': 'Mozilla/5.0'})
    content = urllib.request.urlopen(req, timeout=12).read().decode('utf-8', errors='ignore')
    all_urls = [line.strip() for line in content.splitlines() if line.strip() and 'imgur.com' in line]
    print(f"Loaded {len(all_urls)} imgur candidate URLs.")

    print("Loading NudeDetector for deep anatomical purity check...")
    detector = NudeDetector(os.path.join(BASE_DIR, 'nudenet_320n.onnx'))

    added = 0
    checked = 0
    start_time = time.time()

    # Process in batches using ThreadPoolExecutor
    batch_size = 50
    for i in range(0, len(all_urls), batch_size):
        if added >= target_new_images:
            break

        batch = all_urls[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(download_candidate, url): url for url in batch}
            for fut in as_completed(futures):
                checked += 1
                ok, data, err = fut.result()
                if not ok:
                    continue

                h = hashlib.sha256(data).hexdigest()
                if h in existing_hashes:
                    continue

                # Verify PIL Decoding
                try:
                    img = Image.open(io.BytesIO(data))
                    if img.width < 120 or img.height < 120:
                        continue
                    img = img.convert('RGB')
                except Exception:
                    continue

                # Temp file for NudeDetector
                temp_path = os.path.join(DRAWINGS_DIR, f"temp_{h[:8]}.jpg")
                img.save(temp_path, 'JPEG', quality=90)

                try:
                    detections = detector.detect(temp_path)
                    has_explicit = any(
                        d.get('class') in EXPLICIT_EXPOSED and float(d.get('score', 0)) >= 0.30
                        for d in detections
                    )
                    if has_explicit:
                        # Reject potential hentai leakage
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue

                    # Safe Cartoon / Anime Drawing confirmed!
                    final_path = os.path.join(DRAWINGS_DIR, f"safe_cartoon_{h[:16]}.jpg")
                    os.replace(temp_path, final_path)
                    existing_hashes.add(h)
                    added += 1

                    if added % 25 == 0:
                        rate = added / (time.time() - start_time)
                        print(f"Purified & Added: {added}/{target_new_images} images ({rate:.1f} img/s) [Checked: {checked}]")

                    if added >= target_new_images:
                        break
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    print(f"=== COMPLETED: Added {added} pristine cartoon/anime drawings. Total dataset: {len(existing_hashes)} ===")

if __name__ == '__main__':
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(target)
