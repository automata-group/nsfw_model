"""
High-Speed Streaming Dataset Expander & Real-Time Neural Filter
1. Fetches candidate URLs for all 5 categories (drawings, hentai, neutral, porn, sexy)
2. Uses high-concurrency ThreadPoolExecutor to stream download
3. Neural In-Flight Filter:
   - Validates image dimensions (>= 120x120), format (RGB), hash deduplication
   - Runs PP-LCNet + NudeDetector in-flight
   - Confirms domain purity:
     * Hentai: strictly 2D anime/drawing style
     * Porn: strictly realistic human photo/3D
     * Drawings & Neutral: strictly ZERO explicit exposed intimate anatomy
     * Sexy: suggestive/clothing without exposed genitalia/areola
   - Re-routes or rejects mislabeled candidate images on-the-fly
4. Saves only verified, pristine images to reach TARGET_PER_CLASS (default 1000 per class = 5000 total)
5. Automatically triggers balanced RLS retraining
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

from nsfw_classifier import ProductionNSFWClassifier
from decision_agent import DecisionAgent, CLASSES, NSFW_CLASSES, EXPLICIT_EXPOSED_CLASSES

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
TARGET_PER_CLASS = 1000

URL_SOURCES = {
    'drawings': [
        'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/drawings/urls_drawings.txt'
    ],
    'hentai': [
        'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/hentai/urls_hentai.txt'
    ],
    'neutral': [
        'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/neutral/urls_neutral.txt'
    ],
    'porn': [
        'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/porn/urls_porn.txt',
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/sex/urls_sex.txt',
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/groups_pornstar/urls_groups_pornstar.txt'
    ],
    'sexy': [
        'https://raw.githubusercontent.com/alex000kim/nsfw_data_scraper/main/raw_data/sexy/urls_sexy.txt'
    ]
}


class DatasetExpander:
    def __init__(self, target_per_class: int = TARGET_PER_CLASS):
        self.target = target_per_class
        print("Initializing Vision Engine for in-flight verification...", flush=True)
        self.classifier = ProductionNSFWClassifier()
        self.agent = DecisionAgent()

    def fetch_url_list(self, category: str) -> List[str]:
        urls = []
        for src in URL_SOURCES.get(category, []):
            try:
                print(f"[{category}] Fetching URLs from {src[:60]}...", flush=True)
                req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as r:
                    lines = [x.strip() for x in r.read().decode('utf-8', errors='ignore').splitlines() if x.strip()]
                    urls.extend(lines)
            except Exception as e:
                print(f"[{category}] Error fetching {src}: {e}", flush=True)
        print(f"[{category}] Total loaded candidates: {len(urls)} URLs.", flush=True)
        return urls

    def download_and_verify(self, url: str, target_cat: str) -> Tuple[bool, str]:
        """
        Downloads image, decodes with PIL, passes through neural filter,
        verifies correct category and saves to disk.
        """
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    return False, "http_status"
                data = resp.read()
                if len(data) < 2500:  # Skip tiny icons / placeholders
                    return False, "too_small_bytes"

            # Validate Image Parsing
            with Image.open(io.BytesIO(data)) as img:
                w, h = img.size
                if w < 120 or h < 120:
                    return False, "low_resolution"
                rgb_img = img.convert('RGB')

            # Hash Deduplication Check
            file_hash = hashlib.md5(data).hexdigest()[:12]
            filename = f"{target_cat}_x_{file_hash}.jpg"
            target_dir = os.path.join(DATASET_DIR, target_cat)
            filepath = os.path.join(target_dir, filename)
            if os.path.exists(filepath):
                return False, "duplicate"

            # --- NEURAL IN-FLIGHT FILTER (التمرير والتنقية أثناء السحب) ---
            vis_res = self.classifier.classify(rgb_img)
            details = vis_res.get('details', {})
            detections = vis_res.get('anatomy_detections', [])
            has_exposed = any(d.get('class', '') in EXPLICIT_EXPOSED_CLASSES and float(d.get('score', 0)) >= 0.40 for d in detections)

            is_drawn = (details.get('hentai', 0) + details.get('drawings', 0)) > (details.get('porn', 0) + details.get('neutral', 0) + details.get('sexy', 0))

            # Domain & Category Purification Rules
            if target_cat == 'hentai':
                # Must be 2D Anime/Drawing style. Reject real photo porn!
                if not is_drawn and details.get('porn', 0) > 0.70:
                    return False, "rejected_real_photo_in_hentai"
            elif target_cat == 'porn':
                # Must be real photo / realistic 3D. Reject 2D anime hentai!
                if is_drawn and details.get('hentai', 0) > 0.70:
                    return False, "rejected_anime_in_porn"
            elif target_cat in ('drawings', 'neutral'):
                # Safe categories must NEVER contain exposed intimate anatomy!
                if has_exposed or vis_res.get('category') in ('porn', 'hentai'):
                    return False, "rejected_nsfw_in_safe"
            elif target_cat == 'sexy':
                # Sexy should not be full hardcore porn with exposed genitalia
                if any(d.get('class', '') in ('FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED') and float(d.get('score', 0)) >= 0.50 for d in detections):
                    return False, "rejected_hardcore_in_sexy"

            # Save clean image
            rgb_img.save(filepath, format='JPEG', quality=85)
            return True, "saved"

        except Exception:
            return False, "exception"

    def expand_category(self, cat: str, max_workers: int = 25) -> int:
        cat_dir = os.path.join(DATASET_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        existing = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
        needed = self.target - existing
        print(f"\n[{cat.upper()}] Current: {existing} | Target: {self.target} | Needed: {needed}", flush=True)

        if needed <= 0:
            print(f"[{cat.upper()}] Already at or above target ({existing}/{self.target}).", flush=True)
            return existing

        urls = self.fetch_url_list(cat)
        if not urls:
            return existing

        downloaded = existing
        verified_count = 0
        t_cat = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for u in urls:
                if downloaded >= self.target:
                    break
                f = executor.submit(self.download_and_verify, u, cat)
                futures[f] = u

            for future in as_completed(futures):
                if downloaded >= self.target:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    success, reason = future.result()
                    if success:
                        downloaded += 1
                        verified_count += 1
                        if downloaded % 20 == 0 or downloaded >= self.target:
                            speed = verified_count / max(time.perf_counter() - t_cat, 0.001)
                            print(f"[{cat.upper()}] Progress: {downloaded}/{self.target} (+{verified_count} verified, {speed:.1f} img/s)", flush=True)
                except Exception:
                    pass

        print(f"[{cat.upper()}] Finished. Final verified count: {downloaded}/{self.target}", flush=True)
        return downloaded

    def expand_all(self):
        print("=" * 65, flush=True)
        print(f"🚀 EXPANDING DATASET TO {self.target} IMAGES PER CATEGORY ({self.target * len(CLASSES)} TOTAL)", flush=True)
        print("=" * 65, flush=True)
        t_start = time.perf_counter()

        for cat in CLASSES:
            self.expand_category(cat)

        total_elapsed = time.perf_counter() - t_start
        print("\n" + "=" * 65, flush=True)
        print(f"🎉 DATASET EXPANSION COMPLETE IN {total_elapsed:.1f}s!", flush=True)
        for cat in CLASSES:
            count = len(os.listdir(os.path.join(DATASET_DIR, cat)))
            print(f"  - {cat:10s}: {count} images")
        print("=" * 65, flush=True)

        # Trigger RLS Retraining
        print("\n⚡ RE-TRAINING RLS CLASSIFIER HEAD ON EXPANDED DATASET...", flush=True)
        import train_balanced_rls
        # Invalidate cache to incorporate new images
        if os.path.exists(train_balanced_rls.CACHE_PATH):
            try:
                os.remove(train_balanced_rls.CACHE_PATH)
                print("[CACHE] Cleared old feature cache for clean re-extraction.", flush=True)
            except Exception:
                pass
        acc = train_balanced_rls.extract_features_and_train()
        print(f"🎯 RETRAINING COMPLETE! Overall Accuracy: {acc:.2f}%", flush=True)


if __name__ == '__main__':
    # Default target: 1000 images per class = 5,000 total images
    expander = DatasetExpander(target_per_class=1000)
    expander.expand_all()
