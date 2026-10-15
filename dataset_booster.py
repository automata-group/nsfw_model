import os
import io
import time
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
TARGET_PER_CLASS = 600

BOOSTER_SOURCES = {
    'porn': [
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/groups_pornstar/urls_groups_pornstar.txt',
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/sex/urls_sex.txt',
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/sex_anal/urls_sex_anal.txt',
        'https://raw.githubusercontent.com/EBazarov/nsfw_data_source_urls/master/raw_data/cum-play_cum/urls_cum-play_cum.txt'
    ]
}

def download_and_validate(url, category, save_dir):
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
            if len(data) < 2048:
                return False
            
            with Image.open(io.BytesIO(data)) as img:
                w, h = img.size
                if w < 120 or h < 120:
                    return False
                
                rgb = img.convert('RGB')
                file_hash = hashlib.md5(data).hexdigest()[:12]
                filename = f"{category}_b_{file_hash}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                if os.path.exists(filepath):
                    return False
                
                rgb.save(filepath, format='JPEG', quality=85)
                return True
    except Exception:
        return False

def boost_category(category):
    save_dir = os.path.join(DATASET_DIR, category)
    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    needed = TARGET_PER_CLASS - existing
    print(f"[{category}] Current count: {existing}. Target: {TARGET_PER_CLASS}. Still needed: {needed}")
    if needed <= 0:
        return
    
    all_urls = []
    for src in BOOSTER_SOURCES.get(category, []):
        try:
            req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                lines = [x.strip() for x in r.read().decode('utf-8', errors='ignore').splitlines() if x.strip()]
                all_urls.extend(lines)
        except Exception as e:
            print(f"Error fetching {src}: {e}")
            
    print(f"[{category}] Fetched {len(all_urls)} booster URLs.")
    downloaded = existing
    
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(download_and_validate, u, category, save_dir): u for u in all_urls}
        for future in as_completed(futures):
            if downloaded >= TARGET_PER_CLASS:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                if future.result():
                    downloaded += 1
                    if downloaded % 20 == 0 or downloaded >= TARGET_PER_CLASS:
                        print(f"[{category}] Boosted to: {downloaded}/{TARGET_PER_CLASS}")
            except Exception:
                pass
                
    print(f"[{category}] Final count: {downloaded}")

if __name__ == '__main__':
    boost_category('porn')
