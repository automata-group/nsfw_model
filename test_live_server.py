import os
import urllib.request
import json
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.join(BASE_DIR, "challenge_set")

def check_live(name, rel_path, expected):
    full_path = os.path.join(CHALLENGE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
    if not os.path.exists(full_path):
        print(f"{name:36s} -> FILE NOT FOUND: {full_path}")
        return False
    with open(full_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    req = urllib.request.Request(
        'http://localhost:8765/api/classify',
        data=json.dumps({'image': b64}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        tag = 'NSFW' if res['is_nsfw'] else 'SAFE'
        status = 'PASS' if tag == expected else 'FAIL'
        print(f"{name:36s} -> {res['category']:10s} ({res['confidence']*100:5.1f}%) | {tag} [{status}]")
        return tag == expected

if __name__ == '__main__':
    print("=== FINAL LIVE SERVER HTTP API BENCHMARK ===")
    tests = [
        ('VIP IV Golden Ram (Asset)', 'vip_golden_ram.png', 'SAFE'),
        ('VIP XI Green Oval (Asset)', 'vip_green_oval.png', 'SAFE'),
        ('Gold Header Bar (Asset)', 'gold_header_bar.png', 'SAFE'),
        ('Green Frame Button (Asset)', 'green_frame_button.png', 'SAFE'),
        ('Green Banner Drape (Asset)', 'green_banner_drape.png', 'SAFE'),
        ('Treasure Chest (Asset)', 'treasure_chest.png', 'SAFE'),
        ('Padlock Badge (Asset)', 'padlock_badge.png', 'SAFE'),
        ('Turki Star Logo (Asset)', 'turki_star_logo.png', 'SAFE'),
        ('Strong Guy 3D (Asset)', 'strong_guy_3d_cartoon.png', 'SAFE'),
        ('Roulette Table Board (Asset)', 'roulette_table_board.png', 'SAFE'),
        ('Gold Red Slot Banner (Asset)', 'gold_red_slot_banner.png', 'SAFE'),
        ('Family Guy Poster (Drawing)', 'family_guy_cartoon_cast_poster.png', 'SAFE'),
        ('Tom & Jerry Cat (Drawing)', 'tom_cat_running_cartoon.png', 'SAFE'),
        ('Safe Cartoon Anime 1 (Drawing)', 'safe_cartoon_0100bde0991a6b7d.jpg', 'SAFE'),
        ('Safe Cartoon Anime 2 (Drawing)', 'safe_cartoon_01a3eab2829b117c.jpg', 'SAFE'),
        ('Anime Kiss Chest (NSFW)', 'anime_kiss_chest_erotic.png', 'NSFW'),
        ('3D Nude Render (NSFW)', 'nude_3d_render_woman.png', 'NSFW'),
        ('Doggy Intercourse (NSFW)', 'doggy_style_intercourse_bed.png', 'NSFW'),
        ('Pomegranate Booty (NSFW/Sexy)', 'pomegranate_booty_thong.png', 'NSFW'),
        ('Orange Bikini Beach (NSFW/Sexy)', 'orange_bikini_beach_buttocks.png', 'NSFW'),
        ('Palm Tree Bikini (NSFW/Sexy)', 'palm_tree_bikini_watermelon.png', 'NSFW'),
    ]

    passed = 0
    for name, path, expected in tests:
        if check_live(name, path, expected):
            passed += 1

    print(f"\n==========================================")
    print(f"BENCHMARK RESULT: {passed}/{len(tests)} PASSED ({passed/len(tests)*100:.1f}%)")
    print(f"==========================================")
