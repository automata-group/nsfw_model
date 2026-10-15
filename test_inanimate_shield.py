import os, numpy as np
from PIL import Image
import onnxruntime as ort
from nudenet import NudeDetector

det = NudeDetector('nudenet_320n.onnx')
sess = ort.InferenceSession('nsfw_lcnet050.onnx')
inp_name = sess.get_inputs()[0].name
CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
NSFW_CLASSES = {'hentai', 'porn', 'sexy'}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def classify_hybrid(img_path):
    im = Image.open(img_path).convert('RGB')
    resized = im.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    arr = (arr - MEAN) / STD
    
    logits = sess.run(None, {inp_name: arr.astype(np.float32)})[0]
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = (exp_l / np.sum(exp_l, axis=-1, keepdims=True))[0]
    
    top_idx = int(np.argmax(probs))
    cat = CLASSES[top_idx]
    conf = float(probs[top_idx])
    is_nsfw = bool(cat in NSFW_CLASSES)
    details = {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}
    
    # Detector
    detections = det.detect(np.array(im))
    explicit_labels = {
        'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_EXPOSED', 'ANUS_EXPOSED'
    }
    explicit_hits = [
        d for d in detections 
        if d.get('class') in explicit_labels and float(d.get('score', 0)) >= 0.40
    ]
    
    has_human = any(
        d.get('class') in (
            'FACE_FEMALE', 'FACE_MALE',
            'FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED',
            'FEMALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_COVERED',
            'MALE_GENITALIA_EXPOSED', 'MALE_BREAST_EXPOSED',
            'BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED',
            'BELLY_EXPOSED', 'BELLY_COVERED',
            'ANUS_EXPOSED', 'ANUS_COVERED'
        ) and float(d.get('score', 0)) >= 0.30
        for d in detections
    )
    
    if explicit_hits:
        max_score = max(float(d['score']) for d in explicit_hits)
        if cat in ('drawings', 'hentai') or details.get('drawings', 0) > 0.35:
            cat = 'hentai'
        else:
            cat = 'porn'
        is_nsfw = True
        conf = max(max_score, 0.90)
    elif cat in ('hentai', 'porn'):
        # Inanimate graphic / symbol / logo veto
        if not has_human and conf < 0.85:
            cat = 'drawings' if details.get('drawings', 0) >= details.get('neutral', 0) else 'neutral'
            is_nsfw = False
            conf = max(float(details.get(cat, 0.90)), 0.80)
            
    return cat, conf, is_nsfw

images = [
    ('Treasure Chest (Workspace)', 'crown_badge_regularized.png', 'SAFE'),
    ('Treasure Chest (Cropped)', r'challenge_set\treasure_chest.png', 'SAFE'),
    ('Padlock Badge', r'challenge_set\padlock_badge.png', 'SAFE'),
    ('Turki Star Logo', r'challenge_set\turki_star_logo.png', 'SAFE'),
    ('Strong Guy 3D', r'challenge_set\strong_guy_3d_cartoon.png', 'SAFE'),
    ('Covered Dress Woman', r'challenge_set\covered_dress_woman.jpg', 'SAFE'),
    ('Green Square', r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792025228518.jpg', 'SAFE'),
    ('Anime Kiss Chest', r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792010528312.jpg', 'NSFW'),
    ('Hentai Webpage', r'challenge_set\hentai_mosaic_webpage.png', 'NSFW'),
    ('3D Nude Render', r'challenge_set\nude_3d_render_woman.png', 'NSFW'),
    ('Doggy Intercourse', r'challenge_set\doggy_style_intercourse_bed.png', 'NSFW'),
    ('Real Porn Spread', r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792010235112.jpg', 'NSFW'),
]

print("=== VERIFYING INANIMATE GRAPHIC SHIELD ===")
for name, p, expected in images:
    if os.path.exists(p):
        cat, conf, is_nsfw = classify_hybrid(p)
        tag = 'NSFW' if is_nsfw else 'SAFE'
        status = 'PASS' if tag == expected else 'FAIL'
        print(f"{name:30s} -> {cat:10s} ({conf*100:5.1f}%) | {tag} [{status}]")
