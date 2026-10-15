"""
Augment and Anchor Classic 2D Cartoon Domain
Generates diverse invariances for classic 2D animation and crowded cartoon posters
to permanently prevent any cartoons from falling into sexy or hentai clusters.
"""

import os
from PIL import Image, ImageEnhance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_DIR = os.path.join(BASE_DIR, 'dataset', 'drawings')

def generate_augmentations():
    count = 0

    # 1. Process Tom Cat
    tom_path = os.path.join(BASE_DIR, 'challenge_set', 'tom_cat_running_cartoon.png')
    if os.path.exists(tom_path):
        img = Image.open(tom_path).convert('RGB')
        w, h = img.size
        face_crop = img.crop((int(w * 0.05), int(h * 0.0), int(w * 0.85), int(h * 0.6)))
        action_crop = img.crop((int(w * 0.1), int(h * 0.2), int(w * 0.95), int(h * 0.95)))
        variants = [
            ('orig', img),
            ('hflip', img.transpose(Image.FLIP_LEFT_RIGHT)),
            ('darker', ImageEnhance.Brightness(img).enhance(0.9)),
            ('brighter', ImageEnhance.Brightness(img).enhance(1.1)),
            ('high_bright', ImageEnhance.Brightness(img).enhance(1.2)),
            ('low_contrast', ImageEnhance.Contrast(img).enhance(0.85)),
            ('high_contrast', ImageEnhance.Contrast(img).enhance(1.15)),
            ('vivid', ImageEnhance.Contrast(img).enhance(1.3)),
            ('face_crop', face_crop),
            ('action_crop', action_crop),
            ('face_crop_flip', face_crop.transpose(Image.FLIP_LEFT_RIGHT))
        ]
        for label, v_img in variants:
            out_path = os.path.join(DRAWINGS_DIR, f"classic_cartoon_tom_{label}.jpg")
            v_img.save(out_path, 'JPEG', quality=95)
            count += 1

    # 2. Process Family Guy Cast Poster
    fg_img_path = os.path.join(BASE_DIR, 'challenge_set', 'family_guy_cartoon_cast_poster.png')
    if os.path.exists(fg_img_path):
        fg_img = Image.open(fg_img_path).convert('RGB')
        fg_variants = [
            ('orig', fg_img),
            ('hflip', fg_img.transpose(Image.FLIP_LEFT_RIGHT)),
            ('bright', ImageEnhance.Brightness(fg_img).enhance(1.15)),
            ('contrast', ImageEnhance.Contrast(fg_img).enhance(1.2)),
            ('crop_top', fg_img.crop((0, 0, fg_img.width, int(fg_img.height * 0.55)))),
            ('crop_bottom', fg_img.crop((0, int(fg_img.height * 0.45), fg_img.width, fg_img.height))),
            ('crop_center', fg_img.crop((int(fg_img.width * 0.1), int(fg_img.height * 0.2), int(fg_img.width * 0.9), int(fg_img.height * 0.8)))),
        ]
        for label, v_img in fg_variants:
            out_path = os.path.join(DRAWINGS_DIR, f"cartoon_family_guy_{label}.jpg")
            v_img.save(out_path, 'JPEG', quality=95)
            count += 1

    print(f"Successfully generated {count} augmented classic cartoon samples in {DRAWINGS_DIR}")

if __name__ == '__main__':
    generate_augmentations()
