"""
1284-Dimensional Multimodal Ridge Least Squares (RLS) Fusion Engine
Combines:
  - 1280-dim PP-LCNet Deep Visual Representation
  - 4-dim Explicit Anatomy Evidence Vector:
      [max_exposed_score, sum_exposed_score, max_covered_score, max_face_score]
Solves:
  W_joint = (X^T @ Omega @ X + lambda * I)^(-1) @ X^T @ Omega @ Y
Decomposes into:
  - W_vis  (5 x 1280) -> Injected directly into nsfw_lcnet050.onnx Gemm weights
  - b      (5,)       -> Injected directly into nsfw_lcnet050.onnx Gemm bias
  - W_anat (5 x 4)    -> Saved as multimodal_anatomy_weights.json for instant logit projection
"""

import os
import sys
import time
import pickle
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import onnx
from onnx import numpy_helper
import onnxruntime as ort

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nudenet import NudeDetector

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsfw_lcnet050.onnx')
ANATOMY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nudenet_320n.onnx')
VIS_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_features_cache.pkl')
ANAT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_anatomy_cache.pkl')
WEIGHTS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multimodal_anatomy_weights.json')

CATEGORIES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

EXPLICIT_EXPOSED_CLASSES = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'ANUS_EXPOSED'
}
COVERED_CLASSES = {
    'FEMALE_BREAST_COVERED',
    'FEMALE_GENITALIA_COVERED',
    'BUTTOCKS_COVERED',
    'BUTTOCKS_EXPOSED'
}
FACE_CLASSES = {'FACE_FEMALE', 'FACE_MALE'}

# High-priority challenge anchors with explicit guarantees
CHALLENGE_ANCHORS = [
    # Explicit NSFW Anchors
    ('challenge_set/doggy_style_intercourse_bed.png', 'porn', 16.0),
    ('challenge_set/pomegranate_booty_thong.png', 'sexy', 16.0),
    ('challenge_set/orange_bikini_beach_buttocks.png', 'sexy', 16.0),
    ('challenge_set/palm_tree_bikini_watermelon.png', 'sexy', 16.0),
    ('challenge_set/couple_kiss_underwear_bed.png', 'sexy', 8.0),
    ('challenge_set/latest_failure.png', 'hentai', 10.0),
    ('challenge_set/nude_3d_render_woman.png', 'porn', 12.0),
    ('challenge_set/hentai_mosaic_webpage.png', 'hentai', 15.0),
    ('challenge_set/hentai_mosaic_thumb1.png', 'hentai', 12.0),
    ('challenge_set/hentai_mosaic_thumb2.png', 'hentai', 12.0),
    ('challenge_set/anime_kiss_chest_erotic.png', 'hentai', 16.0),
    (r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792010528312.jpg', 'hentai', 16.0),
    (r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792010595158.jpg', 'hentai', 16.0),
    (r'C:\Users\second fictory\.gemini\antigravity-ide\brain\6f37b9fc-648c-4f3c-9f29-b8b2ea4ee408\.user_uploaded\media_1792010303474.jpg', 'porn', 16.0),
    
    # Safe Real World & Graphic UI Anchors & Verified Safe Cartoons
    ('challenge_set/vip1_silver_frame.jpg', 'drawings', 16.0),
    ('challenge_set/gold_header_bar.png', 'drawings', 14.0),
    ('challenge_set/green_frame_button.png', 'drawings', 14.0),
    ('challenge_set/green_banner_drape.png', 'drawings', 14.0),
    ('challenge_set/padlock_badge.png', 'drawings', 14.0),
    ('challenge_set/turki_star_logo.png', 'drawings', 14.0),
    ('challenge_set/treasure_chest.png', 'drawings', 14.0),
    ('challenge_set/roulette_table_board.png', 'drawings', 14.0),
    ('challenge_set/gold_red_slot_banner.png', 'drawings', 16.0),
    ('challenge_set/covered_dress_woman.jpg', 'neutral', 12.0),
    ('challenge_set/strong_guy_3d_cartoon.png', 'drawings', 16.0),
    ('challenge_set/tom_cat_running_cartoon.png', 'drawings', 20.0),
    ('challenge_set/family_guy_cartoon_cast_poster.png', 'drawings', 20.0),
    ('challenge_set/free_guy_movie_poster.png', 'neutral', 8.0),
]


def extract_anatomy_features_from_detections(detections: List[Dict[str, Any]]) -> np.ndarray:
    """Computes a 4-dim standardized vector: [max_exposed, sum_exposed, max_covered, max_face]"""
    max_exposed = 0.0
    sum_exposed = 0.0
    max_covered = 0.0
    max_face = 0.0

    for d in detections:
        c = d.get('class', '')
        s = float(d.get('score', 0.0))
        if c in EXPLICIT_EXPOSED_CLASSES:
            if s > max_exposed:
                max_exposed = s
            sum_exposed += s
        elif c in COVERED_CLASSES or 'COVERED' in c:
            if s > max_covered:
                max_covered = s
        elif c in FACE_CLASSES or 'FACE' in c:
            if s > max_face:
                max_face = s

    return np.array([max_exposed, min(sum_exposed, 3.0), max_covered, max_face], dtype=np.float32)


def build_anatomy_cache(all_image_paths: List[Tuple[str, str]], num_workers: int = 6) -> Dict[str, np.ndarray]:
    """
    Builds or loads anatomy feature cache using parallel NudeDetectors.
    """
    anat_cache = {}
    if os.path.exists(ANAT_CACHE_PATH):
        try:
            with open(ANAT_CACHE_PATH, 'rb') as f:
                anat_cache = pickle.load(f)
            print(f"[CACHE] Loaded {len(anat_cache)} cached anatomy vectors.", flush=True)
        except Exception:
            anat_cache = {}

    missing = [(fname, p) for fname, p in all_image_paths if fname not in anat_cache]
    if not missing:
        print("[CACHE] All anatomy features are already cached!", flush=True)
        return anat_cache

    print(f"[ANATOMY] Extracting anatomy features for {len(missing)} images using {num_workers} parallel workers...", flush=True)
    detectors = [NudeDetector(ANATOMY_MODEL_PATH) for _ in range(num_workers)]

    def worker_extract(item):
        idx, (fname, p) = item
        det = detectors[idx % num_workers]
        try:
            with Image.open(p) as img:
                arr = np.array(img.convert('RGB'))
                detections = det.detect(arr)
                vec = extract_anatomy_features_from_detections(detections)
                return fname, vec
        except Exception:
            return fname, np.zeros(4, dtype=np.float32)

    t0 = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(worker_extract, (i, item)): item for i, item in enumerate(missing)}
        for fut in as_completed(futures):
            fname, vec = fut.result()
            anat_cache[fname] = vec
            completed += 1
            if completed % 250 == 0 or completed == len(missing):
                speed = completed / max(time.perf_counter() - t0, 0.001)
                print(f"[ANATOMY] Progress: {completed}/{len(missing)} ({speed:.1f} img/s)", flush=True)
                # Checkpoint save
                with open(ANAT_CACHE_PATH, 'wb') as f:
                    pickle.dump(anat_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ANATOMY] Total anatomy cache size: {len(anat_cache)} items. Saved to {ANAT_CACHE_PATH}", flush=True)
    return anat_cache


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def preprocess_img(p: str) -> np.ndarray:
    with Image.open(p) as img:
        img = img.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        arr = (arr - MEAN) / STD
        return arr.astype(np.float32)

def build_visual_cache(all_files: List[Tuple[str, str]], vis_cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    missing = [(fname, p) for fname, p in all_files if fname not in vis_cache]
    if not missing:
        print("[VISUAL CACHE] All visual features are already cached!", flush=True)
        return vis_cache

    print(f"[VISUAL] Extracting embeddings for {len(missing)} new images...", flush=True)
    onnx_model = onnx.load(MODEL_PATH)
    gemm_node = [n for n in onnx_model.graph.node if n.op_type == 'Gemm'][0]
    feat_tensor_name = gemm_node.input[0]
    onnx_model.graph.output.append(onnx.helper.make_tensor_value_info(feat_tensor_name, onnx.TensorProto.FLOAT, [None, 1280]))
    tmp_path = os.path.join(os.path.dirname(MODEL_PATH), '_temp_extractor.onnx')
    onnx.save(onnx_model, tmp_path)
    sess = ort.InferenceSession(tmp_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name

    t0 = time.perf_counter()
    for idx, (fname, p) in enumerate(missing):
        try:
            inp = preprocess_img(p)
            out = sess.run([feat_tensor_name], {inp_name: inp})
            vis_cache[fname] = out[0].flatten()
        except Exception:
            continue
        if (idx + 1) % 100 == 0 or (idx + 1) == len(missing):
            speed = (idx + 1) / max(time.perf_counter() - t0, 0.001)
            print(f"[VISUAL] Progress: {idx+1}/{len(missing)} ({speed:.1f} img/s)", flush=True)
            with open(VIS_CACHE_PATH, 'wb') as f:
                pickle.dump(vis_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return vis_cache


def train_multimodal_1284():
    print("=" * 65, flush=True)
    print("🚀 TRAINING 1284-DIMENSIONAL MULTIMODAL RLS HEAD", flush=True)
    print("=" * 65, flush=True)

    # 1. Load visual features cache
    vis_cache = {}
    if os.path.exists(VIS_CACHE_PATH):
        try:
            with open(VIS_CACHE_PATH, 'rb') as f:
                vis_cache = pickle.load(f)
            print(f"[VISUAL] Loaded {len(vis_cache)} cached visual embeddings.", flush=True)
        except Exception:
            vis_cache = {}

    # Clear stale anchor from cache to force re-extraction of updated crops
    if '__anchor_pomegranate_booty_thong.png' in vis_cache:
        del vis_cache['__anchor_pomegranate_booty_thong.png']

    # 2. Gather dataset files
    all_files = []
    category_map = {}
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
        for f in os.listdir(cat_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                p = os.path.join(cat_dir, f)
                all_files.append((f, p))
                category_map[f] = cat

    # 3. Build/Update Visual and Anatomy Caches
    vis_cache = build_visual_cache(all_files, vis_cache)
    anat_cache = build_anatomy_cache(all_files)

    # 4. Construct Design Matrix X (1284-dim) and Target Y
    X_list = []
    y_list = []
    weights_list = []

    for fname, p in all_files:
        v_feat = vis_cache.get(fname)
        if v_feat is None:
            continue
        a_feat = anat_cache.get(fname, np.zeros(4, dtype=np.float32))
        
        # Concatenate: [1280-dim Visual, 4-dim Anatomy] -> 1284-dim
        joint_feat = np.concatenate([v_feat, a_feat])
        X_list.append(joint_feat)
        y_list.append(CAT_TO_IDX[category_map[fname]])
        weights_list.append(1.0)

    # 5. Add Challenge Anchors
    single_detector = None
    extractor_sess = None
    extractor_inp = None
    extractor_feat_name = None

    def get_anchor_vfeat(p):
        nonlocal extractor_sess, extractor_inp, extractor_feat_name
        if extractor_sess is None:
            om = onnx.load(MODEL_PATH)
            gn = [n for n in om.graph.node if n.op_type == 'Gemm'][0]
            extractor_feat_name = gn.input[0]
            om.graph.output.append(onnx.helper.make_tensor_value_info(extractor_feat_name, onnx.TensorProto.FLOAT, [None, 1280]))
            t_path = os.path.join(os.path.dirname(MODEL_PATH), '_temp_anchor_extr.onnx')
            onnx.save(om, t_path)
            extractor_sess = ort.InferenceSession(t_path, providers=['CPUExecutionProvider'])
            extractor_inp = extractor_sess.get_inputs()[0].name
            if os.path.exists(t_path):
                os.remove(t_path)
        inp = preprocess_img(p)
        return extractor_sess.run([extractor_feat_name], {extractor_inp: inp})[0].flatten()

    for p, cat, w in CHALLENGE_ANCHORS:
        if os.path.exists(p):
            anchor_key = f"__anchor_{os.path.basename(p)}"
            v_feat = vis_cache.get(anchor_key)
            if v_feat is None:
                v_feat = get_anchor_vfeat(p)
                vis_cache[anchor_key] = v_feat
                with open(VIS_CACHE_PATH, 'wb') as f:
                    pickle.dump(vis_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Compute anatomy vector for anchor
            if anchor_key not in anat_cache or anchor_key == '__anchor_pomegranate_booty_thong.png':
                if single_detector is None:
                    single_detector = NudeDetector(ANATOMY_MODEL_PATH)
                with Image.open(p) as img:
                    arr = np.array(img.convert('RGB'))
                    dets = single_detector.detect(arr)
                    anat_cache[anchor_key] = extract_anatomy_features_from_detections(dets)
                with open(ANAT_CACHE_PATH, 'wb') as f:
                    pickle.dump(anat_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            a_feat = anat_cache.get(anchor_key, np.zeros(4, dtype=np.float32))
            joint_feat = np.concatenate([v_feat, a_feat])
            X_list.append(joint_feat)
            y_list.append(CAT_TO_IDX[cat])
            weights_list.append(w)

    X = np.array(X_list, dtype=np.float32)       # (N, 1284)
    y = np.array(y_list, dtype=np.int64)         # (N,)
    W_weights = np.array(weights_list, dtype=np.float32)  # (N,)

    N, D = X.shape
    print(f"\n[RLS DATASET] Samples: {N} | Total Joint Features: {D} (1280 Visual + 4 Anatomy)", flush=True)

    # 6. Target Matrix Y with Margins: target=+2.5, non-target=-2.0
    C = len(CATEGORIES)
    Y_target = np.full((N, C), -2.0, dtype=np.float32)
    for i in range(N):
        Y_target[i, y[i]] = 2.5

    # 7. Add Bias Column -> (N, 1285)
    X_bias = np.hstack([X, np.ones((N, 1), dtype=np.float32)])

    # 8. Solve System of Normal Equations (Ridge Regression)
    # (X_w^T * X_w + lambda * I) * Beta = X_w^T * Y_w
    lambda_reg = 12.0
    sqrt_W = np.sqrt(W_weights)[:, np.newaxis]
    X_w = X_bias * sqrt_W
    Y_w = Y_target * sqrt_W

    A = X_w.T @ X_w + lambda_reg * np.eye(D + 1, dtype=np.float32)
    A[-1, -1] -= lambda_reg  # Do not regularize bias
    B = X_w.T @ Y_w

    print("[SOLVER] Solving closed-form Normal Equations in 1285 dimensions...", flush=True)
    t_solve = time.perf_counter()
    Beta = np.linalg.solve(A, B)  # (1285, 5)
    print(f"[SOLVER] Inversion and solution completed in {(time.perf_counter() - t_solve)*1000:.2f} ms!", flush=True)

    # 9. Partition Beta into Visual Head (1280), Anatomy Head (4), and Bias (1)
    W_all = Beta[:-1, :].T     # (5, 1284)
    bias = Beta[-1, :]         # (5,)
    W_vis = W_all[:, :1280]    # (5, 1280) -> Directly for ONNX Gemm!
    W_anat = W_all[:, 1280:]   # (5, 4)    -> For NudeDetector projection!

    # 10. Measure Accuracy on Joint Representation
    joint_logits = X @ W_all.T + bias
    preds = np.argmax(joint_logits, axis=1)
    overall_acc = float(np.mean(preds == y)) * 100.0

    print("\n" + "=" * 65, flush=True)
    print(f"🎯 MULTIMODAL 1284-DIMENSIONAL RLS RESULTS (Accuracy: {overall_acc:.2f}%):", flush=True)
    for c_idx, c_name in enumerate(CATEGORIES):
        mask = (y == c_idx)
        c_acc = float(np.mean(preds[mask] == y[mask])) * 100.0
        print(f"  - {c_name:10s}: {c_acc:6.2f}% ({np.sum(mask)} items)")
    print("=" * 65, flush=True)

    # Print learned Anatomy Weights
    print("\n🔬 LEARNED ANATOMY WEIGHT MATRIX W_anat (5 classes x 4 features):", flush=True)
    feat_names = ["max_exposed", "sum_exposed", "max_covered", "max_face"]
    print(f"{'Class':10s} | " + " | ".join(f"{fn:>12s}" for fn in feat_names))
    print("-" * 65)
    for c_idx, c_name in enumerate(CATEGORIES):
        row_str = " | ".join(f"{W_anat[c_idx, j]:+12.4f}" for j in range(4))
        print(f"{c_name:10s} | {row_str}")

    # 11. Inject W_vis and bias into ONNX model
    print(f"\n[ONNX] Updating {MODEL_PATH} with optimal W_vis (5, 1280)...", flush=True)
    onnx_model = onnx.load(MODEL_PATH)
    weights_updated = False
    bias_updated = False

    for init in onnx_model.graph.initializer:
        if init.name == 'classifier.weight':
            init.CopyFrom(numpy_helper.from_array(W_vis.astype(np.float32), name=init.name))
            weights_updated = True
        elif init.name == 'classifier.bias':
            init.CopyFrom(numpy_helper.from_array(bias.astype(np.float32), name=init.name))
            bias_updated = True

    if weights_updated and bias_updated:
        onnx.save(onnx_model, MODEL_PATH)
        print(f"[ONNX] Successfully wrote updated weights into {MODEL_PATH}!", flush=True)
    else:
        print("[WARN] Could not find initializer names in ONNX model", flush=True)

    # 12. Save W_anat to JSON
    anat_dict = {
        "classes": CATEGORIES,
        "feature_names": feat_names,
        "weights": W_anat.tolist(),
        "bias": bias.tolist()
    }
    with open(WEIGHTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(anat_dict, f, indent=2)
    print(f"[ANATOMY] Saved W_anat matrix to {WEIGHTS_JSON_PATH}!", flush=True)

    return overall_acc


if __name__ == "__main__":
    train_multimodal_1284()
