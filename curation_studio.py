"""
curation_studio.py
=============================================================================
Human-in-the-Loop Curation & Model Adaptation Studio
Provides:
  1. Automated identification of intercepted, borderline, and misclassified items.
  2. High-speed in-memory image streaming from ZIP archives.
  3. Real-time manual label persistence to human_curated_labels.json.
  4. Instant (<2s) closed-form RLS re-training and live ONNX weights injection.
=============================================================================
"""

import os
import io
import sys
import json
import time
import zipfile
import urllib.parse
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "nsfw_dataset_v1.zip")
CURATED_LABELS_PATH = os.path.join(BASE_DIR, "human_curated_labels.json")
QUEUE_CACHE_PATH = os.path.join(BASE_DIR, "curator_queue_cache.json")
MODEL_PATH = os.path.join(BASE_DIR, "nsfw_lcnet050.onnx")
WEIGHTS_JSON_PATH = os.path.join(BASE_DIR, "multimodal_anatomy_weights.json")

CATEGORIES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
CATEGORY_MAP = {
    "drawing": "drawings",
    "drawings": "drawings",
    "hentai": "hentai",
    "neutral": "neutral",
    "porn": "porn",
    "sexy": "sexy"
}

SAFE_CLASSES = {'drawings', 'neutral'}
NSFW_CLASSES = {'sexy', 'porn', 'hentai'}


class CurationStudioEngine:
    def __init__(self):
        self.curated_labels = self.load_curated_labels()
        self.review_queue = []
        self.zip_handle = None
        self._init_zip()
        self.build_or_load_review_queue()

    def _init_zip(self):
        if os.path.exists(ZIP_PATH):
            try:
                self.zip_handle = zipfile.ZipFile(ZIP_PATH, 'r')
            except Exception as e:
                print(f"[CurationStudio] Error opening ZIP: {e}")

    def load_curated_labels(self) -> Dict[str, Any]:
        if os.path.exists(CURATED_LABELS_PATH):
            try:
                with open(CURATED_LABELS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_curated_labels(self):
        try:
            with open(CURATED_LABELS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.curated_labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CurationStudio] Error saving labels: {e}")

    def build_or_load_review_queue(self):
        """Loads cached review queue or scans benchmark results to extract intercepted items."""
        if os.path.exists(QUEUE_CACHE_PATH):
            try:
                with open(QUEUE_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.review_queue = json.load(f)
                print(f"[CurationStudio] Loaded {len(self.review_queue)} review queue candidates.")
                return
            except Exception:
                pass

        print("[CurationStudio] Building candidate review queue from benchmark misclassifications & edge cases...")
        queue = []
        seen = set()

        # 1. Load misclassifications from benchmark results
        bench_path = os.path.join(BASE_DIR, "blind_benchmark_results.json")
        if os.path.exists(bench_path):
            try:
                with open(bench_path, "r", encoding="utf-8") as f:
                    bench = json.load(f)
                    for item in bench.get("sample_misclassifications", []):
                        fn = item.get("filename", "")
                        if fn and fn not in seen:
                            seen.add(fn)
                            gt = item.get("ground_truth", "")
                            pred = item.get("predicted", "")
                            is_intercepted_safe = (gt in SAFE_CLASSES and pred in NSFW_CLASSES)
                            is_leaked_nsfw = (gt in NSFW_CLASSES and pred in SAFE_CLASSES)
                            queue.append({
                                "id": fn,
                                "filename": fn,
                                "ground_truth": gt,
                                "predicted": pred,
                                "confidence": item.get("confidence", 0.5),
                                "is_intercepted_safe": is_intercepted_safe,
                                "is_leaked_nsfw": is_leaked_nsfw,
                                "reasoning": item.get("reasoning", ""),
                                "status": "pending"
                            })
            except Exception as e:
                print(f"[CurationStudio] Notice loading benchmark: {e}")

        # 2. Add candidates from challenge set
        chal_dir = os.path.join(BASE_DIR, "nsfw", "challenge_set")
        if os.path.exists(chal_dir):
            for f in os.listdir(chal_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and f not in seen:
                    seen.add(f)
                    queue.append({
                        "id": f"challenge:{f}",
                        "filename": f,
                        "ground_truth": "challenge",
                        "predicted": "drawings",
                        "confidence": 0.95,
                        "is_intercepted_safe": False,
                        "is_leaked_nsfw": False,
                        "reasoning": "Golden Challenge Anchor",
                        "status": "pending"
                    })

        # 3. If queue is small, sample interesting drawings and sexy images from zip
        if len(queue) < 100 and self.zip_handle is not None:
            for info in self.zip_handle.infolist()[:1000]:
                if info.is_dir():
                    continue
                fn = info.filename
                if fn not in seen and fn.lower().endswith(('.jpg', '.png')):
                    if 'drawing' in fn.lower() or 'sexy' in fn.lower():
                        seen.add(fn)
                        parts = fn.replace('\\', '/').split('/')
                        gt = parts[-2] if len(parts) >= 2 else "unknown"
                        queue.append({
                            "id": fn,
                            "filename": fn,
                            "ground_truth": CATEGORY_MAP.get(gt.lower(), gt),
                            "predicted": "drawings" if 'drawing' in fn.lower() else "sexy",
                            "confidence": 0.60,
                            "is_intercepted_safe": False,
                            "is_leaked_nsfw": False,
                            "reasoning": "Audit sample",
                            "status": "pending"
                        })
                        if len(queue) >= 300:
                            break

        self.review_queue = queue
        try:
            with open(QUEUE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.review_queue, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        print(f"[CurationStudio] Built review queue with {len(self.review_queue)} candidates.")

    def get_items(self, filter_type: str = "all", page: int = 1, page_size: int = 24) -> Dict[str, Any]:
        """Returns paginated, filtered items for the web UI."""
        items = []
        for it in self.review_queue:
            item_id = it["id"]
            curated = self.curated_labels.get(item_id)
            status = "curated" if curated else "pending"
            curated_cat = curated.get("category") if curated else None

            # Apply filters
            if filter_type == "pending" and status != "pending":
                continue
            elif filter_type == "curated" and status != "curated":
                continue
            elif filter_type == "intercepted_safe" and not it.get("is_intercepted_safe"):
                continue
            elif filter_type == "leaked_nsfw" and not it.get("is_leaked_nsfw"):
                continue

            entry = dict(it)
            entry["status"] = status
            entry["curated_category"] = curated_cat
            # Image URL endpoint
            entry["img_url"] = f"/curator-img/{urllib.parse.quote(it['filename'])}"
            items.append(entry)

        total_items = len(items)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_items = items[start_idx:end_idx]

        total_curated = len(self.curated_labels)
        total_pending = len(self.review_queue) - total_curated

        return {
            "items": page_items,
            "total_items": total_items,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total_items + page_size - 1) // page_size),
            "stats": {
                "total_candidates": len(self.review_queue),
                "curated_count": total_curated,
                "pending_count": max(0, total_pending)
            }
        }

    def set_label(self, item_id: str, new_category: str, is_safe: Optional[bool] = None) -> Dict[str, Any]:
        """Saves a human correction for a specific item."""
        if new_category not in CATEGORIES:
            return {"success": False, "error": f"Invalid category: {new_category}"}

        if is_safe is None:
            is_safe = bool(new_category in SAFE_CLASSES)

        self.curated_labels[item_id] = {
            "category": new_category,
            "is_safe": is_safe,
            "timestamp": time.time()
        }
        self.save_curated_labels()
        return {
            "success": True,
            "item_id": item_id,
            "new_category": new_category,
            "is_safe": is_safe,
            "curated_count": len(self.curated_labels)
        }

    def get_image_bytes(self, filename: str) -> Optional[bytes]:
        """Retrieves raw image bytes from ZIP or local challenge directory."""
        if filename.startswith("challenge:"):
            real_name = filename[len("challenge:"):]
            p = os.path.join(BASE_DIR, "nsfw", "challenge_set", real_name)
            if not os.path.exists(p):
                p = os.path.join(BASE_DIR, "challenge_set", real_name)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read()

        # Check local challenge set directly
        p_direct = os.path.join(BASE_DIR, "nsfw", "challenge_set", filename)
        if os.path.exists(p_direct):
            with open(p_direct, "rb") as f:
                return f.read()

        # Check ZIP archive
        if self.zip_handle is None:
            self._init_zip()
        if self.zip_handle is not None:
            try:
                return self.zip_handle.read(filename)
            except Exception:
                pass
        return None

    def execute_instant_retrain(self) -> Dict[str, Any]:
        """
        Executes immediate closed-form RLS retraining incorporating human feedback.
        Runs in ~2 seconds and live-updates the ONNX model and JSON weights.
        """
        import onnx
        from onnx import numpy_helper
        import pickle

        t0 = time.time()
        vis_cache_path = os.path.join(BASE_DIR, "v28k_visual_cache.pkl")
        anat_cache_path = os.path.join(BASE_DIR, "v28k_anatomy_cache.pkl")

        if not os.path.exists(vis_cache_path):
            return {"success": False, "error": "Visual embeddings cache not found. Please run train_28k_calibrated_rls.py first."}

        with open(vis_cache_path, "rb") as f:
            vis_cache = pickle.load(f)

        anat_cache = {}
        purified_labels = {}
        if os.path.exists(anat_cache_path):
            try:
                with open(anat_cache_path, "rb") as f:
                    saved = pickle.load(f)
                    anat_cache = saved.get("cache", {})
                    purified_labels = saved.get("labels", {})
            except Exception:
                pass

        # Build Design Matrix
        X_list = []
        y_list = []
        w_list = []

        # 1. Base dataset samples
        for fn, feat in vis_cache.items():
            parts = fn.replace('\\', '/').split('/')
            orig_cat = None
            for p in parts[:-1]:
                p_c = p.lower().strip()
                if p_c in CATEGORY_MAP:
                    orig_cat = CATEGORY_MAP[p_c]
                    break
            if not orig_cat:
                continue

            # Check if user curated this item
            user_override = self.curated_labels.get(fn)
            if user_override:
                effective_cat = user_override["category"]
                weight = 50.0  # Maximum weight on user's direct human feedback!
            else:
                effective_cat = purified_labels.get(fn, orig_cat)
                weight = 1.0

            a_feat = anat_cache.get(fn, np.zeros(4, dtype=np.float32))
            joint = np.concatenate([feat, a_feat])

            X_list.append(joint)
            y_list.append(CATEGORIES.index(effective_cat))
            w_list.append(weight)

        # 2. Challenge Anchors
        chal_dir = os.path.join(BASE_DIR, "nsfw", "challenge_set")
        from train_28k_calibrated_rls import CHALLENGE_ANCHORS, preprocess_pil, extract_anatomy_features
        import onnxruntime as ort
        from nudenet import NudeDetector

        om = onnx.load(MODEL_PATH)
        gemm_node = [n for n in om.graph.node if n.op_type == 'Gemm'][0]
        feat_name = gemm_node.input[0]
        om.graph.output.append(onnx.helper.make_tensor_value_info(feat_name, onnx.TensorProto.FLOAT, [None, 1280]))
        single_sess = ort.InferenceSession(om.SerializeToString(), providers=['CPUExecutionProvider'])
        single_inp = single_sess.get_inputs()[0].name
        det = NudeDetector(os.path.join(BASE_DIR, "nudenet_320n.onnx"))

        for rel_p, cat, w in CHALLENGE_ANCHORS:
            fp = os.path.join(chal_dir, rel_p)
            if not os.path.exists(fp):
                fp = os.path.join(BASE_DIR, "challenge_set", rel_p)
            if not os.path.exists(fp):
                continue
            with Image.open(fp) as img:
                inp = preprocess_pil(img)
                vf = single_sess.run([feat_name], {single_inp: inp})[0].flatten()
                dets = det.detect(np.array(img.convert('RGB')))
                af = extract_anatomy_features(dets)
                joint = np.concatenate([vf, af])
                X_list.append(joint)
                y_list.append(CATEGORIES.index(cat))
                w_list.append(w)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        W_sample = np.array(w_list, dtype=np.float32)

        N, D = X.shape
        C = len(CATEGORIES)
        Y_target = np.full((N, C), -2.0, dtype=np.float32)
        for i in range(N):
            Y_target[i, y[i]] = 2.5

        X_bias = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
        lambda_reg = 15.0
        sqrt_W = np.sqrt(W_sample)[:, np.newaxis]
        X_w = X_bias * sqrt_W
        Y_w = Y_target * sqrt_W

        A = X_w.T @ X_w + lambda_reg * np.eye(D + 1, dtype=np.float32)
        A[-1, -1] -= lambda_reg
        B = X_w.T @ Y_w

        Beta = np.linalg.solve(A, B)
        W_all = Beta[:-1, :].T
        bias = Beta[-1, :]
        W_vis = W_all[:, :1280]
        W_anat = W_all[:, 1280:]

        joint_logits = X @ W_all.T + bias
        preds = np.argmax(joint_logits, axis=1)
        acc = float(np.mean(preds == y)) * 100.0

        # Inject into ONNX models
        targets = [
            MODEL_PATH,
            os.path.join(BASE_DIR, "nsfw", "nsfw_lcnet050.onnx"),
            os.path.join(BASE_DIR, "nsfw", "microservice_go", "nsfw_lcnet050.onnx")
        ]
        for tp in targets:
            if os.path.exists(tp):
                m = onnx.load(tp)
                for init in m.graph.initializer:
                    if init.name == 'classifier.weight':
                        init.CopyFrom(numpy_helper.from_array(W_vis.astype(np.float32), name=init.name))
                    elif init.name == 'classifier.bias':
                        init.CopyFrom(numpy_helper.from_array(bias.astype(np.float32), name=init.name))
                onnx.save(m, tp)

        # Update JSON weights
        anat_dict = {
            "classes": CATEGORIES,
            "feature_names": ["max_exposed", "sum_exposed", "max_covered", "max_face"],
            "weights": W_anat.tolist(),
            "bias": bias.tolist()
        }
        for jp in [
            WEIGHTS_JSON_PATH,
            os.path.join(BASE_DIR, "nsfw", "multimodal_anatomy_weights.json"),
            os.path.join(BASE_DIR, "nsfw", "microservice_go", "multimodal_anatomy_weights.json")
        ]:
            if os.path.exists(os.path.dirname(jp)):
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(anat_dict, f, indent=2)

        elapsed = time.time() - t0
        return {
            "success": True,
            "accuracy": round(acc, 2),
            "samples_trained": N,
            "human_feedback_count": len(self.curated_labels),
            "elapsed_seconds": round(elapsed, 2)
        }


# Singleton engine instance
studio_engine = CurationStudioEngine()
