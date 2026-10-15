"""
Decision Agent & Asymmetric Reward Optimization Engine
Combines:
  1. Vision Model (PP-LCNet-050 + NudeDetector Anatomy Gate)
  2. Decision Agent (Logic Shaping, Contradiction Resolution, Contextual Policy)
  3. Mathematical Asymmetric Reward System (Confidence Calibration & Logic Reward)
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
NSFW_CLASSES = {'porn', 'hentai', 'sexy'}
SAFE_CLASSES = {'drawings', 'neutral'}

# Explicit intimate nudity classes (عري صريح حقيقي)
# استثناء البطن (BELLY) والإبط (ARMPITS) والقدمين (FEET) وصدر الرجال لأنها ليست إباحية صريحة
EXPLICIT_EXPOSED_CLASSES = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'ANUS_EXPOSED'
}


def calculate_rewards(json_output: Dict[str, Any], agent_action: str, true_label: str, has_true_exposed_nudity: bool) -> Dict[str, float]:
    """
    Computes mathematically calibrated rewards for both the Vision Model and the Decision Agent.
    """
    details = json_output.get("details", {})
    detections = json_output.get("anatomy_detections", [])
    
    # Analyze anatomical detections: only genuine explicit intimate parts count as EXPOSED
    detected_exposed = any(
        d.get("class", "") in EXPLICIT_EXPOSED_CLASSES and float(d.get("score", 0)) >= 0.40 
        for d in detections
    )
    detected_covered = any(
        ("COVERED" in d.get("class", "") or d.get("class", "") in ("FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED"))
        and float(d.get("score", 0)) >= 0.35
        for d in detections
    )
    detected_only_covered = (len(detections) > 0) and not detected_exposed

    # ==========================================
    # 1. Decision Agent Reward (مكافأة نموذج الوكيل)
    # ==========================================
    agent_reward = 0.0
    agent_is_nsfw = agent_action in NSFW_CLASSES
    true_is_nsfw = true_label in NSFW_CLASSES
    
    if agent_action == true_label:
        # قرار مطابق تماماً للحقيقة
        agent_reward += 2.0 if (true_label in NSFW_CLASSES or true_label == "sexy") else 1.0
    else:
        # عقوبات غير متماثلة حسب خطورة الخطأ
        if true_is_nsfw and not agent_is_nsfw:
            agent_reward -= 5.0  # خطأ كارثي (False Negative): تمرير محتوى إباحي
        elif true_label == "sexy" and agent_is_nsfw:
            agent_reward -= 2.0  # عقوبة متوسطة (False Positive): تشدد غير مبرر ضد ملابس مثيرة
        elif true_label in ("neutral", "drawings") and agent_is_nsfw:
            agent_reward -= 3.0  # عقوبة قاسية: إزعاج المستخدمين بحجب صور عادية
        else:
            agent_reward -= 1.0  # أخطاء فرعية أخرى

    # مكافأة/عقوبة الذكاء المنطقي للوكيل (Shaping Reward - R_logic)
    logic_shaping = 0.0
    
    # حل التناقض: نسبة porn مرتفعة ولكن الأعضاء مغطاة فقط -> الوكيل خفض التصنيف إلى sexy
    porn_prob = details.get("porn", 0.0)
    if (0.50 <= porn_prob <= 0.75) and detected_only_covered:
        if agent_action == "sexy" and true_label == "sexy":
            logic_shaping += 1.5  # مكافأة إضافية لحل التناقض وتصحيح وهم الرؤية!
            
    # عقوبة التهاون: رصد عضو مكشوف واضح وتجاهله الوكيل
    if detected_exposed and not agent_is_nsfw and has_true_exposed_nudity:
        logic_shaping -= 3.0  # عقوبة قاسية لتجاهل عضو مكشوف صريح

    agent_reward += logic_shaping

    # ==========================================
    # 2. Vision Model Reward (مكافأة نموذج الصور)
    # ==========================================
    vision_pred_class = json_output.get("category", "")
    vision_conf = float(json_output.get("confidence", 0.5))
    
    if vision_pred_class == true_label:
        # مكافأة التصنيف الصحيح طردياً مع الثقة
        vision_reward = 1.5 * vision_conf
    else:
        # عقوبة تربيعية على الثقة الخاطئة (Overconfidence Penalty) لمنع الغرور في الصور الملتبسة
        vision_reward = -2.0 * (vision_conf ** 2)

    # تقييم دقة كاشف الأعضاء داخل نموذج الصور
    if has_true_exposed_nudity and detected_exposed:
        vision_reward += 1.0  # كشف عري صريح حقيقي بنجاح
    elif (not has_true_exposed_nudity) and detected_exposed:
        vision_reward -= 1.5  # توهم وجود عري غير موجود في صورة عادية أو مستورة

    return {
        "agent_reward": round(agent_reward, 3),
        "vision_reward": round(vision_reward, 3),
        "logic_shaping": round(logic_shaping, 3),
        "agent_correct": bool(agent_action == true_label),
        "vision_correct": bool(vision_pred_class == true_label)
    }


class DecisionAgent:
    """
    Autonomous Decision Agent for NSFW & Content Moderation.
    Integrates:
      - Probabilistic Vision Outputs (Softmax details)
      - Spatial Anatomy Evidence (Exposed vs Covered vs Human presence)
      - Domain Rules & Contradiction Resolution
      - Contextual Bandits / Reinforcement Policy for adaptive tuning
    """
    def __init__(self, policy_path: Optional[str] = None):
        self.policy_path = policy_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_policy.json")
        self.classes = CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        
        # Adaptive thresholds and bias vector (tunable via RL reward)
        self.weights = {
            "porn_demote_to_sexy_max_conf": 0.75,
            "porn_demote_to_sexy_min_conf": 0.50,
            "exposed_threshold": 0.40,
            "covered_threshold": 0.35,
            "sexy_elevation_threshold": 0.28,
            "class_biases": [0.0, 0.0, 0.0, 0.0, 0.0],  # [drawings, hentai, neutral, porn, sexy]
            "temperature": 1.0
        }
        self.load_policy()

    def load_policy(self) -> None:
        """Loads saved policy configuration if available."""
        if os.path.exists(self.policy_path):
            try:
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.weights.update(data)
            except Exception as e:
                print(f"[DecisionAgent] Notice: Using default policy parameters ({e})")

    def save_policy(self) -> None:
        """Persists the updated agent policy parameters to disk."""
        try:
            with open(self.policy_path, "w", encoding="utf-8") as f:
                json.dump(self.weights, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[DecisionAgent] Error saving policy: {e}")

    def extract_state_features(self, json_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts interpretable logical features from the vision model output."""
        details = json_output.get("details", {})
        detections = json_output.get("anatomy_detections", [])
        
        exposed_hits = [
            d for d in detections 
            if d.get("class", "") in EXPLICIT_EXPOSED_CLASSES and float(d.get("score", 0)) >= self.weights["exposed_threshold"]
        ]
        covered_hits = [
            d for d in detections 
            if ("COVERED" in d.get("class", "") or d.get("class", "") in ("FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED"))
            and float(d.get("score", 0)) >= self.weights["covered_threshold"]
        ]
        face_hits = [
            d for d in detections 
            if "FACE" in d.get("class", "") and float(d.get("score", 0)) >= 0.35
        ]
        
        buttocks_exposed_hits = [
            d for d in detections 
            if d.get("class", "") == "BUTTOCKS_EXPOSED" and float(d.get("score", 0)) >= 0.35
        ]
        
        has_human = bool(face_hits or exposed_hits or covered_hits or buttocks_exposed_hits)
        has_exposed = bool(exposed_hits)
        has_buttocks_exposed = bool(buttocks_exposed_hits)
        has_only_covered = bool(covered_hits) and not has_exposed and not has_buttocks_exposed
        
        return {
            "details": details,
            "vision_cat": json_output.get("category", "neutral"),
            "vision_conf": float(json_output.get("confidence", 0.0)),
            "exposed_hits": exposed_hits,
            "covered_hits": covered_hits,
            "buttocks_exposed_hits": buttocks_exposed_hits,
            "has_human": has_human,
            "has_exposed": has_exposed,
            "has_buttocks_exposed": has_buttocks_exposed,
            "has_only_covered": has_only_covered,
            "max_exposed_score": max([float(d["score"]) for d in exposed_hits], default=0.0),
            "max_buttocks_score": max([float(d["score"]) for d in buttocks_exposed_hits], default=0.0)
        }

    def decide(self, json_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes deductive reasoning and contextual policy to make the optimal action decision.
        """
        state = self.extract_state_features(json_output)
        details = state["details"]
        vis_cat = state["vision_cat"]
        vis_conf = state["vision_conf"]
        
        reasoning_steps = []
        action = vis_cat
        applied_rules = []

        # Rule 1: Contradiction Resolution (Porn with Covered anatomy and no Exposed anatomy -> Sexy)
        # كما نص المستخدم: إذا كانت نسبة porn بين 0.50 و 0.75، و anatomy_detections يحتوي على COVERED فقط (أو يخلو من EXPOSED)، يُخفض التصنيف إلى sexy
        porn_p = details.get("porn", 0.0)
        
        if (self.weights["porn_demote_to_sexy_min_conf"] <= porn_p <= self.weights["porn_demote_to_sexy_max_conf"]) and state["has_only_covered"]:
            action = "sexy"
            applied_rules.append("CONTRADICTION_RESOLVED_COVERED_ANATOMY")
            reasoning_steps.append(
                f"Resolved contradiction: Vision predicted 'porn' ({porn_p:.2f}) but detected only COVERED anatomy (no EXPOSED). Demoted to 'sexy'."
            )

        # Rule 2: Explicit Exposed Anatomy Safety Gate (Definite NSFW Guard)
        # إذا رصد كاشف الأعضاء عضواً مكشوفاً صريحاً بنسبة ثقة >= 0.40، يمنع منعاً باتاً تصنيف الصورة كآمنة
        elif state["has_exposed"]:
            applied_rules.append("EXPLICIT_EXPOSED_ANATOMY_ENFORCED")
            # Determine domain (Anime/Drawing vs Real/3D)
            is_drawn = (details.get("hentai", 0.0) + details.get("drawings", 0.0)) > (details.get("porn", 0.0) + details.get("neutral", 0.0))
            if is_drawn:
                action = "hentai"
                reasoning_steps.append(f"Exposed intimate anatomy detected ({state['max_exposed_score']:.2f}) with drawn style -> Action: hentai.")
            else:
                action = "porn"
                reasoning_steps.append(f"Exposed intimate anatomy detected ({state['max_exposed_score']:.2f}) with realistic style -> Action: porn.")

        # Rule 2.5: Suggestive Buttocks Exposure Gate (Bikini / Swimwear / Thongs / Beach / Posing)
        # رصد مؤخرة مكشوفة BUTTOCKS_EXPOSED بنسبة ثقة >= 0.35 يمنع منعاً باتاً تصنيف الصورة كـ neutral أو آمنة
        elif state["has_buttocks_exposed"]:
            applied_rules.append("SUGGESTIVE_BUTTOCKS_EXPOSURE_ENFORCED")
            action = "sexy"
            reasoning_steps.append(
                f"Suggestive intimate anatomy detected: BUTTOCKS_EXPOSED ({state['max_buttocks_score']:.2f}) -> Enforced action: 'sexy' (NSFW/Suggestive)."
            )

        # Rule 2.8: False Suggestive Disambiguation Guard (Crowded Casts, Art, Non-Intimate Color Clutter)
        # إذا كان تخمين الرؤية 'sexy' بجرعة ثقة ضعيفة (< 0.55)، مع انعدام تام لأي عري أو أجزاء سباحة/حساسة مغطاة:
        # فهذا وهم بصري ناتج عن تدرجات لونية (مثل وجوه شخصيات فاميلي غاي الكرتونية).
        # يُخفض التصنيف فوراً إلى drawings أو neutral حسب الأرجحية البصرية الآمنة
        elif vis_cat == "sexy" and vis_conf < 0.55 and not state["has_exposed"] and not state["has_buttocks_exposed"] and not state["has_only_covered"]:
            if details.get("drawings", 0.0) >= details.get("neutral", 0.0):
                action = "drawings"
                applied_rules.append("FALSE_SUGGESTIVE_DEMOTED_TO_DRAWING")
                reasoning_steps.append(
                    f"False suggestive pattern resolved: Vision predicted weak 'sexy' ({vis_conf:.2f}) without any exposed or covered intimate anatomy. Demoted to safe 'drawings' ({details.get('drawings', 0.0):.2f})."
                )
            else:
                action = "neutral"
                applied_rules.append("FALSE_SUGGESTIVE_DEMOTED_TO_NEUTRAL")
                reasoning_steps.append(
                    f"False suggestive pattern resolved: Vision predicted weak 'sexy' ({vis_conf:.2f}) without any exposed or covered intimate anatomy. Demoted to safe 'neutral' ({details.get('neutral', 0.0):.2f})."
                )

        # Rule 3: Calibrated Bayesian Decision with Policy Bias
        else:
            # When vision is confident and no contradiction exists, trust the calibrated model
            scores = [details.get(c, 0.0) + self.weights["class_biases"][i] for i, c in enumerate(self.classes)]
            best_idx = int(np.argmax(scores))
            candidate_action = self.classes[best_idx]
            
            # Safe Class Protection Rule:
            # Policy bias must NEVER shift a safe prediction ('drawings' or 'neutral') into NSFW ('porn' or 'hentai')
            # unless explicit exposed intimate anatomy was confirmed by the anatomy detector.
            if vis_cat in SAFE_CLASSES and candidate_action in NSFW_CLASSES and not state["has_exposed"]:
                action = vis_cat
                applied_rules.append("SAFE_GRAPHIC_SHIELD_ENFORCED")
                reasoning_steps.append(f"Protected safe class '{vis_cat}' from policy bias shift to '{candidate_action}' (no exposed intimate anatomy detected).")
            elif vis_conf >= 0.65:
                action = vis_cat
                applied_rules.append("TRUST_HIGH_CONF_VISION")
                reasoning_steps.append(f"High-confidence vision prediction verified ({vis_cat}: {vis_conf:.2f}).")
            else:
                action = candidate_action
                if action != vis_cat:
                    applied_rules.append("POLICY_BIAS_CORRECTION")
                    reasoning_steps.append(f"Policy bias shifted low-confidence prediction from {vis_cat} to {action}.")
                else:
                    applied_rules.append("TRUST_VISION_PREDICTION")
                    reasoning_steps.append(f"Standard verification passed: Trusting vision prediction '{action}'.")

        is_nsfw = bool(action in NSFW_CLASSES)
        agent_conf = round(float(details.get(action, vis_conf)), 4)

        return {
            "agent_action": action,
            "is_nsfw": is_nsfw,
            "agent_confidence": agent_conf,
            "applied_rules": applied_rules,
            "reasoning": " | ".join(reasoning_steps),
            "vision_category": vis_cat,
            "vision_confidence": vis_conf
        }

    def update_policy(self, state_features: Dict[str, Any], action: str, reward: float, learning_rate: float = 0.01) -> None:
        """
        Policy Gradient / Contextual Bandit update based on received reward.
        Increases bias for high-reward actions, decreases for penalized actions.
        """
        act_idx = self.class_to_idx.get(action, None)
        if act_idx is not None:
            # Gradient ascent on class bias: step in direction of reward
            self.weights["class_biases"][act_idx] += learning_rate * np.clip(reward, -2.0, 2.0)
            # Normalization and conservative clipping to prevent runaway false positive drift
            mean_bias = float(np.mean(self.weights["class_biases"]))
            self.weights["class_biases"] = [
                float(np.clip(round(b - mean_bias, 5), -0.03, 0.03)) 
                for b in self.weights["class_biases"]
            ]
