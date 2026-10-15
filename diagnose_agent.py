import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nsfw_classifier import ProductionNSFWClassifier
from decision_agent import DecisionAgent, calculate_rewards
from stream_dataset_pipeline import stream_dataset

clf = ProductionNSFWClassifier()
agent = DecisionAgent()

for path, label, has_exposed in stream_dataset('dataset', max_per_class=10):
    vis = clf.classify(path)
    dec = agent.decide(vis)
    v_cat = vis['category']
    a_act = dec['agent_action']
    if a_act != v_cat or a_act != label:
        rew = calculate_rewards(vis, a_act, label, has_exposed)
        print(f"Path: {path}")
        print(f"  True: {label} | Vis: {v_cat} ({vis['confidence']:.2f}) | Agent: {a_act} (Rew: {rew['agent_reward']:+0.2f}, VisRew: {rew['vision_reward']:+0.2f})")
        print(f"  Rules: {dec['applied_rules']}")
        print(f"  Anatomy: {vis['anatomy_detections']}")
        print("-" * 50)
