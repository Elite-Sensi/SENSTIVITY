"""
AI Sensitivity Matcher - All bugs fixed:
1. No decimals - int(round()) everywhere
2. Free look correct keys: tpp_character_vehicle / camera_parachuting / fpp_character
3. Gyro zero fix - returns None when gyro off
"""

import os
import pickle
import numpy as np
from typing import Optional
from models.dataset import SENSITIVITY_DATASET, DEVICE_TIERS

MODEL = None
MODEL_PATH = "models/saved/sensitivity_model.pkl"

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                MODEL = pickle.load(f)
            print("Neural Network model loaded OK")
        except Exception as e:
            print(f"Could not load model: {e} - using averaging")
            MODEL = None
    else:
        print("No trained model - using smart averaging")

load_model()

DEVICE_FEATURES = {
    "iphone 15 pro max":        [1.00, 1.0, 0.8, 1.0],
    "iphone 15 pro":            [1.00, 0.9, 0.8, 1.0],
    "iphone 14 pro max":        [0.95, 1.0, 0.8, 1.0],
    "iphone 14 pro":            [0.95, 0.9, 0.8, 1.0],
    "iphone 13 pro max":        [0.90, 1.0, 0.8, 1.0],
    "iphone 13 pro":            [0.90, 0.9, 0.8, 1.0],
    "iphone 12 pro max":        [0.85, 1.0, 0.8, 1.0],
    "iphone 12 pro":            [0.85, 0.9, 0.8, 1.0],
    "iphone 11":                [0.70, 0.85, 0.6, 1.0],
    "samsung galaxy s24 ultra": [1.00, 1.0, 1.0, 0.9],
    "samsung galaxy s23 ultra": [0.95, 1.0, 1.0, 0.9],
    "samsung galaxy s22 ultra": [0.90, 1.0, 1.0, 0.9],
    "samsung galaxy s22":       [0.80, 0.9, 1.0, 0.9],
    "rog phone 7":              [1.00, 1.0, 1.0, 0.85],
    "rog phone 6":              [0.95, 1.0, 1.0, 0.85],
    "rog phone 5":              [0.90, 1.0, 1.0, 0.85],
    "oneplus 12":               [0.90, 0.95, 1.0, 0.85],
    "oneplus 11":               [0.85, 0.95, 1.0, 0.85],
    "poco f5 pro":              [0.70, 0.9, 1.0, 0.75],
    "poco f5":                  [0.65, 0.85, 1.0, 0.75],
    "redmi note 12 pro":        [0.55, 0.8, 0.8, 0.75],
    "samsung galaxy a54":       [0.45, 0.8, 0.6, 0.90],
    "redmi 12":                 [0.35, 0.75, 0.5, 0.75],
    "tecno spark 20":           [0.25, 0.70, 0.4, 0.50],
}

SCOPE_KEYS = ["tpp","fpp","red_dot","scope_2x","scope_3x","scope_4x","scope_6x","scope_8x"]
FL_KEYS    = ["tpp_character_vehicle","camera_parachuting","fpp_character"]

def get_device_features(device_name):
    key = device_name.lower().strip()
    if key in DEVICE_FEATURES:
        return DEVICE_FEATURES[key]
    for k, v in DEVICE_FEATURES.items():
        if k in key or key in k:
            return v
    for tier, devices in DEVICE_TIERS.items():
        for d in devices:
            if d in key or key in d:
                if tier == "high": return [0.85, 0.9, 0.9, 0.85]
                if tier == "mid":  return [0.60, 0.8, 0.7, 0.75]
                if tier == "low":  return [0.35, 0.7, 0.5, 0.60]
    return [0.60, 0.8, 0.7, 0.75]

def encode_input(device, fingers, gyro, skill="intermediate"):
    dev_f        = get_device_features(device)
    fingers_norm = (int(fingers) - 2) / 4.0
    gyro_flag    = 1.0 if gyro else 0.0
    if skill == "beginner":  skill_vec = [1.0, 0.0, 0.0]
    elif skill == "pro":     skill_vec = [0.0, 0.0, 1.0]
    else:                    skill_vec = [0.0, 1.0, 0.0]
    return np.array(dev_f + [fingers_norm, gyro_flag] + skill_vec, dtype=np.float32)

def predict_with_nn(device, fingers, gyro, skill):
    if MODEL is None:
        return None
    try:
        X = encode_input(device, fingers, gyro, skill).reshape(1, -1)
        models = MODEL["models"]
        def get_section(name, keys, gyro_cap=False):
            net   = models[name]["net"]
            scale = models[name]["scale"]
            pred  = net.predict(X)[0] * scale
            pred  = np.clip(pred, 1, scale)
            result = {k: int(round(float(v))) for k, v in zip(keys, pred)}
            if gyro_cap:
                result = {k: min(v, 400) for k, v in result.items()}
            return result
        return {
            "camera":        get_section("camera",        SCOPE_KEYS),
            "ads":           get_section("ads",           SCOPE_KEYS),
            "free_look":     get_section("free_look",     FL_KEYS),
            "gyroscope":     get_section("gyroscope",     SCOPE_KEYS, gyro_cap=True) if gyro else None,
            "gyroscope_ads": get_section("gyroscope_ads", SCOPE_KEYS, gyro_cap=True) if gyro else None,
        }
    except Exception as e:
        print(f"NN prediction failed: {e}")
        return None

def normalize_device(device):
    return device.lower().strip()

def get_device_tier(d):
    for tier, devices in DEVICE_TIERS.items():
        if d in devices: return tier
    for tier, devices in DEVICE_TIERS.items():
        for dev in devices:
            if dev in d or d in dev: return tier
    return None

def average_sensitivity(entries, key):
    valid = [e[key] for e in entries if e.get(key) is not None]
    if not valid:
        return None
    keys = valid[0].keys()
    return {k: int(round(sum(v[k] for v in valid) / len(valid))) for k in keys}

GYRO_KEYS = {"gyroscope", "gyroscope_ads"}

def apply_skill(d, skill, is_gyro=False):
    if not d:
        return None
    m = 0.85 if skill == "beginner" else (1.10 if skill == "pro" else 1.0)
    result = {k: int(round(v * m)) for k, v in d.items()}
    if is_gyro:
        result = {k: min(v, 400) for k, v in result.items()}
    return result

DEFAULT_CAM  = {"tpp":100,"fpp":90,"red_dot":55,"scope_2x":42,"scope_3x":33,"scope_4x":26,"scope_6x":19,"scope_8x":14}
DEFAULT_ADS  = {"tpp":52,"fpp":57,"red_dot":52,"scope_2x":43,"scope_3x":36,"scope_4x":29,"scope_6x":21,"scope_8x":16}
DEFAULT_FL   = {"tpp_character_vehicle":78,"camera_parachuting":95,"fpp_character":72}
DEFAULT_GYRO = {"tpp":300,"fpp":300,"red_dot":200,"scope_2x":180,"scope_3x":160,"scope_4x":140,"scope_6x":110,"scope_8x":90}
DEFAULT_GYRA = {"tpp":280,"fpp":280,"red_dot":190,"scope_2x":170,"scope_3x":150,"scope_4x":130,"scope_6x":100,"scope_8x":80}

def generate_sensitivity(device: str, fingers: str, gyro: bool, skill_level: Optional[str] = None) -> dict:
    skill = skill_level or "intermediate"

    nn = predict_with_nn(device, fingers, gyro, skill)
    if nn:
        return {
            "device_matched": device, "match_type": "neural-network",
            "player_name": None,
            "camera": nn["camera"], "ads": nn["ads"], "free_look": nn["free_look"],
            "gyroscope": nn["gyroscope"], "gyroscope_ads": nn["gyroscope_ads"],
            "confidence_score": 0.95,
            "players_used": [p["player_name"] for p in SENSITIVITY_DATASET],
        }

    device_norm = normalize_device(device)
    device_tier = get_device_tier(device_norm)

    matches = [p for p in SENSITIVITY_DATASET
               if normalize_device(p["device"]) == device_norm
               and p["finger_layout"] == fingers and p["gyro"] == gyro]
    if matches:
        match_type, confidence, matched_device = "exact", 1.0, device
    else:
        matches = [p for p in SENSITIVITY_DATASET
                   if p["finger_layout"] == fingers and p["gyro"] == gyro]
        if matches:
            match_type, confidence, matched_device = "partial", 0.75, f"Similar ({fingers}-finger)"
        else:
            matches = [p for p in SENSITIVITY_DATASET if p["finger_layout"] == fingers]
            if matches:
                match_type, confidence, matched_device = "finger-match", 0.65, f"{fingers}-finger players"
            else:
                matches = [p for p in SENSITIVITY_DATASET if p.get("device_tier") == device_tier] or SENSITIVITY_DATASET
                match_type, confidence, matched_device = "tier-based", 0.50, f"{device_tier or 'general'} tier"

    gyro_matches  = [p for p in matches if p.get("gyroscope")]
    gyra_matches  = [p for p in matches if p.get("gyroscope_ads")]

    return {
        "device_matched": matched_device, "match_type": match_type, "player_name": None,
        "camera":        apply_skill(average_sensitivity(matches, "camera"),     skill) or DEFAULT_CAM,
        "ads":           apply_skill(average_sensitivity(matches, "ads"),        skill) or DEFAULT_ADS,
        "free_look":     apply_skill(average_sensitivity(matches, "free_look"),  skill) or DEFAULT_FL,
        "gyroscope":     apply_skill(average_sensitivity(gyro_matches, "gyroscope"),    skill, is_gyro=True) if gyro else None,
        "gyroscope_ads": apply_skill(average_sensitivity(gyra_matches, "gyroscope_ads"),skill, is_gyro=True) if gyro else None,
        "confidence_score": confidence,
        "players_used": [p["player_name"] for p in matches],
    }
