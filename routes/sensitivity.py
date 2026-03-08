from fastapi import APIRouter, HTTPException
from models.schemas import SensitivityRequest
from services.matcher import generate_sensitivity

router = APIRouter()

@router.post("/generate-sensitivity")
def generate(request: SensitivityRequest):
    try:
        result = generate_sensitivity(
            device=request.device,
            fingers=request.fingers.value,
            gyro=request.gyro,
            skill_level=request.skill_level,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
def list_devices():
    from models.dataset import DEVICE_TIERS
    return DEVICE_TIERS

@router.get("/players")
def list_players():
    from models.dataset import SENSITIVITY_DATASET
    return [{"name": p["player_name"], "team": p["team"], "device": p["device"],
             "finger_layout": p["finger_layout"], "gyro": p["gyro"]} for p in SENSITIVITY_DATASET]
