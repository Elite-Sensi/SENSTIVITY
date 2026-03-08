from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class FingerLayout(str, Enum):
    two   = "2"
    three = "3"
    four  = "4"
    five  = "5"
    six   = "6"


class SensitivityRequest(BaseModel):
    device:      str           = Field(..., example="iPhone 14 Pro Max")
    fingers:     FingerLayout  = Field(..., example="4")
    gyro:        bool          = Field(..., example=True)
    skill_level: Optional[str] = Field(None, example="intermediate")


class CameraSensitivity(BaseModel):
    tpp:      int
    fpp:      int
    red_dot:  int
    scope_2x: int
    scope_3x: int
    scope_4x: int
    scope_6x: int
    scope_8x: int


class ADSSensitivity(BaseModel):
    tpp:      int
    fpp:      int
    red_dot:  int
    scope_2x: int
    scope_3x: int
    scope_4x: int
    scope_6x: int
    scope_8x: int


class FreeLookSensitivity(BaseModel):
    tpp_character_vehicle: int
    camera_parachuting:    int
    fpp_character:         int


class GyroscopeSensitivity(BaseModel):
    tpp:      int
    fpp:      int
    red_dot:  int
    scope_2x: int
    scope_3x: int
    scope_4x: int
    scope_6x: int
    scope_8x: int


class SensitivityResponse(BaseModel):
    device_matched:   str
    match_type:       str
    player_name:      Optional[str]
    camera:           CameraSensitivity
    ads:              ADSSensitivity
    free_look:        FreeLookSensitivity
    gyroscope:        Optional[GyroscopeSensitivity]
    gyroscope_ads:    Optional[GyroscopeSensitivity]
    confidence_score: float
    players_used:     Optional[List[str]] = None
