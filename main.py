from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from routes.sensitivity import router as sensitivity_router

app = FastAPI(title="PUBG Sensitivity API", version="3.0")

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensitivity_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "PUBG Sensitivity Maker API is running", "version": "3.0"}

@app.get("/health")
def health():
    return {"status": "ok"}
