from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysis import router as analysis_router

app = FastAPI(
    title="AgriSight API",
    description="Pretrained computer-vision pipeline for drone crop imagery.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analysis_router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "operational"}
