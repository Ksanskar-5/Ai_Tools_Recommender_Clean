from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Import routers
from backend.api.auth_routes import router as auth_router
from backend.api.routes import router as recommender_router

app = FastAPI(title="AI Tool Recommender", version="1.0")


# ---- Enable CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve Frontend ----
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
index_file = os.path.join(frontend_dir, "index.html")

app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


# ---- Routers (FIXED PREFIXES) ----
# Authentication should be under /api/auth/*
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(recommender_router, prefix="/api", tags=["Recommendation"])



# ---- Serve index.html ----
@app.get("/", tags=["Frontend"])
def serve_index():
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "error", "message": "index.html not found"}
