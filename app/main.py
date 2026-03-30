from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import router

app = FastAPI(title="Babblebunch AI")

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTES
# =========================
app.include_router(router)

# =========================
# STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# =========================
# HOME PAGE
# =========================
@app.get("/")
def serve_home():
    return FileResponse("app/static/index.html")