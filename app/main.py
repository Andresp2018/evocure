"""
EVOCURE FastAPI Application.

Serves the zero-CDN local HTML5/CSS3/Canvas2D frontend, mounts static assets,
routes REST endpoints under /api/v1, and manages the real-time simulation WebSocket.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router, ws_router
from app.api.evolution import router as evolution_router
from app.services.session_manager import get_session_manager

logger = logging.getLogger("evocure.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager to start and cleanly shut down simulation background loops."""
    logger.info("Initializing EVOCURE Session Manager...")
    sm = get_session_manager()
    await sm.start_background_loop()
    yield
    logger.info("Shutting down EVOCURE Session Manager...")
    await sm.stop_background_loop()


app = FastAPI(
    title="EVOCURE - Chemical Evolution & Disease Simulator",
    description="Local, offline, epistemically grounded chemical evolution and virtual disease simulator.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include Routers
app.include_router(api_router)
app.include_router(evolution_router)
app.include_router(ws_router)


@app.get("/", response_class=FileResponse, summary="EVOCURE Main User Interface")
async def serve_index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # Fallback if not yet written
        return FileResponse(str(STATIC_DIR / "index.html"))
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
