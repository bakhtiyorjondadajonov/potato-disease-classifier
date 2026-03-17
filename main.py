import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from model_loader import load_model
from services.gemini_service import init_gemini
from routes import health, predict, advice, calendar, severity, analyze

# Logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading model and initializing services")
    load_model(settings.model_path)
    init_gemini(settings.gemini_api_key)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Plant Disease Classifier API",
    version="3.0.0",
    description="CNN-based potato disease classification plus Gemini-powered multi-plant disease analysis (tomato, corn, pepper, apple, strawberry) with treatment advice, crop calendars, and severity analysis.",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(advice.router)
app.include_router(calendar.router)
app.include_router(severity.router)
app.include_router(analyze.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
