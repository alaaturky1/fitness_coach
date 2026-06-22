from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.rest import router as rest_router
from app.api.ws import router as ws_router
from app.core.logging import configure_logging
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="Real-Time AI Fitness Coaching Backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.limiter = limiter
    app.include_router(rest_router)
    app.include_router(ws_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors()})

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return rate_limit_exceeded_handler(request, exc)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": "internal_server_error"})

    return app


app = create_app()
