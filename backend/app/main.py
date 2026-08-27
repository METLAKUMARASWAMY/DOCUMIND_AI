import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth_routes import router as auth_router
from app.api.vision_alias_routes import router as vision_alias_router
from app.api.vision_routes import router as vision_router
from app.core.config import settings
from app.core.exceptions import install_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import install_security_middleware
from app.core.observability import install_observability
from app.services.auth import auth_service
from app.services.vision.persistence import vision_persistence


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(vision_persistence.initialize)
    await asyncio.to_thread(auth_service.initialize_persistence)
    yield
    await asyncio.to_thread(vision_persistence.close)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="DocuMind AI",
        version="0.1.0",
        description="Intelligent Enterprise Document Intelligence Platform",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_security_middleware(app)
    install_observability(app)
    install_exception_handlers(app)

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> JSONResponse:
        health = vision_persistence.health()
        status_code = 200 if health["status"] == "ready" else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": health["status"], "persistence": health},
        )

    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(vision_alias_router, prefix="/api/v1")
    app.include_router(vision_router, prefix="/api/v1")
    if settings.enable_demo_api:
        from app.api.finance_routes import router as finance_router
        from app.api.routes import router as demo_router

        app.include_router(demo_router, prefix="/api/v1/demo-legacy")
        app.include_router(finance_router, prefix="/api/v1/demo-legacy")
    return app


app = create_app()
