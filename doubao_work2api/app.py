from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .accounts import AccountPool
from .config import Settings
from .errors import AuthenticationError, GatewayError
from .models import ChatCompletionRequest, ModelCard, ModelList
from .service import ChatService
from .transport import DirectTransport

logger = logging.getLogger(__name__)

MODEL_IDS = ["deepseek-v4.1-flash"]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pool = AccountPool(
        settings.accounts_dir,
        cooldown_seconds=settings.cooldown_seconds,
    )
    pool.load()
    transport = DirectTransport(settings)
    service = ChatService(
        pool,
        transport,
        model_key=settings.model_key,
        app_id=settings.app_id,
        workspace_root=settings.data_dir / "workspaces",
        default_model=settings.default_model,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.api_key:
            logger.warning("API key authentication is disabled")
        yield
        await transport.close()

    app = FastAPI(
        title="DoubaoWork2API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pool = pool
    app.state.service = service

    async def require_auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not settings.api_key:
            return
        supplied = x_api_key or ""
        if (
            not supplied
            and authorization
            and authorization.lower().startswith("bearer ")
        ):
            supplied = authorization[7:].strip()
        if supplied != settings.api_key:
            raise AuthenticationError("Invalid API key.")

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        _: Request, exc: GatewayError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.as_openai(),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "doubao-work2api",
            "version": __version__,
            "accounts": len(pool.accounts),
            "available_accounts": sum(
                1 for account in pool.accounts if account.available
            ),
            "time": int(time.time()),
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=item) for item in MODEL_IDS])

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_auth)],
    )
    async def chat_completions(payload: ChatCompletionRequest):
        if payload.stream:
            return StreamingResponse(
                service.stream(payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await service.complete(payload)

    @app.get("/admin/accounts", dependencies=[Depends(require_auth)])
    async def admin_accounts() -> dict[str, Any]:
        return {"data": pool.snapshot()}

    return app
