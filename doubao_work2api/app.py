from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .accounts import AccountPool
from .catalog import MODEL_CATALOG, resolve_upstream_key
from .config import Settings
from .desktop import DesktopScanError, scan_desktop_account
from .errors import AuthenticationError, GatewayError
from .logs import RequestLog
from .models import ChatCompletionRequest, ModelCard, ModelList
from .service import ChatService
from .transport import DirectTransport

logger = logging.getLogger(__name__)

MODEL_IDS = [item.id for item in MODEL_CATALOG]
WEB_DIR = Path(__file__).parent / "web"


class AccountEnabledRequest(BaseModel):
    enabled: bool


class AccountImportRequest(BaseModel):
    account: dict[str, Any] | list[dict[str, Any]] | None = None
    accounts: list[dict[str, Any]] | None = None


class LoginRequest(BaseModel):
    api_key: str = ""


class DesktopScanRequest(BaseModel):
    cdp_url: str = "http://127.0.0.1:9222"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pool = AccountPool(
        settings.accounts_dir,
        cooldown_seconds=settings.cooldown_seconds,
    )
    pool.load()
    transport = DirectTransport(settings)
    request_log = RequestLog()
    service = ChatService(
        pool,
        transport,
        model_key=settings.model_key,
        app_id=settings.app_id,
        workspace_root=settings.data_dir / "workspaces",
        default_model=settings.default_model,
        request_log=request_log,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.api_key:
            logger.warning("API key authentication is disabled")
        yield
        await transport.close()

    app = FastAPI(
        title="doubaowork2api",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pool = pool
    app.state.service = service
    app.state.request_log = request_log
    app.mount("/ui-assets", StaticFiles(directory=WEB_DIR), name="ui-assets")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path == "/" or request.url.path.startswith("/ui-assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

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
            "service": "doubaowork2api",
            "version": __version__,
            "accounts": len(pool.accounts),
            "available_accounts": sum(
                1 for account in pool.accounts if account.available
            ),
            "time": int(time.time()),
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models() -> ModelList:
        return ModelList(
            data=[
                ModelCard(id=item.id, owned_by="doubao-work")
                for item in MODEL_CATALOG
            ]
        )

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_auth)],
    )
    async def chat_completions(payload: ChatCompletionRequest):
        resolve_upstream_key(payload.normalized_model)
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

    @app.get("/admin/logs", dependencies=[Depends(require_auth)])
    async def admin_logs(limit: int = 100) -> dict[str, Any]:
        return {"data": request_log.snapshot(limit)}

    @app.get("/admin/config", dependencies=[Depends(require_auth)])
    async def admin_config() -> dict[str, Any]:
        return {
            "version": __version__,
            "models": [item.public_dict() for item in MODEL_CATALOG],
            "default_model": settings.default_model,
            "api_key_configured": bool(settings.api_key),
            "accounts_dir": str(settings.accounts_dir),
            "upstream": settings.upstream_base,
            "timeout_seconds": settings.request_timeout_seconds,
            "cooldown_seconds": settings.cooldown_seconds,
            "listen": f"{settings.host}:{settings.port}",
        }

    @app.post(
        "/admin/accounts/scan-desktop",
        dependencies=[Depends(require_auth)],
    )
    async def scan_desktop(
        payload: DesktopScanRequest | None = None,
    ) -> dict[str, Any]:
        try:
            result = await scan_desktop_account(
                pool,
                cdp_url=(payload.cdp_url if payload else "http://127.0.0.1:9222"),
            )
        except DesktopScanError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "created": result.created,
            "cookie_count": result.cookie_count,
            "account": result.account.public_dict(),
        }

    @app.post("/admin/login")
    async def admin_login(payload: LoginRequest) -> dict[str, Any]:
        if settings.api_key and payload.api_key != settings.api_key:
            raise AuthenticationError("Invalid API key.")
        return {"ok": True}

    @app.post(
        "/admin/accounts/import",
        dependencies=[Depends(require_auth)],
    )
    async def import_accounts(payload: AccountImportRequest) -> dict[str, Any]:
        if payload.accounts is not None:
            documents = payload.accounts
        elif isinstance(payload.account, list):
            documents = payload.account
        elif payload.account is not None:
            documents = [payload.account]
        else:
            documents = []
        if not documents:
            raise HTTPException(status_code=400, detail="No accounts supplied.")
        imported: list[str] = []
        errors: list[str] = []
        for index, document in enumerate(documents, start=1):
            try:
                account = await pool.upsert(document)
                imported.append(account.id)
            except Exception as exc:
                logger.exception("Account import failed for entry %s", index)
                errors.append(f"account {index}: {exc}")
        return {"imported": imported, "errors": errors}

    @app.patch(
        "/admin/accounts/{account_id}",
        dependencies=[Depends(require_auth)],
    )
    async def set_account_enabled(
        account_id: str, payload: AccountEnabledRequest
    ) -> dict[str, Any]:
        try:
            account = await pool.set_enabled(account_id, payload.enabled)
        except KeyError:
            raise HTTPException(status_code=404, detail="Account not found.")
        return {"data": account.public_dict()}

    @app.post(
        "/admin/accounts/{account_id}/clear-cooldown",
        dependencies=[Depends(require_auth)],
    )
    async def clear_account_cooldown(account_id: str) -> dict[str, Any]:
        try:
            account = await pool.clear_cooldown(account_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Account not found.")
        return {"data": account.public_dict()}

    @app.get("/", include_in_schema=False)
    async def web_ui() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app
