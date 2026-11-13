from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from . import __version__
from .accounts import AccountPool
from .catalog import MODEL_CATALOG, normalize_model, resolve_upstream_key
from .config import Settings
from .desktop import DesktopScanError, scan_desktop_account
from .errors import AuthenticationError, GatewayError
from .logs import RequestLog
from .models import ChatCompletionRequest, ModelCard, ModelList
from .security import SecurityStore
from .service import ChatService
from .protocol import validate_request
from .transport import DirectTransport

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"
SESSION_COOKIE = "doubao_admin_session"


class LoginRequest(BaseModel):
    password: str = Field(max_length=1024)


class PasswordRequest(BaseModel):
    current_password: str = Field(max_length=1024)
    new_password: str = Field(min_length=8, max_length=1024)


class AccountImportRequest(BaseModel):
    account: dict[str, Any] | list[dict[str, Any]] | None = None
    accounts: list[dict[str, Any]] | None = None


class AccountPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    name: str | None = Field(default=None, max_length=128)
    remark: str | None = Field(default=None, max_length=512)
    group: str | None = Field(default=None, max_length=64)


class AccountSelection(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=10000)


class BatchRequest(AccountSelection):
    action: Literal["enable", "disable", "delete", "clear-cooldown", "group"]
    group: str = Field(default="", max_length=64)


def create_app(settings: Settings | None = None) -> FastAPI:
    initial = settings or Settings.from_env()
    security = SecurityStore(initial)
    runtime = security.runtime
    settings = replace(
        initial, default_model=runtime["default_model"],
        request_timeout_seconds=runtime["timeout_seconds"],
        cooldown_seconds=runtime["cooldown_seconds"],
    )
    pool = AccountPool(settings.accounts_dir, cooldown_seconds=settings.cooldown_seconds)
    pool.load()
    transport = DirectTransport(settings)
    request_log = RequestLog()
    service = ChatService(
        pool, transport, model_key=settings.model_key, app_id=settings.app_id,
        workspace_root=settings.data_dir / "workspaces",
        default_model=settings.default_model, request_log=request_log,
    )
    key_active: dict[str, int] = {}
    key_lock = asyncio.Lock()
    config_lock = asyncio.Lock()
    login_failures: dict[str, list[float]] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await transport.close()

    app = FastAPI(title="doubaowork2api", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.security = security
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
        if request.url.path.startswith("/admin/"):
            response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/" or request.url.path.startswith("/ui-assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

    def check_origin(request: Request) -> None:
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return
        origin = request.headers.get("origin")
        if origin:
            expected = urlsplit(str(request.url))
            actual = urlsplit(origin)
            if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
                raise HTTPException(403, "不允许跨站管理请求。")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "不允许跨站管理请求。")

    async def require_admin(request: Request) -> None:
        check_origin(request)
        if not security.session_valid(request.cookies.get(SESSION_COOKIE, "")):
            raise HTTPException(401, "请使用管理密码登录控制台。")

    async def require_api(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        supplied = x_api_key or ""
        if not supplied and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        principal = security.authenticate_api(supplied)
        if principal is None:
            raise AuthenticationError("Invalid API key.")
        return principal

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError):
        return JSONResponse(status_code=exc.status_code, content=exc.as_openai())

    @app.exception_handler(ValueError)
    async def invalid_value_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def missing_handler(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "指定对象不存在。"})

    @app.get("/health")
    async def health():
        return {
            "ok": True, "service": "doubaowork2api", "version": __version__,
            "accounts": len(pool.accounts),
            "available_accounts": sum(account.available for account in pool.accounts),
            "time": int(time.time()),
        }

    @app.get("/v1/models")
    async def models(key: dict = Depends(require_api)) -> ModelList:
        allowed = {normalize_model(model) for model in key["models"]}
        return ModelList(data=[
            ModelCard(
                id=item.id, owned_by="doubao-work",
                context_window=item.context_window,
                reasoning_efforts=list(item.reasoning_levels),
                default_reasoning_effort=item.default_reasoning_effort,
            )
            for item in MODEL_CATALOG if not allowed or item.id in allowed
        ])

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, key: dict = Depends(require_api)):
        if "model" not in payload.model_fields_set:
            payload = payload.model_copy(update={"model": app.state.settings.default_model})
        resolve_upstream_key(payload.normalized_model)
        allowed_models = {normalize_model(model) for model in key["models"]}
        if allowed_models and payload.normalized_model not in allowed_models:
            raise HTTPException(403, "此 API Key 无权调用该模型。")
        validate_request(payload)
        async with key_lock:
            active = key_active.get(key["id"], 0)
            if key["concurrency_limit"] and active >= key["concurrency_limit"]:
                return JSONResponse(status_code=429, content={"error": {
                    "type": "rate_limit_error", "code": "key_concurrency_limit",
                    "message": "此 API Key 已达到并发上限。",
                }})
            key_active[key["id"]] = active + 1
        released = False

        async def release():
            nonlocal released
            async with key_lock:
                if not released:
                    remaining = key_active.get(key["id"], 1) - 1
                    if remaining:
                        key_active[key["id"]] = remaining
                    else:
                        key_active.pop(key["id"], None)
                    released = True

        options = {
            "allowed_ids": set(key["accounts"]) if key["accounts"] else None,
            "key_id": key["id"],
        }
        if payload.stream:
            async def chunks():
                try:
                    async for chunk in service.stream(payload, **options):
                        yield chunk
                except GatewayError as exc:
                    yield "data: " + json.dumps(exc.as_openai(), ensure_ascii=False) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await release()
            return StreamingResponse(
                chunks(), media_type="text/event-stream", background=BackgroundTask(release),
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await service.complete(payload, **options)
        finally:
            await release()

    @app.post("/admin/login")
    async def admin_login(payload: LoginRequest, request: Request, response: Response):
        check_origin(request)
        address = request.client.host if request.client else "unknown"
        now = time.monotonic()
        failures = [stamp for stamp in login_failures.get(address, []) if now - stamp < 60]
        if len(failures) >= 10:
            raise HTTPException(429, "登录失败次数过多，请稍后再试。")
        if not await asyncio.to_thread(security.verify_admin, payload.password):
            if len(login_failures) >= 1024:
                login_failures.clear()
            login_failures[address] = failures + [now]
            raise HTTPException(401, "管理密码不正确。")
        login_failures.pop(address, None)
        response.set_cookie(
            SESSION_COOKIE, security.create_session(), httponly=True, samesite="strict",
            secure=request.url.scheme == "https", max_age=43200, path="/admin",
        )
        response.headers["Cache-Control"] = "no-store"
        return {"ok": True}

    @app.post("/admin/logout", dependencies=[Depends(require_admin)])
    async def admin_logout(request: Request, response: Response):
        security.delete_session(request.cookies.get(SESSION_COOKIE, ""))
        response.delete_cookie(SESSION_COOKIE, path="/admin")
        return {"ok": True}

    @app.post("/admin/password", dependencies=[Depends(require_admin)])
    async def admin_password(payload: PasswordRequest, response: Response):
        await asyncio.to_thread(security.set_admin_password, payload.current_password, payload.new_password)
        response.delete_cookie(SESSION_COOKIE, path="/admin")
        return {"ok": True}

    def config_document():
        current = app.state.settings
        return {
            "version": __version__, "models": [item.public_dict() for item in MODEL_CATALOG],
            "default_model": current.default_model,
            "api_key_configured": bool(security.list_keys()),
            "accounts_dir": str(current.accounts_dir), "upstream": current.upstream_base,
            "timeout_seconds": current.request_timeout_seconds,
            "cooldown_seconds": current.cooldown_seconds,
            "listen": f"{current.host}:{current.port}", "auth": "admin_cookie",
            "runtime": security.runtime,
        }

    @app.get("/admin/config", dependencies=[Depends(require_admin)])
    async def admin_config():
        return config_document()

    @app.put("/admin/config", dependencies=[Depends(require_admin)])
    async def save_config(payload: dict[str, Any]):
        if "default_model" in payload:
            if not isinstance(payload["default_model"], str):
                raise ValueError("默认模型必须是字符串。")
            resolve_upstream_key(payload["default_model"])
            payload["default_model"] = normalize_model(payload["default_model"])
        async with key_lock, config_lock:
            # Active streams finish with their current transport configuration.
            if key_active or any(account.in_flight for account in pool.accounts):
                raise HTTPException(409, "有请求正在执行，请在请求结束后保存设置。")
            runtime = security.update_runtime(payload)
            current = replace(app.state.settings, default_model=runtime["default_model"],
                              request_timeout_seconds=runtime["timeout_seconds"],
                              cooldown_seconds=runtime["cooldown_seconds"])
            await transport.configure(current)
            pool.cooldown_seconds = current.cooldown_seconds
            service.default_model = current.default_model
            app.state.settings = current
        return config_document()

    def validate_key_refs(payload):
        if "models" in payload:
            if not isinstance(payload["models"], list):
                raise ValueError("models 必须是列表。")
            for model in payload["models"]:
                if not isinstance(model, str):
                    raise ValueError("模型列表只能包含模型 ID 字符串。")
                resolve_upstream_key(model)
            payload["models"] = list(dict.fromkeys(normalize_model(model) for model in payload["models"]))
        if "accounts" in payload:
            if not isinstance(payload["accounts"], list) or any(
                not isinstance(account_id, str) or pool.find(account_id) is None
                for account_id in payload["accounts"]
            ):
                raise ValueError("accounts 必须是已有账号 ID 的列表。")

    @app.get("/admin/keys", dependencies=[Depends(require_admin)])
    async def keys():
        return {"data": security.list_keys()}

    @app.post("/admin/keys", dependencies=[Depends(require_admin)])
    async def add_key(payload: dict[str, Any]):
        validate_key_refs(payload)
        return security.create_key(payload)

    @app.patch("/admin/keys/{key_id}", dependencies=[Depends(require_admin)])
    async def patch_key(key_id: str, payload: dict[str, Any]):
        validate_key_refs(payload)
        return security.update_key(key_id, payload)

    @app.delete("/admin/keys/{key_id}", dependencies=[Depends(require_admin)])
    async def delete_key(key_id: str):
        security.delete_key(key_id)
        return {"ok": True}

    @app.get("/admin/accounts", dependencies=[Depends(require_admin)])
    async def admin_accounts():
        return {"data": pool.snapshot()}

    @app.get("/admin/logs", dependencies=[Depends(require_admin)])
    async def admin_logs(limit: int = 100):
        return {"data": request_log.snapshot(limit)}

    @app.post("/admin/accounts/scan-offline", dependencies=[Depends(require_admin)])
    async def scan_offline():
        try:
            result = await scan_desktop_account(pool)
        except DesktopScanError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"created": result.created, "cookie_count": result.cookie_count,
                "source": result.source, "account": result.account.public_dict()}

    @app.post("/admin/accounts/import", dependencies=[Depends(require_admin)])
    async def import_accounts(payload: AccountImportRequest):
        documents = payload.accounts if payload.accounts is not None else (
            payload.account if isinstance(payload.account, list)
            else [payload.account] if payload.account else []
        )
        if not documents:
            raise HTTPException(400, "没有提供账号。")
        imported, errors = [], []
        created = updated = 0
        for index, document in enumerate(documents, 1):
            before = {account.id for account in pool.accounts}
            try:
                account = await pool.upsert(document)
                imported.append(account.id)
                if account.id in before:
                    updated += 1
                else:
                    created += 1
            except (ValueError, OSError) as exc:
                errors.append(f"account {index}: {exc}")
        return {"imported": imported, "errors": errors, "created": created, "updated": updated}

    @app.post("/admin/accounts/export", dependencies=[Depends(require_admin)])
    async def export_accounts(payload: AccountSelection):
        return JSONResponse(
            content={"accounts": pool.export_accounts(list(dict.fromkeys(payload.ids)))},
            headers={"Cache-Control": "no-store",
                     "Content-Disposition": 'attachment; filename="doubao-accounts.json"'},
        )

    @app.post("/admin/accounts/batch", dependencies=[Depends(require_admin)])
    async def batch_accounts(payload: BatchRequest):
        ids = list(dict.fromkeys(payload.ids))
        if payload.action == "delete":
            deleted = await pool.delete_many(ids)
            await transport.forget_accounts(deleted)
            return {"succeeded": deleted, "errors": []}
        succeeded, errors = [], []
        for account_id in ids:
            try:
                if payload.action in {"enable", "disable"}:
                    await pool.set_enabled(account_id, payload.action == "enable")
                elif payload.action == "group":
                    await pool.update_metadata(account_id, {"group": payload.group})
                else:
                    await pool.clear_cooldown(account_id)
                succeeded.append(account_id)
            except (KeyError, ValueError, OSError):
                errors.append({"id": account_id, "message": "账号不存在、忙碌或无法保存。"})
        return {"succeeded": succeeded, "errors": errors}

    @app.patch("/admin/accounts/{account_id}", dependencies=[Depends(require_admin)])
    async def patch_account(account_id: str, payload: AccountPatch):
        fields = payload.model_dump(exclude_unset=True)
        if not fields or any(value is None for value in fields.values()):
            raise ValueError("请提供有效的账号修改字段。")
        account = await pool.update_metadata(account_id, fields)
        return {"data": account.public_dict()}

    @app.post("/admin/accounts/{account_id}/clear-cooldown", dependencies=[Depends(require_admin)])
    async def clear_account_cooldown(account_id: str):
        return {"data": (await pool.clear_cooldown(account_id)).public_dict()}

    @app.get("/", include_in_schema=False)
    async def web_ui():
        return FileResponse(WEB_DIR / "index.html")

    return app
