from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"
UPSTREAM = "https://www.doubao.com/chat/completion"
MODEL_KEY = "person-deepseek-v4-1-flash-261017"


def build_payload(prompt: str, workspace: Path, device_id: str) -> dict[str, Any]:
    import time
    import uuid

    now = int(time.time())
    local_message_id = str(uuid.uuid4())
    local_conversation_id = f"local_{uuid.uuid4().hex[:16]}"
    workspace_text = str(workspace)
    task_param = {
        "runtime_type": 2,
        "sandbox_auth_type": 1,
        "device_name": "PROBE-PC",
        "folder_name": "",
        "local_app_id": "1044603",
        "local_device_id": device_id,
        "workspace": workspace_text,
    }
    general_task_param = {
        "action": 0,
        "thread_local_message_id": [local_message_id],
        "client_option": {
            "enable_sandbox": True,
            "os": "Windows",
            "shared_folder_path": [workspace_text],
            "workspace": workspace_text,
            "sandbox_auth_type": 1,
        },
        "runtime_type": 2,
        "agent_task_param": task_param,
        "server_do_compare": True,
    }
    return {
        "client_meta": {
            "local_conversation_id": local_conversation_id,
            "conversation_id": "",
            "bot_id": "7338286299411103781",
            "last_section_id": "",
            "last_message_index": None,
        },
        "messages": [
            {
                "local_message_id": local_message_id,
                "content_block": [
                    {
                        "block_type": 10000,
                        "content": {
                            "text_block": {
                                "text": prompt,
                                "icon_url": "",
                                "icon_url_dark": "",
                                "summary": "",
                            },
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "extra_info": {},
                        "append_fields": [],
                    }
                ],
                "message_status": 0,
            }
        ],
        "option": {
            "send_message_scene": "",
            "create_time_ms": now * 1000,
            "collect_id": "",
            "is_audio": False,
            "answer_with_suggest": False,
            "agent_mode": 1,
            "tts_switch": False,
            "need_deep_think": 10001,
            "click_clear_context": False,
            "from_suggest": False,
            "is_regen": False,
            "is_replace": False,
            "is_from_click_option": False,
            "is_from_click_softlink": False,
            "disable_sse_cache": False,
            "select_text_action": "",
            "is_select_text": False,
            "resend_for_regen": False,
            "scene_type": 0,
            "unique_key": str(uuid.uuid4()),
            "start_seq": 0,
            "need_create_conversation": True,
            "conversation_init_option": {
                "project_id": "",
                "need_ack_conversation": True,
            },
            "conversation_init_ext": {
                "agent_task_param": json.dumps(
                    task_param, ensure_ascii=False, separators=(",", ":")
                ),
                "model_item_key": MODEL_KEY,
                "reasoning_effort": "7",
                "mode_id": "3",
            },
            "regen_query_id": [],
            "edit_query_id": [],
            "regen_instruction": "",
            "no_replace_for_regen": False,
            "message_from": 0,
            "shared_app_name": "",
            "shared_app_id": "",
            "sse_recv_event_options": {"support_chunk_delta": True},
            "support_lazy_fetch_stream": True,
            "is_ai_playground": False,
            "is_old_user": True,
            "general_task_param": general_task_param,
            "recovery_option": {
                "is_recovery": False,
                "req_create_time_sec": now,
                "append_sse_event_scene": 0,
            },
            "message_storage_type": 0,
            "related_deleted_message_ids": {},
            "connector_info_list": [],
            "model_config": {
                "model_item_key": MODEL_KEY,
                "model_extra_params": {
                    "memory_profile": "300k_230",
                    "total_window_size": "300000",
                },
                "reasoning_effort": 7,
            },
            "aggregate_params": {
                "mention_skill_list": "[]",
                "mention_plugin_list": "[]",
                "mention_ext": "[{}]",
                "conversation_mode": "",
                "mode_id": "3",
                "model_item_key": MODEL_KEY,
                "agent_mode": "1",
                "reasoning_effort": "7",
                "provider_id": "",
            },
        },
        "user_context": [],
        "ext": {
            "general_task_param": json.dumps(
                general_task_param,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "use_deep_think": "10001",
            "agent_mode": "1",
            "conversation_init_option": json.dumps(
                {"project_id": "", "need_ack_conversation": True},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "sub_conv_firstmet_type": "1",
            "collection_id": "",
            "is_finish": "1",
            "commerce_credit_config_enable": "0",
        },
    }


async def capture(
    prompt: str, workspace: Path
) -> tuple[str, dict[str, str], dict[str, str], dict[str, Any]]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(
            CDP_URL, timeout=120_000
        )
        context = browser.contexts[0]
        page = next(
            item for item in context.pages if "doubaowork-chat" in item.url
        )
        identity = await page.evaluate(
            """() => {
                const tea = JSON.parse(
                    localStorage.getItem('__tea_cache_tokens_1044603') || '{}'
                );
                return {
                    device_id: tea.user_unique_id || '',
                    web_id: tea.web_id || '',
                };
            }"""
        )
        cookies = {
            item["name"]: item["value"]
            for item in await context.cookies("https://www.doubao.com")
        }
        captured: asyncio.Future = asyncio.get_running_loop().create_future()

        async def on_request(request) -> None:
            if "chat/completion" not in request.url:
                return
            print("REQUEST", request.method, request.url[:500])
            if captured.done():
                return
            final_url = request.url.replace(
                "chrome://doubaowork-chat", "https://www.doubao.com", 1
            )
            captured.set_result(
                (
                    final_url,
                    await request.all_headers(),
                    cookies,
                    identity,
                )
            )

        page.on("request", on_request)
        payload = build_payload(
            prompt, workspace, str(identity.get("device_id") or "")
        )
        await page.evaluate(
            """async ([body]) => {
                const tea = JSON.parse(
                    localStorage.getItem('__tea_cache_tokens_1044603') || '{}'
                );
                const sam = JSON.parse(
                    localStorage.getItem('samantha_web_web_id') || '{}'
                );
                const params = {
                    aid: '1044603',
                    real_aid: '1044603',
                    device_id: tea.user_unique_id || '',
                    tea_uuid: tea.user_unique_id || '',
                    web_id: tea.web_id || '',
                    device_platform: 'web',
                    doubao_device_platform: 'desktop',
                    language: 'zh',
                    region: 'CN',
                    sys_region: 'CN',
                    pkg_type: 'release_version',
                    version_code: '20800',
                    pc_version: '2.36.11',
                    doubao_pc_version: '2.36.11',
                    runtime: 'web',
                    runtime_version: '3.45.3',
                    samantha_web: '1',
                    web_platform: 'desktop',
                    'use-olympus-account': '1',
                    client_platform: 'pc_client',
                    chromium_version: '147.0.7727.149',
                    channel: 'win',
                    fp: 'verify_' + (tea.user_unique_id || ''),
                    web_tab_id: crypto.randomUUID(),
                    tz_name: 'Asia/Shanghai'
                };
                const query = new URLSearchParams(params).toString();
                const csrf = document.cookie.match(
                    /(?:^|;\\s*)passport_csrf_token=([^;]+)/
                );
                const headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'agw-js-conv': 'str',
                    'last-event-id': 'undefined',
                    'x-lgw-req-sdk-type': '3',
                    'x-net-sdk-domain-dispatch': '1',
                    'x-request-id': crypto.randomUUID(),
                    'x-ttnet-bypass-decompression': '1'
                };
                if (csrf) headers['x-tt-passport-csrf-token'] = csrf[1];
                await fetch('/chat/completion?' + query, {
                    method: 'POST',
                    headers,
                    body,
                    credentials: 'include'
                });
            }""",
            [json.dumps(payload, ensure_ascii=False, separators=(",", ":"))],
        )
        return await asyncio.wait_for(captured, timeout=60)


async def replay(
    url: str,
    headers: dict[str, str],
    cookies: dict[str, str],
    payload: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if mode == "nosig":
        query.pop("msToken", None)
        query.pop("a_bogus", None)
    elif mode == "tamper":
        query["web_tab_id"] = "00000000-0000-4000-8000-000000000000"
    target = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(sorted(query.items())),
            "",
        )
    )
    request_headers = {
        key: value
        for key, value in headers.items()
        if key.lower()
        in {
            "accept",
            "content-type",
            "agw-js-conv",
            "last-event-id",
            "x-lgw-req-sdk-type",
            "x-net-sdk-domain-dispatch",
            "x-request-id",
            "x-ttnet-bypass-decompression",
            "x-flow-trace",
            "x-tt-passport-csrf-token",
        }
    }
    request_headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36 SamanthaDoubaoWork/2.36.11"
    )
    async with httpx.AsyncClient(
        cookies=cookies,
        timeout=httpx.Timeout(180, connect=20),
        follow_redirects=True,
    ) as client:
        async with client.stream(
            "POST", target, headers=request_headers, json=payload
        ) as response:
            body = (await response.aread()).decode("utf-8", "replace")
            return {
                "mode": mode,
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "body_prefix": body[:1200],
            }


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="只回复 PROBE")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "DoubaoWork" / "chats" / "probe",
    )
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    url, headers, cookies, identity = await capture(args.prompt, args.workspace)
    payload = build_payload(
        args.prompt, args.workspace, str(identity.get("device_id") or "")
    )
    print(
        json.dumps(
            {
                "captured": True,
                "device_id": str(identity.get("device_id") or "")[:6] + "...",
                "web_id": str(identity.get("web_id") or "")[:6] + "...",
                "has_ms_token": "msToken" in url,
                "has_a_bogus": "a_bogus" in url,
                "cookie_count": len(cookies),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for mode in ("exact", "tamper", "nosig"):
        try:
            result = await replay(url, headers, cookies, payload, mode)
        except Exception as exc:
            result = {
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
