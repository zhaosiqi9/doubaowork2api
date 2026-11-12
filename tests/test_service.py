from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from doubao_work2api.accounts import Account, AccountPool
from doubao_work2api.errors import UpstreamError
from doubao_work2api.models import ChatCompletionRequest
from doubao_work2api.service import ChatService


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def stream(
        self, account: Account, payload: dict
    ) -> AsyncIterator[dict]:
        self.calls.append(account.id)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        for event in response:
            yield event


def make_pool(tmp_path: Path) -> AccountPool:
    pool = AccountPool(tmp_path, cooldown_seconds=10)
    pool._accounts = [
        Account(
            id="one",
            cookies={"sessionid": "x"},
            device_id="d1",
            web_id="w1",
        ),
        Account(
            id="two",
            cookies={"sessionid": "y"},
            device_id="d2",
            web_id="w2",
        ),
    ]
    return pool


@pytest.mark.asyncio
async def test_service_rotates_to_second_account_on_failure(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            UpstreamError("temporary failure"),
            [
                {
                    "_event": "STREAM_CHUNK",
                    "patch_op": [
                        {
                            "patch_value": {
                                "content_block": [
                                    {
                                        "block_type": 10000,
                                        "content": {
                                            "text_block": {"text": "OK"}
                                        },
                                    }
                                ]
                            }
                        }
                    ],
                }
            ],
        ]
    )
    pool = make_pool(tmp_path)
    service = ChatService(
        pool,
        transport,  # type: ignore[arg-type]
        model_key="key",
        app_id="app",
        workspace_root=tmp_path,
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}]
    )
    result = await service.complete(request)
    assert result["choices"][0]["message"]["content"] == "OK"
    assert transport.calls == ["one", "two"]


@pytest.mark.asyncio
async def test_service_does_not_retry_after_partial_output(
    tmp_path: Path,
) -> None:
    class PartialTransport(FakeTransport):
        async def stream(
            self, account: Account, payload: dict
        ) -> AsyncIterator[dict]:
            self.calls.append(account.id)
            yield {
                "_event": "STREAM_CHUNK",
                "patch_op": [
                    {
                        "patch_value": {
                            "content_block": [
                                {
                                    "block_type": 10000,
                                    "content": {"text_block": {"text": "partial"}},
                                }
                            ]
                        }
                    }
                ],
            }
            raise UpstreamError("connection dropped")

    transport = PartialTransport([])
    pool = make_pool(tmp_path)
    service = ChatService(
        pool,
        transport,  # type: ignore[arg-type]
        model_key="key",
        app_id="app",
        workspace_root=tmp_path,
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}]
    )
    with pytest.raises(UpstreamError, match="connection dropped"):
        await service.complete(request)
    assert transport.calls == ["one"]
