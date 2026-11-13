from __future__ import annotations

import json

from fastapi.testclient import TestClient

from doubao_work2api.app import create_app
from doubao_work2api.config import Settings
from doubao_work2api.protocol import chunk_payload, completion_payload, sse_data


HEADERS = {"Authorization": "Bearer synthetic-api"}


def app_with_service(tmp_path, *, answer="RESPONSE_OK", reasoning="", finish="stop"):
    app = create_app(Settings(api_key="synthetic-api", accounts_dir=tmp_path / "accounts", data_dir=tmp_path / "data"))
    captured = []

    async def complete(request, **options):
        captured.append(request)
        response = completion_payload(request_id="chatcmpl_test", model=request.normalized_model, content=answer, reasoning=reasoning)
        response["choices"][0]["finish_reason"] = finish
        return response

    async def stream(request, **options):
        captured.append(request)
        if reasoning:
            yield sse_data(chunk_payload(request_id="chatcmpl_test", model=request.normalized_model, reasoning=reasoning))
        for text in [answer[:4], answer[4:]]:
            yield sse_data(chunk_payload(request_id="chatcmpl_test", model=request.normalized_model, content=text))
        yield sse_data(chunk_payload(request_id="chatcmpl_test", model=request.normalized_model, finish_reason=finish))
        yield sse_data("[DONE]")

    app.state.service.complete = complete
    app.state.service.stream = stream
    return app, captured


def events(response):
    return [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]


def test_create_continue_retrieve_delete_and_instruction_scope(tmp_path):
    app, captured = app_with_service(tmp_path)
    with TestClient(app) as client:
        first = client.post("/v1/responses", headers=HEADERS, json={"input": "first question", "instructions": "first-turn-only"})
        assert first.status_code == 200
        body = first.json()
        assert body["object"] == "response" and body["status"] == "completed"
        response_id = body["id"]
        assert client.get(f"/v1/responses/{response_id}", headers=HEADERS).json() == body
        inputs = client.get(f"/v1/responses/{response_id}/input_items", headers=HEADERS).json()
        assert inputs["data"][0]["role"] == "user"
        second = client.post("/v1/responses", headers=HEADERS, json={"previous_response_id": response_id, "input": "second question"})
        assert second.status_code == 200
        assert [message.content for message in captured[-1].messages] == ["first question", "RESPONSE_OK", "second question"]
        assert client.delete(f"/v1/responses/{response_id}", headers=HEADERS).json()["deleted"] is True
        assert client.get(f"/v1/responses/{response_id}", headers=HEADERS).status_code == 404


def test_store_false_and_key_isolation(tmp_path):
    app, _ = app_with_service(tmp_path)
    other = app.state.security.create_key({"name": "other"})
    with TestClient(app) as client:
        unstored = client.post("/v1/responses", headers=HEADERS, json={"input": "private", "store": False}).json()
        assert client.get(f"/v1/responses/{unstored['id']}", headers=HEADERS).status_code == 404
        stored = client.post("/v1/responses", headers=HEADERS, json={"input": "private"}).json()
        assert client.get(f"/v1/responses/{stored['id']}", headers={"Authorization": "Bearer " + other["key"]}).status_code == 404


def test_stream_incremental_events_final_body_and_reasoning_replay(tmp_path):
    app, captured = app_with_service(tmp_path, reasoning="  preserved reasoning\n")
    with TestClient(app) as client:
        response = client.post("/v1/responses", headers=HEADERS, json={"input": "question", "stream": True, "include": ["reasoning.encrypted_content"]})
        assert response.status_code == 200
        emitted = events(response)
        assert emitted[0]["type"] == "response.created"
        assert any(event["type"] == "response.output_text.delta" for event in emitted)
        assert emitted[-1]["type"] == "response.completed"
        assert [event["sequence_number"] for event in emitted] == sorted(event["sequence_number"] for event in emitted)
        final = emitted[-1]["response"]
        reasoning = next(item for item in final["output"] if item["type"] == "reasoning")
        done_item = next(event["item"] for event in emitted if event["type"] == "response.output_item.done" and event["item"]["type"] == "reasoning")
        assert done_item == reasoning
        assert reasoning["encrypted_content"]
        replay = client.post("/v1/responses", headers=HEADERS, json={"input": final["output"] + [{"role": "user", "content": "continue"}], "store": False})
        assert replay.status_code == 200
        assert any(getattr(message, "reasoning_content", None) == "  preserved reasoning\n" for message in captured[-1].messages)


def test_compaction_token_is_replayable_and_not_plaintext(tmp_path):
    app, captured = app_with_service(tmp_path, answer="User asked to keep the marker ALPHA.")
    with TestClient(app) as client:
        response = client.post("/v1/responses/compact", headers=HEADERS, json={"input": [{"role": "user", "content": "remember ALPHA"}, {"role": "assistant", "content": "noted"}, {"role": "user", "content": "continue"}]})
        assert response.status_code == 200
        compact = response.json()
        assert compact["object"] == "response.compaction"
        assert "ALPHA" not in compact["output"][0]["encrypted_content"]
        replay = client.post("/v1/responses", headers=HEADERS, json={"input": compact["output"] + [{"role": "user", "content": "what did I ask?"}]})
        assert replay.status_code == 200
        assert any("ALPHA" in str(message.content) for message in captured[-1].messages)


def test_incomplete_stream_is_not_reported_completed(tmp_path):
    app, _ = app_with_service(tmp_path, answer="partial", finish="length")
    with TestClient(app) as client:
        response = client.post("/v1/responses", headers=HEADERS, json={"input": "question", "stream": True, "max_output_tokens": 16})
    emitted = events(response)
    assert emitted[-1]["type"] == "response.incomplete"
    assert emitted[-1]["response"]["incomplete_details"]["reason"] == "max_output_tokens"


def test_response_input_image_and_json_schema_map_to_chat(tmp_path):
    app, captured = app_with_service(tmp_path, answer='{"value":1}')
    with TestClient(app) as client:
        response = client.post("/v1/responses", headers=HEADERS, json={
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "describe"}, {"type": "input_image", "image_url": "https://images.example.invalid/photo.png"}]}],
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": True}},
        })
    assert response.status_code == 200
    assert captured[0].messages[0].content[1]["type"] == "image_url"
    assert captured[0].response_format["type"] == "json_schema"


def test_item_reference_is_owner_scoped(tmp_path):
    app, captured = app_with_service(tmp_path)
    with TestClient(app) as client:
        first = client.post("/v1/responses", headers=HEADERS, json={"input": "question"}).json()
        item = first["output"][0]["id"]
        replay = client.post("/v1/responses", headers=HEADERS, json={"input": [{"type": "item_reference", "id": item}, {"role": "user", "content": "continue"}]})
        assert replay.status_code == 200
        from doubao_work2api.protocol import text_content
        assert text_content(captured[-1].messages[0]) == "RESPONSE_OK"


def test_anthropic_endpoint_is_not_added(tmp_path):
    app, _ = app_with_service(tmp_path)
    with TestClient(app) as client:
        assert client.post("/v1/messages", headers=HEADERS, json={}).status_code == 404
