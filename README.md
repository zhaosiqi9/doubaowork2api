# DoubaoWork2API

把豆包工作电脑版的 DeepSeek 模型直接转换为 OpenAI 兼容 API。服务端只保存
用户自行导入的账号 Cookie 和 `device_id` / `web_id`，运行时直接请求豆包
上游，不需要运行 DoubaoWork、浏览器、CDP 或 Windows 桌面。

## MVP 功能

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI 流式 `chat.completion.chunk` SSE 和非流式响应
- `Authorization: Bearer <key>` / `x-api-key` 鉴权
- 多账号轮换；401/403 禁用账号，429/5xx 临时冷却
- 账号文件只从本地 `accounts/` 读取

## 账号格式

把每个账号保存为 `accounts/*.json`。支持以下任一形式：

```json
{
  "id": "account-1",
  "device_id": "your-device-id",
  "web_id": "your-web-id",
  "cookies": {
    "sessionid": "...",
    "passport_csrf_token": "..."
  }
}
```

也可以使用 `"cookie": "sessionid=...; passport_csrf_token=..."`，或从浏览器
导出的 cookie 数组。`device_id` 通常来自
`localStorage.__tea_cache_tokens_1044603.user_unique_id`，`web_id` 来自同一
对象的 `web_id`。

账号目录默认是 `accounts/`，已被 `.gitignore` 排除。不要把账号文件提交到
GitHub，也不要贴到聊天中。

如果本机已有登录中的 DoubaoWork，可以通过 CDP 导出一个账号：

```powershell
python scripts/export_desktop_account.py `
  --name account-1 `
  --output accounts/account-1.json
```

CDP 必须只监听 `127.0.0.1`。导出完成后就可以关闭 DoubaoWork；本项目运行时
不再需要它。

## 启动

```powershell
cd DoubaoWork2API-hub
python -m pip install -e .
$env:DOUBAO_WORK_API_KEY = "sk-your-key"
python -m doubao_work2api
```

默认监听 `127.0.0.1:18789`。Linux 服务器同样运行：

```bash
python3 -m pip install -e .
DOUBAO_WORK_HOST=0.0.0.0 DOUBAO_WORK_API_KEY='sk-your-key' python3 -m doubao_work2api
```

公网部署必须放在带 TLS 和访问控制的反向代理之后，并且不要暴露账号目录。

Docker：

```bash
docker compose up -d --build
```

## 调用

```bash
curl http://127.0.0.1:18789/v1/models \
  -H "Authorization: Bearer sk-your-key"

curl http://127.0.0.1:18789/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "只回复 OK"}]
  }'
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOUBAO_WORK_HOST` | `127.0.0.1` | 监听地址 |
| `DOUBAO_WORK_PORT` | `18789` | 监听端口 |
| `DOUBAO_WORK_API_KEY` | `sk-local-change-me` | 网关 API Key |
| `DOUBAO_WORK_ACCOUNTS_DIR` | `accounts` | 账号 JSON 目录 |
| `DOUBAO_WORK_DATA_DIR` | `data` | 工作区等运行时数据 |
| `DOUBAO_WORK_TIMEOUT` | `180` | 单次上游请求超时秒数 |
| `DOUBAO_WORK_COOLDOWN_SECONDS` | `60` | 限流/故障冷却秒数 |

## 协议验证

`scripts/probe_direct_replay.py` 是从已登录 DoubaoWork 的 CDP 会话抓取一次真实
请求，再用外部 `httpx` 重放的诊断脚本。它只用于验证上游协议，不是运行时依赖。
正式的最小直连验证脚本是 `scripts/live_direct_probe.py`。

```powershell
python scripts/probe_direct_replay.py --prompt "只回复 PROBE"
```

当前实验已确认 `POST https://www.doubao.com/chat/completion` 在携带有效 Cookie、
`device_id`、`web_id` 时返回 SSE，并且移除 `msToken` / `a_bogus` 后仍能建立
SSE 流。正式版本仍会继续用完整新请求体验证最终文本和 Linux 环境。

2026-11-13 的实测结果：

- 新会话、新消息 ID、新 `unique_key` 的完整请求返回 `OK`
- 本网关非流式接口返回 `GATEWAY_OK` / `FINAL_OK`
- 流式接口返回标准 OpenAI SSE，文本为 `STREAM_OK`
- `system` 消息可用
- 请求体使用 `os=Linux`、Linux workspace 和 `device_name=LINUX-PROBE` 仍成功

这次测试是在 Windows 上用外部 `httpx` 客户端直连豆包上游完成，不经过
DoubaoWork 页面；Linux 服务器仍需在目标网络做一次最终回归。

## 安全与限制

- 本项目的账号文件包含登录凭证，按密码级别保护。
- 不读取 DoubaoWork 的 Cookie 数据库，也不内置签名算法。
- 上游协议、模型键和风控可能变化；当前只实现文本 Chat Completions。
- `usage` 为本地估算，不是上游真实 token 账单。
