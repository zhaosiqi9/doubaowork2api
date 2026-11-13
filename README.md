# doubaowork2api

把豆包工作电脑版的模型转换为 OpenAI 兼容 API。服务端只保存用户自行导入的
账号 Cookie 和 `device_id` / `web_id`，运行时直接请求豆包上游，不需要持续
运行豆包工作电脑版、浏览器或 Windows 桌面。

当前版本：`0.1.0`。这是文本 Chat Completions 的首个公开版本，不包含图片、
视频或其他媒体接口。

## MVP 功能

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI 流式 `chat.completion.chunk` SSE 和非流式响应
- `Authorization: Bearer <key>` / `x-api-key` 鉴权
- 多账号轮换；401/403 禁用账号，429/5xx 临时冷却
- 账号文件只从本地 `accounts/` 读取
- 同源新版 WebUI：总览、网关、账号、用量、设置、日志
- 模型目录与豆包上游 `model_item_key` 分离映射

## 模型目录

`/v1/models` 当前列出以下 8 个模型。对外名称与上游内部键分开，请求某个模型
时会使用对应上游键，不会把未知模型静默改走默认 DeepSeek。

| 对外模型 | 显示名称 | 倍率 |
| --- | --- | --- |
| `deepseek-v4.1-flash` | DeepSeek V4.1 Flash | 0.01x |
| `seed-lite-7b` | 豆包 2.1 Lite | 免费 |
| `person-kimi-k3-1-pro-261111` | Kimi K3.1 Pro | 0.95x |
| `person-kimi-k3-1-pro-highspeed-261111` | Kimi K3.1 Pro 高速 | 1.44x |
| `person-kimi-k3-261111` | Kimi K3 | 1.48x |
| `person-glm-5-3-261111` | GLM 5.3 | 0.80x |
| `person-glm-5-3-flash-261111` | GLM 5.3 Flash | 0.08x |
| `person-deepseek-v4-pro-ga-260813` | DeepSeek V4 Pro | 0.46x |

模型可用性仍取决于当前账号、地区和豆包上游状态。目录展示不等于承诺无限额度。

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

如果本机已有登录过的豆包工作电脑版，可以直接在控制台点「离线扫描」：

Windows 离线扫描会读取客户端留下的 Cookie 数据库和 Local Storage，
使用当前 Windows 用户的 DPAPI 解密，然后写入本地 `accounts/`。
客户端必须关闭，否则 Chromium 会独占 Cookie 数据库。读取完成后所有
临时文件和数据库句柄立即释放，不要求客户端常驻，也不需要开放 `9222`。

```text
1. 登录一次豆包工作电脑版
2. 完全退出客户端
3. 打开控制台 → 账号 → 离线扫描
```

离线扫描不需要安装 Playwright、启动调试端口或让客户端常驻。
Playwright 不属于正式运行依赖，也不会随基础安装或 wheel 安装。

## 启动

```powershell
cd doubaowork2api
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

浏览器打开 `http://127.0.0.1:18789/`，输入 `DOUBAO_WORK_API_KEY` 即可进入
控制台。控制台支持：

- 粘贴单个/多个账号 JSON，或一次选择多个 JSON 文件导入
- 账号启用、停用、解除冷却
- 查看设备标识摘要、冷却剩余时间和最近错误
- 查看模型目录与上游映射
- 直接发起普通或流式模型测试
- 查看请求成功率、耗时、模型分布和日志
- 查看最近 200 条请求的元数据日志（不保存提示词、响应正文和 Cookie）

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

当前实验已确认 `POST https://www.doubao.com/chat/completion` 在携带有效 Cookie、
`device_id`、`web_id` 时返回 SSE，并且移除 `msToken` / `a_bogus` 后仍能建立
SSE 流。

2026-11-13 的实测结果：

- 新会话、新消息 ID、新 `unique_key` 的完整请求返回 `OK`
- 本网关非流式接口返回 `GATEWAY_OK` / `FINAL_OK`
- 流式接口返回标准 OpenAI SSE，文本为 `STREAM_OK`
- `system` 消息可用
- 请求体使用 `os=Linux`、Linux workspace 和 `device_name=LINUX-PROBE` 仍成功

这次测试是在 Windows 上用外部 `httpx` 客户端直连豆包上游完成，不经过
豆包工作电脑版页面；Linux 服务器仍需在目标网络做一次最终回归。

0.1.0 发布前的补充实测：

- `deepseek-v4.1-flash` 经本网关返回 `UI_SMOKE_OK`
- `seed-lite-7b` 经本网关返回 `SEED_SMOKE_OK`
- 两条请求均由同一个已启用账号完成，并写入正确模型名的请求日志

## 安全与限制

- 本项目的账号文件包含登录凭证，按密码级别保护。
- 「离线扫描」只在用户点击时读取本机豆包客户端的 Cookie 数据库和 Local
  Storage，使用当前 Windows 用户的 DPAPI 解密，数据只写入本地 `accounts/`；
  不会上传到第三方。运行时不需要豆包客户端常驻，也不内置签名算法。
- 上游协议、模型键和风控可能变化；当前只实现文本 Chat Completions。
- `usage` 为本地估算，不是上游真实 token 账单。
