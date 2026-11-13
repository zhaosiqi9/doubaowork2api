# doubaowork2api

把豆包工作电脑版的模型转换为 OpenAI 兼容 API。服务端只保存用户自行导入的
账号 Cookie 和 `device_id` / `web_id`，运行时直接请求豆包上游，不需要持续
运行豆包工作电脑版、浏览器或 Windows 桌面。

当前版本：`0.2.1`。支持文字与图片输入的 Chat Completions、工具调用文本桥接，
不包含生图或生视频接口。

## MVP 功能

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI 流式 `chat.completion.chunk` SSE 和非流式响应
- `Authorization: Bearer <key>` / `x-api-key` 鉴权
- 多账号轮换；401/403 禁用账号，429/5xx 临时冷却
- 账号文件只从本地 `accounts/` 读取
- 同源新版 WebUI：总览（含模型清单）、账号、用量、设置、日志
- 独立管理密码与管理会话，多 API Key、账号/模型白名单和 Key 并发上限
- 账号搜索、状态/分组/错误码筛选，多选导出、删除、启停和分组
- 每账号独立 HTTP 会话与 Cookie 容器，导入更新保留管理信息
- 模型目录与豆包上游 `model_item_key` 分离映射
- 工具定义、工具选择及工具结果历史转换；流式与非流式返回 OpenAI `tool_calls`
- DeepSeek Flash 五档思考强度映射，总览展示已核实的模型能力
- 图片 URL / Base64 输入、按账号上传与图片附件转换，面板支持多选图片测试

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
$env:DOUBAO_WORK_ADMIN_PASSWORD = "your-admin-password"
python -m doubao_work2api
```

默认监听 `127.0.0.1:18789`。Linux 服务器同样运行：

```bash
python3 -m pip install -e .
DOUBAO_WORK_HOST=0.0.0.0 DOUBAO_WORK_API_KEY='sk-your-key' DOUBAO_WORK_ADMIN_PASSWORD='your-admin-password' python3 -m doubao_work2api
```

公网部署必须放在带 TLS 和访问控制的反向代理之后，并且不要暴露账号目录。

浏览器打开 `http://127.0.0.1:18789/`，输入独立的管理密码进入控制台。
首次初始化取 `DOUBAO_WORK_ADMIN_PASSWORD`，未设置时为 `admin`，请在设置中修改。
API Key 仅用于模型接口，不能登录面板或操作账号。控制台支持：

- 粘贴单个/多个账号 JSON，或一次选择多个 JSON 文件导入
- 账号昵称、备注、分组、筛选，以及对勾选账号批量启停、删除、分组和解除冷却
- 按选择范围导出完整账号 JSON，再通过导入恢复；导出文件包含登录凭证
- 多 API Key 管理；密钥创建时仅显示一次，列表和配置文件不保存明文 Key
- 为 Key 绑定账号、模型和并发上限；模型清单和实际请求执行相同的模型限制
- 保存默认模型、请求超时和故障冷却；修改管理密码后所有管理会话失效
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

## 工具调用与思考强度

`POST /v1/chat/completions` 支持 OpenAI `tools`（`type: "function"`）、
`tool_choice`（`auto`、`none`、`required` 或指定函数）与 `parallel_tool_calls`。
工具定义会转换为提示词，模型按约定输出结构化文本后，网关校验并返回
`message.tool_calls` 和 `finish_reason: "tool_calls"`。工具由调用方执行，
随后以 `role: "tool"`、`tool_call_id` 回传；网关保留调用 ID、名称和参数历史。
这属于文本桥接，效果取决于模型遵循约定的能力，不等同于上游原生工具协议。
工具请求的答案先缓冲校验后再发送 SSE；普通文本请求仍即时增量返回。

DeepSeek Flash 可传 `"reasoning_effort": "low"`、`"medium"`、`"high"`、
`"xhigh"`、`"max"`，分别映射官方档位 `3 / 4 / 5 / 6 / 7`，不传时使用
默认最高档。其他模型的可配置档位尚未核实，显式请求时会明确报错。

DeepSeek Flash 的总上下文窗口由官方配置及真实响应确认是 **300,000 tokens**。
系统提示和历史也占用窗口；独立最大输入、最大输出尚未核实。
音频、视频等未支持的输入会明确报错；未支持的生成控制参数也不会被静默忽略。
用量中的 token 数仍为本地估算，不能用于精确计费。

## 图片输入

在用户消息中使用 OpenAI `image_url` 内容块，可与文字混合，也可只发送图片：

```json
{
  "model": "deepseek-v4.1-flash",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<图片Base64>"}}
    ]
  }]
}
```

`url` 也可使用公网 HTTP/HTTPS 图片地址。面板「总览 → 模型测试」可直接多选本地图片。
支持 PNG、JPEG、WebP、GIF，每次最多 **8 张**，每张最多 **10 MiB**，总计最多
**20 MiB**，以上限制包含历史消息中的图片。下载 URL 只允许公网主机的 80/443 端口，
不接受登录凭证、内网地址和指向内网的重定向；无法下载的图片可改用 Base64。

网关会把图片上传到豆包官方存储并注册为当前账号的图片资源，再随聊天请求发送。
图片仅在请求内存中处理，不在网关落盘，上传授权和图片 URI 不在账号间共享。
公网图片下载连接独立于账号连接，不发送账号 Cookie 或网关 API Key。

DeepSeek Flash 已用合成图实测识别形状、颜色和数字；其他模型的具体识图能力仍取决于上游。
`image_url.detail` 可传 `auto / low / high`，目前均采用上游默认处理，不对应精确的
OpenAI 图片 token 计价。响应 `usage` 仅估算文本，图片 token 消耗尚未计入。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOUBAO_WORK_HOST` | `127.0.0.1` | 监听地址 |
| `DOUBAO_WORK_PORT` | `18789` | 监听端口 |
| `DOUBAO_WORK_API_KEY` | `sk-local-change-me` | 网关 API Key |
| `DOUBAO_WORK_ADMIN_PASSWORD` | `admin` | 首次初始化管理密码，与 API Key 独立 |
| `DOUBAO_WORK_ACCOUNTS_DIR` | `accounts` | 账号 JSON 目录 |
| `DOUBAO_WORK_DATA_DIR` | `data` | 工作区等运行时数据 |
| `DOUBAO_WORK_TIMEOUT` | `180` | 单次上游请求超时秒数 |
| `DOUBAO_WORK_COOLDOWN_SECONDS` | `60` | 限流/故障冷却秒数 |

设置保存在 `data/settings.json`。首次初始化时环境变量中的 API Key 会导入为
默认 Key；后续以持久化设置为准，停用或删除后不会因重启恢复。
默认模型、超时、冷却在面板保存后生效并在重启后保留；有上游请求运行时先等待完成再保存。
端口、目录和上游地址仍由环境变量配置并在重启后生效。

模型测试需要单独填写 API Key。管理会话仅用于 `/admin/*`，不授予模型调用权限。
导入导出、管理设置和 API Key 限制适用于单进程部署；多实例不要共用可写账号/设置目录。

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

0.2.0 补充实测：

- 指定 `lookup_test_value` 合成函数和 `{"key":"ALPHA"}` 参数，真实上游经转换后返回标准 `tool_calls`。
- 回传合成工具结果后，流式响应正确输出 `BRIDGE_020_OK` 并以 `[DONE]` 结束。
- DeepSeek Flash 五档思考参数均可完成请求；这不代表已测出各档推理质量的差异。

0.2.1 补充实测：

- 仅发送合成图及不含答案的描述请求，DeepSeek Flash 正确识别红色正方形、蓝色圆形和数字 `314`。
- 公网图片 URL 经 DNS 校验、固定 IP 与 TLS SNI 的下载路径实测通过。

## 安全与限制

- 本项目的账号文件包含登录凭证，按密码级别保护。
- 「离线扫描」只在用户点击时读取本机豆包客户端的 Cookie 数据库和 Local
  Storage，使用当前 Windows 用户的 DPAPI 解密，数据只写入本地 `accounts/`；
  不会上传到第三方。运行时不需要豆包客户端常驻，也不内置签名算法。
- 上游协议、模型键和风控可能变化；当前实现文字/图片 Chat Completions 和工具调用文本桥接。
- `usage` 为本地文本估算，不含精确图片 token，不是上游真实 token 账单。
