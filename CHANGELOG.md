# Changelog

## 0.1.0 - 2026-11-13

首个公开版本。

### 新增

- 豆包工作电脑版账号池，支持直接请求上游 SSE 并转换为 OpenAI 兼容响应。
- `GET /v1/models`、`POST /v1/chat/completions`、流式 SSE 与非流式响应。
- 8 个模型目录条目，对外模型名和上游 `model_item_key` 独立映射。
- 新版六页控制台：总览、网关、账号、用量、设置、日志。
- 控制台账号导入、本机客户端扫描、启停、解除冷却和模型测试。
- 请求日志与用量聚合，只保留元数据，不保存提示词、响应正文或 Cookie。
- Windows / Linux 启动脚本、Docker 部署和基础测试。

### 验证

- `29 passed`。
- `deepseek-v4.1-flash` 与 `seed-lite-7b` 均完成真实上游调用。
