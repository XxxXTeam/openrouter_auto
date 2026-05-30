# openrouter-auto

OpenAI 兼容的 OpenRouter 反代，自动每 5 分钟刷新 OpenRouter 上所有 `:free` 模型，并额外纳入 `openrouter/auto` 自动选模入口；基于 `key.txt` 做 API key 轮询，遇到上游 `429` 会直接返回给调用方。

## 安装

```powershell
uv sync           # 或 pip install -e .
```

## 配置

项目启动时会自动读取根目录 `.env`。可以复制 `.env.example` 后按需修改：

```powershell
Copy-Item .env.example .env
```

`OPENAI_API_KEY` 是访问本服务的外部认证 key。OpenAI SDK 会把 `api_key` 自动发送为 `Authorization: Bearer <key>`：

```env
OPENAI_API_KEY=sk-local-change-me
```

在项目根目录创建 `key.txt`，每行一个 OpenRouter API key（首次启动若不存在会自动创建空文件）：

```
sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
sk-or-v1-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

可选环境变量：

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `OPENAI_API_KEY` | 空 | 访问本服务的单个 Bearer key；为空则不启用外部认证 |
| `OPENAI_API_KEYS` | 空 | 访问本服务的多个 Bearer key，逗号分隔 |
| `OPENROUTER_BASE` | `https://openrouter.ai/api/v1` | OpenRouter API 地址 |
| `MODEL_REFRESH_INTERVAL` | `300` | 模型刷新秒数 |
| `OPENROUTER_KEY_FILE` | `key.txt` | key 文件路径 |
| `UPSTREAM_TIMEOUT` | `300` | 上游请求超时 |

## 启动

```powershell
uv run main.py
# 或
python main.py
```

## 接口

完全兼容 OpenAI SDK，把 `base_url` 指向本服务即可：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local-change-me")
print(client.models.list())
resp = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct:free",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

支持端点：

- `GET /v1/models` —— 返回缓存的免费模型（`:free` 模型和 `openrouter/auto`）
- `GET /v1/models/{id}`
- `POST /v1/chat/completions`（含 `stream: true`）
- `POST /v1/completions`
- `GET /` —— 服务概览
- `POST /admin/reload-keys` —— 热加载 `key.txt`
- `POST /admin/refresh-models` —— 立即刷新模型列表
