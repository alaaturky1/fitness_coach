# Real-Time AI Fitness Coaching Backend

Production API backend for real-time fitness coaching over REST and WebSocket.

## Server Entry Point

The ASGI application is:

```bash
app.main:app
```

The included container starts it with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Required Environment

Set these on the server:

```bash
FITCOACH_API_KEY=replace-with-a-secure-key
FITCOACH_REQUIRE_API_KEY=true
FITCOACH_LOG_LEVEL=INFO
```

Optional Redis session storage:

```bash
FITCOACH_USE_REDIS=true
FITCOACH_REDIS_URL=redis://user:password@redis-host:6379/0
```

Optional LLM feedback:

```bash
FITCOACH_LLM_ENABLED=true
FITCOACH_LLM_API_KEY=replace-with-provider-key
FITCOACH_LLM_MODEL=gpt-4o-mini
```

## API

- `GET /health`
- `POST /start-session`
- `POST /analyze-frame`
- `POST /end-session`
- `GET /session-summary/{session_id}`
- `GET /stats`
- `WS /ws/session/{session_id}`

All protected endpoints use:

```http
X-API-Key: <FITCOACH_API_KEY>
```

For browser WebSocket clients that cannot send custom headers, pass:

```text
?x_api_key=<FITCOACH_API_KEY>
```

## Deployment Contents

The server only needs:

- `app/`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`
- `.env.example`

Local virtual environments, caches, debug scripts, tests, and browser demo files are intentionally excluded.
# llm-model
# llm-model
# fitness_coach
# fitness_coach
