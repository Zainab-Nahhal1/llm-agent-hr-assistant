Sample requests and usage for the AI HR Assistant.

- Start the app (see project README for setup).
- Use the following examples to interact with the HTTP API.

Example: Check leave balance (curl)

```bash
curl -s -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Check my leave balance for E12345"}'
```

Example: Clear session

```bash
curl -X POST http://127.0.0.1:5000/clear
```
