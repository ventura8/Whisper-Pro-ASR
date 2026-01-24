---
description: Build and run the Docker container
---

> [!CAUTION]
> **NEVER build or run this workflow locally!** This project requires Intel NPU/GPU hardware that is not available on this development machine.


# Build & Run

// turbo-all

1. Build and start:
```bash
docker compose up -d --build
```

2. Watch logs:
```bash
docker compose logs -f
```

3. Stop:
```bash
docker compose down
```
