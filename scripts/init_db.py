"""DDL entrypoint: python -m scripts.init_db  (used by CI and locally)."""
import asyncio

from app.db import close_pool, init_db


async def main() -> None:
    await init_db()
    await close_pool()
    print("Initialized controls schema with vector(384).")


if __name__ == "__main__":
    asyncio.run(main())
