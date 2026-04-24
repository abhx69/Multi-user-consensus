"""
Database Token Bridge for Express Backend Integration.

Reads OAuth tokens from the MySQL `user_connections` table shared
with the Express backend (super-fishstick/gaprio-backend).

Schema:
    user_connections(
        user_id INT, provider VARCHAR(50),
        access_token TEXT, refresh_token TEXT,
        expires_at DATETIME, metadata JSON
    )

Uses contextvars to thread user_id through async call chains
without modifying every function signature.
"""

import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any

import aiomysql

from gaprio.config import settings

logger = logging.getLogger(__name__)

# Context variable for passing user_id through async call chains
_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

# Connection pool (created lazily on first use)
_pool: aiomysql.Pool | None = None


def set_current_user_id(user_id: str | None) -> None:
    """Set the current user ID in the async context."""
    _current_user_id.set(user_id)


def get_current_user_id() -> str | None:
    """Get the current user ID from the async context."""
    return _current_user_id.get()


async def _get_pool() -> aiomysql.Pool | None:
    """Get or create the MySQL connection pool."""
    global _pool
    
    if _pool is not None and not _pool.closed:
        return _pool
    
    if not settings.db_host or not settings.db_name:
        logger.debug("MySQL not configured, skipping DB token bridge")
        return None
    
    try:
        _pool = await aiomysql.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            minsize=1,
            maxsize=5,
            autocommit=True,
        )
        logger.info(f"MySQL pool created for DB token bridge ({settings.db_host}:{settings.db_port}/{settings.db_name})")
        return _pool
    except Exception as e:
        logger.warning(f"MySQL connection failed (DB tokens unavailable): {e}")
        return None


async def get_connection_tokens(user_id: int | str, provider: str) -> dict[str, Any] | None:
    """
    Fetch OAuth tokens for a user+provider from the MySQL database.
    
    Args:
        user_id: The user's ID in the Express backend
        provider: Provider name (e.g., 'google', 'slack', 'asana')
    
    Returns:
        Dict with access_token, refresh_token, expires_at, metadata
        or None if not found / DB unavailable
    """
    pool = await _get_pool()
    if pool is None:
        return None
    
    try:
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT access_token, refresh_token, expires_at, metadata "
                    "FROM user_connections WHERE user_id = %s AND provider = %s",
                    (int(user_id), provider),
                )
                row = await cur.fetchone()
                
                if row:
                    logger.debug(f"DB tokens found for user={user_id}, provider={provider}")
                    return {
                        "access_token": row["access_token"],
                        "refresh_token": row["refresh_token"],
                        "expires_at": row["expires_at"],
                        "metadata": row["metadata"],
                    }
                else:
                    logger.debug(f"No DB tokens for user={user_id}, provider={provider}")
                    return None
    except Exception as e:
        logger.warning(f"DB token fetch failed: {e}")
        return None


async def update_connection_tokens(
    user_id: int | str,
    provider: str,
    access_token: str,
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
) -> bool:
    """
    Update OAuth tokens in the MySQL database after a refresh.
    
    Keeps the Express backend's token store in sync when the
    Python agent refreshes an access token.
    """
    pool = await _get_pool()
    if pool is None:
        return False
    
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                if refresh_token:
                    await cur.execute(
                        "UPDATE user_connections SET access_token = %s, "
                        "refresh_token = %s, expires_at = %s, updated_at = NOW() "
                        "WHERE user_id = %s AND provider = %s",
                        (access_token, refresh_token, expires_at, int(user_id), provider),
                    )
                else:
                    await cur.execute(
                        "UPDATE user_connections SET access_token = %s, "
                        "expires_at = %s, updated_at = NOW() "
                        "WHERE user_id = %s AND provider = %s",
                        (access_token, expires_at, int(user_id), provider),
                    )
                logger.debug(f"DB tokens updated for user={user_id}, provider={provider}")
                return True
    except Exception as e:
        logger.warning(f"DB token update failed: {e}")
        return False


async def close_pool() -> None:
    """Close the MySQL connection pool."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("MySQL pool closed")
