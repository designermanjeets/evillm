"""Database engine configuration for Logistics Email AI."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import structlog

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Global engine instance
_database_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_database_url() -> str:
    """Get database connection URL from settings."""
    settings = get_settings()
    return settings.database.url


async def get_database_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _database_engine
    
    if _database_engine is None:
        database_url = get_database_url()
        
        logger.info("Creating database engine", url=database_url.replace(database_url.split('@')[-1], '***'))
        
        _database_engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            poolclass=NullPool,  # Use connection pooling in production
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "server_settings": {
                    "application_name": "evillm"
                }
            }
        )
        
        logger.info("Database engine created successfully")
    
    return _database_engine


async def get_session_factory() -> async_sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    
    if _session_factory is None:
        engine = await get_database_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=None,  # Use default session class
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        logger.info("Session factory created successfully")
    
    return _session_factory


async def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        engine = await get_database_engine()
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
        return True
    except Exception as exc:
        logger.error("Database connection check failed", exc_info=exc)
        return False


async def get_database_info() -> dict:
    """Get database information including version and current user."""
    try:
        engine = await get_database_engine()
        async with engine.begin() as conn:
            # Get PostgreSQL version
            version_result = await conn.execute(text("SELECT version()"))
            version = await version_result.fetchone()
            
            # Get current user
            user_result = await conn.execute(text("SELECT current_user"))
            user = await user_result.fetchone()
            
            # Get current database
            db_result = await conn.execute(text("SELECT current_database()"))
            database = await db_result.fetchone()
            
            return {
                "version": version[0] if version else "Unknown",
                "user": user[0] if user else "Unknown",
                "database": database[0] if database else "Unknown",
                "status": "connected"
            }
    except Exception as exc:
        logger.error("Failed to get database info", exc_info=exc)
        return {
            "status": "error",
            "error": str(exc)
        }


async def close_database_engine():
    """Close the database engine."""
    global _database_engine, _session_factory
    
    if _session_factory:
        await _session_factory.close()
        _session_factory = None
        logger.info("Session factory closed")
    
    if _database_engine:
        await _database_engine.dispose()
        _database_engine = None
        logger.info("Database engine closed")
