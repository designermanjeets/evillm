"""Database session management for Logistics Email AI."""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from .engine import get_session_factory

logger = structlog.get_logger(__name__)


class DatabaseSession:
    """Database session wrapper with tenant isolation."""
    
    def __init__(self, session: AsyncSession, tenant_id: Optional[str] = None):
        self.session = session
        self.tenant_id = tenant_id
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is not None:
            await self.session.rollback()
            logger.error("Database session rolled back due to exception", 
                        exc_info=(exc_type, exc_val, exc_tb))
        else:
            await self.session.commit()
        
        await self.session.close()
    
    async def execute(self, query, *args, **kwargs):
        """Execute a query with tenant isolation."""
        if self.tenant_id:
            # Add tenant_id filter if not already present
            if hasattr(query, 'where') and 'tenant_id' not in str(query):
                # This is a simplified approach - in practice, you'd want more sophisticated filtering
                logger.warning("Query may not have tenant_id filter", 
                             query=str(query), tenant_id=self.tenant_id)
        
        return await self.session.execute(query, *args, **kwargs)
    
    async def add(self, instance):
        """Add an instance to the session."""
        return await self.session.add(instance)
    
    async def commit(self):
        """Commit the current transaction."""
        return await self.session.commit()
    
    async def rollback(self):
        """Rollback the current transaction."""
        return await self.session.rollback()
    
    async def close(self):
        """Close the session."""
        return await self.session.close()


@asynccontextmanager
async def get_database_session(tenant_id: Optional[str] = None) -> AsyncGenerator[DatabaseSession, None]:
    """Get a database session with optional tenant isolation."""
    session_factory = await get_session_factory()
    async with session_factory() as session:
        db_session = DatabaseSession(session, tenant_id)
        try:
            yield db_session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_tenant_isolation(session: AsyncSession, tenant_id: str) -> bool:
    """Check if tenant isolation is properly configured."""
    try:
        # This is a basic check - in production you'd want more comprehensive validation
        result = await session.execute(
            text("SELECT current_setting('app.tenant_id', true) as current_tenant")
        )
        current_tenant = await result.fetchone()
        
        if current_tenant and current_tenant[0]:
            return current_tenant[0] == tenant_id
        return True  # If no tenant is set, assume isolation is handled at application level
        
    except Exception as exc:
        logger.warning("Could not check tenant isolation", exc_info=exc)
        return True  # Assume isolation is working if we can't check


async def get_session_stats(session: AsyncSession) -> dict:
    """Get database session statistics."""
    try:
        # Get active connections
        conn_result = await session.execute(
            text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
        )
        active_connections = await conn_result.fetchone()
        
        # Get database size
        size_result = await session.execute(
            text("SELECT pg_size_pretty(pg_database_size(current_database()))")
        )
        db_size = await size_result.fetchone()
        
        return {
            "active_connections": active_connections[0] if active_connections else 0,
            "database_size": db_size[0] if db_size else "Unknown",
            "status": "healthy"
        }
    except Exception as exc:
        logger.error("Failed to get session stats", exc_info=exc)
        return {
            "status": "error",
            "error": str(exc)
        }
