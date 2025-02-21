# risk_rag_system/utils/connection_manager.py

from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
from loguru import logger
from pydantic import BaseModel
import aiohttp
import chromadb
from concurrent.futures import ThreadPoolExecutor
from chromadb.config import Settings
import queue
import time


class ConnectionConfig(BaseModel):
    """Configuration for connection pools"""
    max_connections: int = 10
    connection_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    pool_recycle: int = 3600  # 1 hour

class ConnectionPool:
    """Generic connection pool implementation"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pool: queue.Queue = queue.Queue(maxsize=config.max_connections)
        self._active_connections = 0
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
        
    async def get_connection(self) -> Any:
        """Get a connection from the pool"""
        async with self._lock:
            try:
                # Try to get existing connection
                if not self._pool.empty():
                    conn = self._pool.get_nowait()
                    if self._is_connection_valid(conn):
                        return conn
                    else:
                        await self._close_connection(conn)
                
                # Create new connection if under limit
                if self._active_connections < self.config.max_connections:
                    conn = await self._create_connection()
                    self._active_connections += 1
                    return conn
                    
                # Wait for available connection
                raise asyncio.TimeoutError("Connection pool exhausted")
                
            except Exception as e:
                logger.error(f"Error getting connection: {e}")
                raise
    
    async def release_connection(self, conn: Any) -> None:
        """Release connection back to pool"""
        async with self._lock:
            try:
                if self._is_connection_valid(conn):
                    self._pool.put_nowait(conn)
                else:
                    await self._close_connection(conn)
                    self._active_connections -= 1
            except Exception as e:
                logger.error(f"Error releasing connection: {e}")
                raise
    
    async def cleanup(self) -> None:
        """Cleanup old connections"""
        async with self._lock:
            current_time = time.time()
            if current_time - self._last_cleanup < self.config.pool_recycle:
                return
                
            try:
                active_conns = []
                while not self._pool.empty():
                    conn = self._pool.get_nowait()
                    if self._is_connection_valid(conn):
                        active_conns.append(conn)
                    else:
                        await self._close_connection(conn)
                        self._active_connections -= 1
                
                for conn in active_conns:
                    self._pool.put_nowait(conn)
                    
                self._last_cleanup = current_time
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                raise
    
    async def close(self) -> None:
        """Close all connections"""
        async with self._lock:
            try:
                while not self._pool.empty():
                    conn = self._pool.get_nowait()
                    await self._close_connection(conn)
                self._active_connections = 0
            except Exception as e:
                logger.error(f"Error closing connections: {e}")
                raise
    
    async def _create_connection(self) -> Any:
        """Create new connection - to be implemented by subclasses"""
        raise NotImplementedError
        
    async def _close_connection(self, conn: Any) -> None:
        """Close connection - to be implemented by subclasses"""
        raise NotImplementedError
        
    def _is_connection_valid(self, conn: Any) -> bool:
        """Check if connection is valid - to be implemented by subclasses"""
        raise NotImplementedError

class ChromaDBPool(ConnectionPool):
    """Connection pool for ChromaDB"""
    
    def __init__(
        self,
        config: ConnectionConfig,
        chroma_settings: Dict[str, Any]
    ):
        super().__init__(config)
        self.chroma_settings = chroma_settings
        
    async def _create_connection(self) -> chromadb.Client:
        """Create ChromaDB connection"""
        return chromadb.Client(
            Settings(**self.chroma_settings)
        )
        
    async def _close_connection(self, conn: chromadb.Client) -> None:
        """Close ChromaDB connection"""
        await conn.close()
        
    def _is_connection_valid(self, conn: chromadb.Client) -> bool:
        """Check if ChromaDB connection is valid"""
        try:
            conn.heartbeat()
            return True
        except:
            return False

class HTTPPool(ConnectionPool):
    """Connection pool for HTTP sessions"""
    
    async def _create_connection(self) -> aiohttp.ClientSession:
        """Create HTTP session"""
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.connection_timeout)
        )
        
    async def _close_connection(self, conn: aiohttp.ClientSession) -> None:
        """Close HTTP session"""
        await conn.close()
        
    def _is_connection_valid(self, conn: aiohttp.ClientSession) -> bool:
        """Check if HTTP session is valid"""
        return not conn.closed

class ConnectionManager:
    """Manages different connection pools"""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()
        
    async def get_pool(
        self,
        pool_type: str,
        config: Optional[ConnectionConfig] = None,
        **kwargs
    ) -> ConnectionPool:
        """Get or create connection pool"""
        async with self._lock:
            if pool_type not in self.pools:
                pool_config = config or ConnectionConfig()
                if pool_type == "chromadb":
                    self.pools[pool_type] = ChromaDBPool(
                        pool_config,
                        kwargs.get("chroma_settings", {})
                    )
                elif pool_type == "http":
                    self.pools[pool_type] = HTTPPool(pool_config)
                else:
                    raise ValueError(f"Unknown pool type: {pool_type}")
                    
            return self.pools[pool_type]
    
    async def cleanup(self) -> None:
        """Cleanup all pools"""
        async with self._lock:
            for pool in self.pools.values():
                await pool.cleanup()
    
    async def close(self) -> None:
        """Close all pools"""
        async with self._lock:
            for pool in self.pools.values():
                await pool.close()
            self.pools.clear()

# Global connection manager instance
connection_manager = ConnectionManager()