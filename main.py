# risk_rag_system/main.py

from typing import Dict, Any, Optional, List, Set
import asyncio
from pathlib import Path
import json
from loguru import logger
import uuid
from datetime import datetime
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import time
import os

from orchestrator.graph_manager import GraphManager
from orchestrator.confidence_router import ConfidenceRouter
from orchestrator.state_tracker import StateTracker
from llm.model_registry import ModelRegistry
from nodes.base_node import NodeInput, NodeOutput
from nodes.retriever_node import RetrieverNode
from nodes.grader_node import GraderNode
from nodes.rewriter_node import RewriterNode
from nodes.generator_node import GeneratorNode
from nodes.web_search_node import WebSearchNode
from validation.confidence_scoring.scorer import ConfidenceScorer
from validation.confidence_scoring.threshold import ThresholdManager
from utils.connection_manager import connection_manager
from utils.monitoring import system_monitor, PerformanceMonitor, MetricConfig
from utils.error_handler import error_handler

class SystemConfig:
    """System configuration"""
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("./config/system.json")
        self.load_config()

    def load_config(self) -> None:
        """Load system configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # System paths
        self.data_dir = Path(config.get("data_dir", "./data"))
        self.model_dir = Path(config.get("model_dir", "./models"))
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        
        # System settings
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        self.enable_web_search = config.get("enable_web_search", True)
        self.enable_monitoring = config.get("enable_monitoring", True)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 10)
        
        # Create directories
        for directory in [self.data_dir, self.model_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class RiskRAGSystem:
    """Main system orchestrator"""
    
    @classmethod
    async def create(cls, config: Optional[SystemConfig] = None) -> 'RiskRAGSystem':
        """Factory method to create and initialize the system asynchronously"""
        instance = cls.__new__(cls)
        await instance._async_init(config)
        return instance
        
    async def _async_init(self, config: Optional[SystemConfig] = None) -> None:
        """Async initialization"""
        self.config = config or SystemConfig()
        
        # Initialize components
        self.graph_manager = GraphManager()
        self.confidence_router = ConfidenceRouter()
        self.state_tracker = StateTracker()
        self.model_registry = ModelRegistry()
        self.confidence_scorer = ConfidenceScorer()
        self.threshold_manager = ThresholdManager()
        
        # Initialize metrics collector and performance monitor early
        if self.config.enable_monitoring:
            # Initialize required metrics for performance monitoring
            await system_monitor.metrics_collector.register_metric(MetricConfig(
                name="request_duration",
                type="histogram",
                description="Request duration in seconds",
                unit="seconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                labels=["query_type"]  # Add required labels
            ))
            await system_monitor.metrics_collector.register_metric(MetricConfig(
                name="request_throughput",
                type="counter",
                description="Request throughput",
                unit="requests/second",
                labels=["query_type"]
            ))
            await system_monitor.metrics_collector.register_metric(MetricConfig(
                name="error_rate",
                type="gauge",
                description="Error rate percentage",
                unit="percent",
                labels=["query_type"]
            ))
            await system_monitor.metrics_collector.register_metric(MetricConfig(
                name="queue_size",
                type="gauge",
                description="Request queue size",
                unit="requests"
            ))
            await system_monitor.metrics_collector.register_metric(MetricConfig(
                name="model_latency",
                type="histogram",
                description="Model inference latency",
                unit="seconds",
                labels=["model_name", "query_type"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            ))
            
        # Concurrency control
        self.request_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_requests
        )
        self.is_shutting_down = False
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4)
        )
        
        # Set up nodes
        self._setup_nodes()
        
        # Setup shutdown handling
        self._setup_shutdown_handlers()
        
        logger.info("Initialized RiskRAGSystem")

    def _setup_nodes(self) -> None:
        """Set up system nodes"""
        # Create nodes
        self.retriever = RetrieverNode("retriever")
        self.grader = GraderNode("grader")
        self.rewriter = RewriterNode("rewriter")
        self.generator = GeneratorNode("generator")
        if self.config.enable_web_search:
            self.web_search = WebSearchNode("web_search")
        
        # Add nodes to graph
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.rewriter)
        self.graph_manager.add_node(self.generator)
        if self.config.enable_web_search:
            self.graph_manager.add_node(self.web_search)
        
        # Add edges
        self.graph_manager.add_edge("retriever", "grader", "default")
        self.graph_manager.add_edge("grader", "generator", "default")
        self.graph_manager.add_edge("grader", "rewriter", "low_confidence")
        self.graph_manager.add_edge("rewriter", "retriever", "default")
        if self.config.enable_web_search:
            self.graph_manager.add_edge("retriever", "web_search", "no_results")
            self.graph_manager.add_edge("web_search", "grader", "default")

    def _setup_shutdown_handlers(self) -> None:
        """Setup system shutdown handlers"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        if not self.is_shutting_down:
            logger.info("Initiating graceful shutdown")
            asyncio.create_task(self.shutdown())

    def _track_task(self, task: asyncio.Task) -> None:
        """Track active task"""
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    async def start(self) -> None:
        """Start the system"""
        try:
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                await system_monitor.start()
            
            logger.info("RiskRAGSystem started")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            await self.shutdown()
            raise

    @error_handler.with_retry
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process user query"""
        if self.is_shutting_down:
            raise RuntimeError("System is shutting down")

        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Ensure default query type
        query_type = context.get("query_type", "general") if context else "general"
        
        # Acquire semaphore for concurrency control
        async with self.request_semaphore:
            try:
                # Start session tracking
                await self.state_tracker.start_session(
                    session_id=session_id,
                    metadata={
                        "query": query,
                        "context": context,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Create input with proper content field
                input_data = NodeInput(
                    content=query,  # Pass query as content
                    metadata={
                        "session_id": session_id,
                        "filters": context.get("filters", {}),
                        "top_k": context.get("top_k", 5),
                        **(context or {})
                    }
                )
                
                # Execute workflow
                output = await self.graph_manager.execute(input_data)
                
                # Calculate confidence
                confidence_score = await self.confidence_scorer.calculate_confidence(
                    response=output.content,
                    query=query,
                    context=output.metadata.get("source_documents", []),
                    retrieval_scores=output.metadata.get("retrieval_scores", {}),
                    generation_metadata=output.metadata.get("generation_metadata", {})
                )
                
                # Apply thresholds
                threshold = await self.threshold_manager.get_threshold(
                    "overall",
                    context=context
                )
                
                # Record performance metrics
                duration = time.time() - start_time
                await system_monitor.performance_monitor.record_request(
                    duration=duration,
                    success=True,
                    labels={"query_type": query_type}  # Use only valid labels
                )

                # Prepare response
                response = {
                    "answer": output.content,
                    "confidence_score": confidence_score.score,
                    "confidence_metrics": confidence_score.metrics.dict(),
                    "confidence_reasoning": confidence_score.reasoning,
                    "meets_threshold": confidence_score.score >= threshold,
                    "metadata": {
                        "session_id": session_id,
                        "execution_path": output.metadata.get("execution_path", []),
                        "sources": output.metadata.get("source_documents", []),
                        "processing_time": duration,
                        "threshold_used": threshold
                    }
                }

                # Record feedback if available
                if context and "feedback" in context:
                    await self.threshold_manager.record_feedback(
                        threshold_type="overall",
                        confidence=confidence_score.score,
                        was_correct=context["feedback"]["correct"],
                        metadata={
                            "query_type": context.get("query_type"),
                            "difficulty": context.get("difficulty")
                        }
                    )

                # End session tracking
                await self.state_tracker.end_session(
                    session_id=session_id,
                    status="completed",
                    metrics={
                        "confidence_score": confidence_score.score,
                        "processing_time": duration,
                        "num_sources": len(output.metadata.get("source_documents", [])),
                        "path_length": len(output.metadata.get("execution_path", []))
                    }
                )

                return response

            except Exception as e:
                # Record error metrics
                await system_monitor.performance_monitor.record_request(
                    duration=time.time() - start_time,
                    success=False,
                    labels={"query_type": context.get("query_type", "unknown")}
                )
                
                # End session with error
                await self.state_tracker.end_session(
                    session_id=session_id,
                    status="error",
                    metrics={"error": str(e)}
                )
                
                raise

    async def shutdown(self) -> None:
        """Perform graceful shutdown"""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        logger.info("Starting graceful shutdown")
        
        try:
            # Stop accepting new requests
            self.request_semaphore = asyncio.Semaphore(0)
            
            # Wait for active tasks to complete
            if self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks")
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
            
            # Cleanup components
            if self.config.enable_monitoring and hasattr(system_monitor, 'stop'):
                await system_monitor.stop()
            
            if hasattr(connection_manager, 'close'):
                await connection_manager.close()
                
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown()
            
            # Cleanup model resources
            if hasattr(self, 'model_registry'):
                self.model_registry.cleanup()
            
            logger.info("Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

    async def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            "graph_state": self.graph_manager.get_state(),
            "models": self.model_registry.get_state(),
            "confidence": {
                "scorer": self.confidence_scorer.get_state(),
                "thresholds": self.threshold_manager.get_state()
            },
            "metrics": await system_monitor.get_system_metrics(),
            "system_health": await self.state_tracker.get_system_health(),
            "active_tasks": len(self.active_tasks),
            "is_shutting_down": self.is_shutting_down
        }

async def run_system():
    """Run the RAG system"""
    # Configure logging
    logger.add(
        sys.stdout,
        format="{time} {level} {message}",
        filter="risk_rag_system",
        level="INFO"
    )
    
    # Initialize system using factory method
    system = await RiskRAGSystem.create()
    
    # Start the system
    await system.start()
    
    try:
        # Keep the system running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(run_system())