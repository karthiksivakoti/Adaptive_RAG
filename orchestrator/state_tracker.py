# risk_rag_system/orchestrator/state_tracker.py

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel
from loguru import logger
import time
from datetime import datetime
import json
from pathlib import Path

class NodeState(BaseModel):
    """State information for a single node"""
    node_id: str
    status: str  # active, idle, error
    last_active: float
    processed_items: int
    errors: List[Dict[str, Any]]
    metrics: Dict[str, float]

class SystemMetrics(BaseModel):
    """System-wide metrics"""
    total_processed: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0

class StateTracker:
    """Tracks and manages system state"""
    
    def __init__(self):
        self.node_states: Dict[str, NodeState] = {}
        self.system_metrics = SystemMetrics()
        self.state_history: List[Dict[str, Any]] = []
        self.active_sessions: Set[str] = set()
        self._initialize_storage()
        logger.info("Initialized StateTracker")

    def _initialize_storage(self) -> None:
        """Initialize state storage"""
        self.storage_dir = Path("./data/state")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def update_node_state(
        self,
        node_id: str,
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update state for a specific node"""
        try:
            current_time = time.time()
            
            if node_id not in self.node_states:
                self.node_states[node_id] = NodeState(
                    node_id=node_id,
                    status="idle",
                    last_active=current_time,
                    processed_items=0,
                    errors=[],
                    metrics={}
                )
            
            state = self.node_states[node_id]
            state.status = status
            state.last_active = current_time
            
            if status == "active":
                state.processed_items += 1
            
            if metrics:
                state.metrics.update(metrics)
            
            if error:
                state.errors.append({
                    **error,
                    "timestamp": current_time
                })
                
            # Update system metrics
            await self._update_system_metrics()
            
            # Save state snapshot
            self._save_state_snapshot()
            
            logger.debug(f"Updated state for node {node_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error updating node state: {e}")
            raise

    async def start_session(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Start tracking a new processing session"""
        self.active_sessions.add(session_id)
        
        # Record session start
        self.state_history.append({
            "type": "session_start",
            "session_id": session_id,
            "timestamp": time.time(),
            "metadata": metadata
        })

    async def end_session(
        self,
        session_id: str,
        status: str,
        metrics: Dict[str, Any]
    ) -> None:
        """End a processing session"""
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            
            # Record session end
            self.state_history.append({
                "type": "session_end",
                "session_id": session_id,
                "timestamp": time.time(),
                "status": status,
                "metrics": metrics
            })

    async def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        total_processed = sum(
            state.processed_items for state in self.node_states.values()
        )
        total_errors = sum(
            len(state.errors) for state in self.node_states.values()
        )
        
        # Calculate success rate
        if total_processed > 0:
            success_rate = (total_processed - total_errors) / total_processed
        else:
            success_rate = 0.0
            
        # Calculate average processing time
        processing_times = []
        for entry in self.state_history[-100:]:  # Last 100 entries
            if entry["type"] == "session_end" and "metrics" in entry:
                if "processing_time" in entry["metrics"]:
                    processing_times.append(entry["metrics"]["processing_time"])
        
        avg_processing_time = (
            sum(processing_times) / len(processing_times)
            if processing_times else 0.0
        )
        
        # Calculate throughput (items/minute)
        recent_processed = len([
            entry for entry in self.state_history[-60:]  # Last minute
            if entry["type"] == "session_end"
            and entry["status"] == "success"
        ])
        
        self.system_metrics = SystemMetrics(
            total_processed=total_processed,
            success_rate=success_rate,
            avg_processing_time=avg_processing_time,
            error_rate=1.0 - success_rate,
            throughput=recent_processed
        )

    def _save_state_snapshot(self) -> None:
        """Save current state snapshot to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.storage_dir / f"state_snapshot_{timestamp}.json"
        
        snapshot = {
            "timestamp": time.time(),
            "node_states": {
                node_id: state.dict()
                for node_id, state in self.node_states.items()
            },
            "system_metrics": self.system_metrics.dict(),
            "active_sessions": list(self.active_sessions)
        }
        
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    async def get_node_metrics(
        self,
        node_id: str,
        time_range: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get metrics for a specific node"""
        if node_id not in self.node_states:
            return {}
            
        state = self.node_states[node_id]
        metrics = state.metrics.copy()
        
        # Add time-based metrics
        if time_range:
            start_time = time.time() - time_range
            relevant_history = [
                entry for entry in self.state_history
                if entry["timestamp"] >= start_time
                and "node_id" in entry
                and entry["node_id"] == node_id
            ]
            
            metrics.update({
                "recent_processed": len(relevant_history),
                "recent_errors": len([
                    entry for entry in relevant_history
                    if entry.get("status") == "error"
                ])
            })
            
        return metrics

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        return {
            "metrics": self.system_metrics.dict(),
            "node_status": {
                node_id: {
                    "status": state.status,
                    "health": self._calculate_node_health(state)
                }
                for node_id, state in self.node_states.items()
            },
            "active_sessions": len(self.active_sessions)
        }

    def _calculate_node_health(self, state: NodeState) -> float:
        """Calculate health score for a node"""
        if not state.processed_items:
            return 1.0
            
        error_rate = len(state.errors) / state.processed_items
        time_since_active = time.time() - state.last_active
        
        # Combine factors into health score
        health_score = 1.0
        
        # Penalize for errors
        health_score *= (1.0 - error_rate)
        
        # Penalize for inactivity
        if time_since_active > 300:  # 5 minutes
            health_score *= 0.9
        if time_since_active > 3600:  # 1 hour
            health_score *= 0.7
            
        return max(0.0, min(1.0, health_score))

    def get_state(self) -> Dict[str, Any]:
        """Get current state tracker state"""
        return {
            "active_nodes": len(self.node_states),
            "active_sessions": len(self.active_sessions),
            "history_entries": len(self.state_history),
            "metrics": self.system_metrics.dict()
        }