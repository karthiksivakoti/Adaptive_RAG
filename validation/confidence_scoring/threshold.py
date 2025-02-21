# risk_rag_system/validation/confidence_scoring/threshold.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
import json
from pathlib import Path

class ThresholdConfig(BaseModel):
    """Configuration for threshold management"""
    min_samples: int = 100
    adjustment_period: int = 24  # hours
    max_adjustment: float = 0.1
    history_window: int = 7  # days
    target_error_rate: float = 0.05
    enable_auto_adjust: bool = True

class ThresholdStats(BaseModel):
    """Statistics for threshold adjustment"""
    samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    error_rate: float

class ThresholdManager:
    """Manages confidence score thresholds"""
    
    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
        storage_path: Optional[Path] = None
    ):
        self.config = config or ThresholdConfig()
        self.storage_path = storage_path or Path("./data/thresholds")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.thresholds: Dict[str, float] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.stats: Dict[str, ThresholdStats] = {}
        
        self._load_state()
        logger.info("Initialized ThresholdManager")

    def _load_state(self) -> None:
        """Load saved thresholds and history"""
        state_file = self.storage_path / "threshold_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.thresholds = state.get("thresholds", {})
                    self.history = state.get("history", {})
                    
                    # Convert stats from dict to ThresholdStats
                    self.stats = {
                        k: ThresholdStats(**v) if isinstance(v, dict) else v
                        for k, v in state.get("stats", {}).items()
                    }
            except Exception as e:
                logger.error(f"Error loading threshold state: {e}")
                self._initialize_defaults()
        else:
            self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize default thresholds"""
        self.thresholds = {
            "retrieval": 0.7,
            "generation": 0.8,
            "factual": 0.75,
            "overall": 0.7
        }
        self.history = {k: [] for k in self.thresholds}
        self.stats = {
            k: ThresholdStats(
                samples=0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                error_rate=0.0
            )
            for k in self.thresholds
        }

    def _save_state(self) -> None:
        """Save current state"""
        state_file = self.storage_path / "threshold_state.json"
        try:
            state = {
                "thresholds": self.thresholds,
                "history": self.history,
                "stats": {k: v.dict() for k, v in self.stats.items()}
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving threshold state: {e}")

    async def get_threshold(
        self,
        threshold_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get threshold value with optional context adjustment"""
        base_threshold = self.thresholds.get(
            threshold_type,
            self.thresholds["overall"]
        )
        
        if not context:
            return base_threshold
        
        # Adjust based on context
        adjustment = await self._calculate_context_adjustment(
            threshold_type,
            context
        )
        
        adjusted = base_threshold + adjustment
        return max(0.0, min(1.0, adjusted))

    async def _calculate_context_adjustment(
        self,
        threshold_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate threshold adjustment based on context"""
        adjustment = 0.0
        
        # Adjust based on criticality
        if context.get("criticality") == "high":
            adjustment += 0.1
        elif context.get("criticality") == "low":
            adjustment -= 0.05
        
        # Adjust based on confidence history
        if threshold_type in self.history:
            recent_history = self.history[threshold_type][-10:]
            if recent_history:
                avg_confidence = np.mean([h["confidence"] for h in recent_history])
                if avg_confidence > 0.9:
                    adjustment -= 0.05
                elif avg_confidence < 0.6:
                    adjustment += 0.05
        
        # Limit maximum adjustment
        return max(-self.config.max_adjustment,
                  min(self.config.max_adjustment, adjustment))

    async def record_feedback(
        self,
        threshold_type: str,
        confidence: float,
        was_correct: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record feedback for threshold adjustment"""
        # Add to history
        entry = {
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "was_correct": was_correct,
            "metadata": metadata or {}
        }
        self.history[threshold_type].append(entry)
        
        # Update stats
        stats = self.stats[threshold_type]
        threshold = self.thresholds[threshold_type]
        
        stats.samples += 1
        if confidence >= threshold:
            if was_correct:
                stats.true_positives += 1
            else:
                stats.false_positives += 1
        else:
            if was_correct:
                stats.true_negatives += 1
            else:
                stats.false_negatives += 1
        
        # Calculate metrics
        total_positive = stats.true_positives + stats.false_positives
        total_actual = stats.true_positives + stats.false_negatives
        
        if total_positive > 0:
            stats.precision = stats.true_positives / total_positive
        if total_actual > 0:
            stats.recall = stats.true_positives / total_actual
        
        if stats.precision + stats.recall > 0:
            stats.f1_score = 2 * (stats.precision * stats.recall) / (stats.precision + stats.recall)
        
        stats.error_rate = (stats.false_positives + stats.false_negatives) / stats.samples
        
        # Trigger threshold adjustment if needed
        if self.config.enable_auto_adjust:
            await self._adjust_threshold(threshold_type)
        
        # Save updated state
        self._save_state()

    async def _adjust_threshold(self, threshold_type: str) -> None:
        """Adjust threshold based on feedback"""
        stats = self.stats[threshold_type]
        
        # Check if we have enough samples
        if stats.samples < self.config.min_samples:
            return
        
        # Check if adjustment period has passed
        last_adjustment = None
        for entry in reversed(self.history[threshold_type]):
            if entry.get("was_adjustment"):
                last_adjustment = datetime.fromisoformat(entry["timestamp"])
                break
        
        if last_adjustment and (datetime.now() - last_adjustment).total_seconds() < self.config.adjustment_period * 3600:
            return
        
        # Calculate optimal threshold
        current_threshold = self.thresholds[threshold_type]
        
        if stats.error_rate > self.config.target_error_rate:
            # Too many errors, increase threshold
            adjustment = min(
                self.config.max_adjustment,
                (stats.error_rate - self.config.target_error_rate) * 0.5
            )
            new_threshold = current_threshold + adjustment
        else:
            # Error rate is good, try to lower threshold if precision allows
            if stats.precision > 0.95:
                adjustment = -min(
                    self.config.max_adjustment,
                    (0.95 - stats.error_rate) * 0.3
                )
                new_threshold = current_threshold + adjustment
            else:
                return
        
        # Apply adjustment
        self.thresholds[threshold_type] = max(0.0, min(1.0, new_threshold))
        
        # Record adjustment
        self.history[threshold_type].append({
            "timestamp": datetime.now().isoformat(),
            "was_adjustment": True,
            "old_threshold": current_threshold,
            "new_threshold": self.thresholds[threshold_type],
            "reason": f"error_rate={stats.error_rate:.3f}, precision={stats.precision:.3f}"
        })
        
        logger.info(
            f"Adjusted {threshold_type} threshold: {current_threshold:.3f} -> {self.thresholds[threshold_type]:.3f}"
        )

    async def get_stats(
        self,
        threshold_type: Optional[str] = None,
        time_range: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get threshold statistics"""
        if threshold_type:
            if threshold_type not in self.stats:
                raise ValueError(f"Unknown threshold type: {threshold_type}")
                
            stats = self.stats[threshold_type]
            history = self.history[threshold_type]
        else:
            stats = {k: v.dict() for k, v in self.stats.items()}
            history = self.history
        
        if time_range:
            cutoff = datetime.now() - timedelta(days=time_range)
            if threshold_type:
                history = [
                    h for h in history
                    if datetime.fromisoformat(h["timestamp"]) >= cutoff
                ]
            else:
                history = {
                    k: [h for h in v if datetime.fromisoformat(h["timestamp"]) >= cutoff]
                    for k, v in history.items()
                }
        
        return {
            "stats": stats.dict() if isinstance(stats, ThresholdStats) else stats,
            "history": history,
            "current_thresholds": self.thresholds if not threshold_type else {
                threshold_type: self.thresholds[threshold_type]
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "thresholds": self.thresholds,
            "stats": {k: v.dict() for k, v in self.stats.items()},
            "config": self.config.dict(),
            "history_size": {k: len(v) for k, v in self.history.items()}
        }