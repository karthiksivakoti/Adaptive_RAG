# risk_rag_system/utils/monitoring.py

from typing import Dict, Any, Optional, List, Set, Callable
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
import psutil
import numpy as np
from loguru import logger
import json
from pathlib import Path
import torch
import time
from collections import deque
import aiofiles
import os

class MetricValue(BaseModel):
    """Structure for metric values"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = {}

class MetricConfig(BaseModel):
    """Configuration for metrics collection"""
    name: str
    type: str  # counter, gauge, histogram
    description: str
    unit: str = ""
    labels: List[str] = []
    buckets: Optional[List[float]] = None  # for histograms

class AlertConfig(BaseModel):
    """Configuration for metric alerts"""
    metric_name: str
    condition: str  # "above", "below", "equals"
    threshold: float
    duration: int = 0  # seconds, 0 for immediate
    severity: str = "warning"
    labels: Dict[str, str] = {}

class Alert(BaseModel):
    """Structure for alerts"""
    config: AlertConfig
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    value: float
    message: str

class MetricsStorage:
    """Handles metric persistence"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def store_metric(self, metric: MetricValue, metric_name: str) -> None:
        """Store metric value to disk"""
        async with self._lock:
            file_path = self.storage_dir / f"{metric_name}_{datetime.now():%Y%m%d}.jsonl"
            async with aiofiles.open(file_path, mode='a') as f:
                await f.write(f"{metric.json()}\n")

    async def load_metrics(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Load metrics from disk"""
        metrics = []
        pattern = f"{metric_name}_*.jsonl"
        
        for file_path in self.storage_dir.glob(pattern):
            async with aiofiles.open(file_path, mode='r') as f:
                async for line in f:
                    metric = MetricValue.parse_raw(line)
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    metrics.append(metric)
        
        return sorted(metrics, key=lambda x: x.timestamp)

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.metrics: Dict[str, Dict[str, deque]] = {}
        self.configs: Dict[str, MetricConfig] = {}
        self.window_size = 3600  # 1 hour of history
        self._lock = asyncio.Lock()
        self.storage = MetricsStorage(storage_dir or Path("./data/metrics"))

    async def register_metric(self, config: MetricConfig) -> None:
        """Register a new metric"""
        async with self._lock:
            if config.name not in self.metrics:
                self.metrics[config.name] = {}
                self.configs[config.name] = config
                logger.info(f"Registered metric: {config.name}")

    async def record_value(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value"""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not registered")
            
        config = self.configs[metric_name]
        labels = labels or {}
        
        # Validate labels
        if set(labels.keys()) != set(config.labels):
            raise ValueError(f"Invalid labels for metric {metric_name}")
        
        # Create label key
        label_key = json.dumps(labels, sort_keys=True)
        
        metric_value = MetricValue(
            value=value,
            timestamp=datetime.now(),
            labels=labels
        )
        
        async with self._lock:
            if label_key not in self.metrics[metric_name]:
                self.metrics[metric_name][label_key] = deque(maxlen=self.window_size)
            
            self.metrics[metric_name][label_key].append(metric_value)
            
            # Store metric to disk
            await self.storage.store_metric(metric_value, metric_name)

    async def get_metric_values(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Get metric values with optional filtering"""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not registered")
        
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        
        # Combine in-memory and stored metrics
        async with self._lock:
            memory_values = []
            if label_key in self.metrics[metric_name]:
                memory_values = list(self.metrics[metric_name][label_key])
            
            stored_values = await self.storage.load_metrics(
                metric_name,
                start_time,
                end_time
            )
            
            all_values = memory_values + stored_values
            
            # Filter by time
            if start_time:
                all_values = [v for v in all_values if v.timestamp >= start_time]
            if end_time:
                all_values = [v for v in all_values if v.timestamp <= end_time]
            
            # Remove duplicates and sort
            unique_values = {v.timestamp: v for v in all_values}
            return sorted(unique_values.values(), key=lambda x: x.timestamp)

class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.stopping = False
        asyncio.create_task(self._setup_metrics())

    async def _setup_metrics(self) -> None:
        """Setup resource metrics"""
        metrics = [
            MetricConfig(
                name="system_cpu_usage",
                type="gauge",
                description="System CPU usage percentage",
                unit="percent"
            ),
            MetricConfig(
                name="system_memory_usage",
                type="gauge",
                description="System memory usage percentage",
                unit="percent"
            ),
            MetricConfig(
                name="system_disk_usage",
                type="gauge",
                description="System disk usage percentage",
                unit="percent"
            ),
            MetricConfig(
                name="gpu_memory_usage",
                type="gauge",
                description="GPU memory usage percentage",
                unit="percent",
                labels=["device"]
            ),
            MetricConfig(
                name="network_io",
                type="counter",
                description="Network I/O bytes",
                unit="bytes",
                labels=["direction"]
            )
        ]
        
        for metric in metrics:
            await self.collector.register_metric(metric)

    async def start_monitoring(self, interval: float = 1.0) -> None:
        """Start resource monitoring"""
        self.stopping = False
        
        while not self.stopping:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
                await asyncio.sleep(interval)

    async def _collect_metrics(self) -> None:
        """Collect current resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        await self.collector.record_value("system_cpu_usage", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        await self.collector.record_value("system_memory_usage", memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        await self.collector.record_value("system_disk_usage", disk.percent)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        await self.collector.record_value(
            "network_io",
            net_io.bytes_sent,
            {"direction": "sent"}
        )
        await self.collector.record_value(
            "network_io",
            net_io.bytes_recv,
            {"direction": "received"}
        )
        
        # GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    usage_percent = (memory_allocated / total_memory) * 100
                    
                    await self.collector.record_value(
                        "gpu_memory_usage",
                        usage_percent,
                        {"device": f"cuda:{i}"}
                    )
                except Exception as e:
                    logger.error(f"Error collecting GPU metrics for device {i}: {e}")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.stopping = True

class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        asyncio.create_task(self._setup_metrics())

    async def _setup_metrics(self) -> None:
        """Setup performance metrics"""
        metrics = [
            MetricConfig(
                name="request_duration",
                type="histogram",
                description="Request duration in seconds",
                unit="seconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            ),
            MetricConfig(
                name="request_throughput",
                type="counter",
                description="Request throughput",
                unit="requests/second"
            ),
            MetricConfig(
                name="error_rate",
                type="gauge",
                description="Error rate percentage",
                unit="percent"
            ),
            MetricConfig(
                name="queue_size",
                type="gauge",
                description="Request queue size",
                unit="requests"
            ),
            MetricConfig(
                name="model_latency",
                type="histogram",
                description="Model inference latency",
                unit="seconds",
                labels=["model_name"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            )
        ]
        
        for metric in metrics:
            await self.collector.register_metric(metric)

    async def record_request(
        self,
        duration: float,
        success: bool,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record request metrics"""
        await self.collector.record_value(
            "request_duration",
            duration,
            labels
        )
        
        await self.collector.record_value(
            "request_throughput",
            1,
            labels
        )
        
        if not success:
            error_rate = await self._calculate_error_rate()
            await self.collector.record_value(
                "error_rate",
                error_rate,
                labels
            )

    async def record_model_latency(
        self,
        model_name: str,
        latency: float
    ) -> None:
        """Record model inference latency"""
        await self.collector.record_value(
            "model_latency",
            latency,
            {"model_name": model_name}
        )

    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        now = datetime.now()
        start_time = now - timedelta(minutes=5)
        
        try:
            throughput = await self.collector.get_metric_values(
                "request_throughput",
                start_time=start_time
            )
            errors = await self.collector.get_metric_values(
                "error_rate",
                start_time=start_time
            )
            
            if not throughput:
                return 0.0
            
            total_requests = sum(v.value for v in throughput)
            total_errors = len(errors)
            
            return (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")
            return 0.0

class AlertManager:
    """Manages system alerts"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_configs: Optional[List[AlertConfig]] = None
    ):
        self.collector = metrics_collector
        self.alert_configs = alert_configs or []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.stopping = False
        self._setup_default_configs()

    def _setup_default_configs(self) -> None:
        """Setup default alert configurations"""
        default_configs = [
            AlertConfig(
                metric_name="system_cpu_usage",
                condition="above",
                threshold=90.0,
                duration=300,  # 5 minutes
                severity="critical"
            ),
            AlertConfig(
                metric_name="system_memory_usage",
                condition="above",
                threshold=85.0,
                duration=300,
                severity="critical"
            ),
            AlertConfig(
                metric_name="error_rate",
                condition="above",
                threshold=5.0,
                duration=300,
                severity="warning"
            ),
            AlertConfig(
                metric_name="request_duration",
                condition="above",
                threshold=5.0,
                duration=300,
                severity="warning"
            )
        ]
        
        self.alert_configs.extend(default_configs)

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler"""
        self.alert_handlers.append(handler)

    async def start_monitoring(self, interval: float = 1.0) -> None:
        """Start alert monitoring"""
        self.stopping = False
        
        while not self.stopping:
            try:
                for config in self.alert_configs:
                    await self._check_alert(config)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(interval)

    async def _check_alert(self, config: AlertConfig) -> None:
        """Check single alert condition"""
        try:
            # Get recent metric values
            values = await self.collector.get_metric_values(
                config.metric_name,
                config.labels,
                start_time=datetime.now() - timedelta(seconds=config.duration)
            )
            
            if not values:
                return
            
            current_value = values[-1].value
            alert_key = f"{config.metric_name}_{json.dumps(config.labels, sort_keys=True)}"
            
            # Calculate threshold violation
            threshold_violated = False
            if config.condition == "above":
                threshold_violated = current_value > config.threshold
            elif config.condition == "below":
                threshold_violated = current_value < config.threshold
            elif config.condition == "equals":
                threshold_violated = abs(current_value - config.threshold) < 1e-6
            
            # Check duration condition
            condition_met = False
            if config.duration == 0:
                condition_met = threshold_violated
            else:
                violation_start = None
                for value in values:
                    if config.condition == "above" and value.value > config.threshold:
                        violation_start = violation_start or value.timestamp
                    elif config.condition == "below" and value.value < config.threshold:
                        violation_start = violation_start or value.timestamp
                    elif config.condition == "equals" and abs(value.value - config.threshold) < 1e-6:
                        violation_start = violation_start or value.timestamp
                    else:
                        violation_start = None
                
                if violation_start:
                    duration = (datetime.now() - violation_start).total_seconds()
                    condition_met = duration >= config.duration
            
            # Handle alert state changes
            if condition_met:
                if alert_key not in self.active_alerts:
                    alert = Alert(
                        config=config,
                        triggered_at=datetime.now(),
                        value=current_value,
                        message=self._format_alert_message(config, current_value)
                    )
                    self.active_alerts[alert_key] = alert
                    
                    # Notify handlers
                    for handler in self.alert_handlers:
                        try:
                            await self._call_handler(handler, alert)
                        except Exception as e:
                            logger.error(f"Error in alert handler: {e}")
            else:
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved_at = datetime.now()
                    
                    # Notify handlers of resolution
                    for handler in self.alert_handlers:
                        try:
                            await self._call_handler(handler, alert)
                        except Exception as e:
                            logger.error(f"Error in alert handler: {e}")
                    
                    del self.active_alerts[alert_key]
                    
        except Exception as e:
            logger.error(f"Error checking alert {config.metric_name}: {e}")

    def _format_alert_message(self, config: AlertConfig, value: float) -> str:
        """Format alert message"""
        return (
            f"Alert: {config.metric_name} is {config.condition} threshold "
            f"({value:.2f} {config.condition} {config.threshold:.2f})"
        )

    async def _call_handler(self, handler: Callable, alert: Alert) -> None:
        """Call alert handler with proper async handling"""
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)

    def stop_monitoring(self) -> None:
        """Stop alert monitoring"""
        self.stopping = True

class SystemMonitor:
    """Main monitoring system coordinator"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.metrics_collector = MetricsCollector(storage_dir)
        # Initialize these as None
        self.resource_monitor = None
        self.performance_monitor = None
        self.alert_manager = None
        self.tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize all monitoring components"""
        self.resource_monitor = ResourceMonitor(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        await self.resource_monitor._setup_metrics()

    async def start(self) -> None:
        """Start all monitoring components"""
        try:
            # Initialize monitors here
            self.resource_monitor = ResourceMonitor(self.metrics_collector)
            await self.resource_monitor._setup_metrics()
            self.performance_monitor = PerformanceMonitor(self.metrics_collector)
            self.alert_manager = AlertManager(self.metrics_collector)
            
            self.tasks = [
                asyncio.create_task(self.resource_monitor.start_monitoring()),
                asyncio.create_task(self.alert_manager.start_monitoring())
            ]
            logger.info("Started system monitoring")
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")

    async def stop(self) -> None:
        """Stop all monitoring components"""
        try:
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
            if self.alert_manager:
                self.alert_manager.stop_monitoring()
            
            for task in self.tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.tasks.clear()
            logger.info("Stopped system monitoring")
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    async def get_system_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[MetricValue]]:
        """Get all system metrics"""
        metrics = {}
        for metric_name in self.metrics_collector.metrics:
            values = await self.metrics_collector.get_metric_values(
                metric_name,
                start_time=start_time,
                end_time=end_time
            )
            metrics[metric_name] = values
        return metrics

# Create global instance
system_monitor = SystemMonitor()