# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from prometheus_client import Counter, Gauge, Histogram

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger

PromMetric: TypeAlias = Gauge | Counter | Histogram
PromMetricT = TypeVar("PromMetricT", bound=PromMetric)

logger = init_logger(__name__)


@dataclass
class KVConnectorStats:
    """
    Base class for KV Connector Stats, a container for transfer performance
    metrics or otherwise important telemetry from the connector.
    All sub-classes need to be serializable as stats are sent from worker to
    logger process.
    """

    # ------【PD 分离】stats 原始数据容器，子类在其中累积观测值，需可序列化传给 logger 进程 ------
    data: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """Reset the stats, clear the state."""
        # ------【PD 分离】抽象接口：清空状态，由具体 connector 实现 ------
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """
        Aggregate stats with another `KVConnectorStats` object.
        """
        # ------【PD 分离】抽象接口：跨 worker 汇总另一份 stats，由具体 connector 实现 ------
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        # ------【PD 分离】抽象接口：把一个时间段的观测归约成代表值(avg/median/sum)，供 logger 汇总 ------
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        # ------【PD 分离】抽象接口：判断是否无观测数据，由具体 connector 实现 ------
        raise NotImplementedError


class KVConnectorLogging:
    def __init__(self, kv_transfer_config: KVTransferConfig | None):
        # Instantiate the connector's stats class.
        # ------【PD 分离】配置了 connector 时解析其类，供后续 build_kv_connector_stats 使用 ------
        if kv_transfer_config and kv_transfer_config.kv_connector:
            self.connector_cls = KVConnectorFactory.get_connector_class(
                kv_transfer_config
            )
        # ------【PD 分离】初始化累加器为空 ------
        self.reset()

    def reset(self):
        # ------【PD 分离】清空传输 stats 累加器，准备下一统计区间 ------
        self.transfer_stats_accumulator: KVConnectorStats | None = None

    def observe(self, transfer_stats_data: dict[str, Any]):
        # Should not be called when a KVConnector is not configured.
        # ------【PD 分离】未配置 connector 时不应被调用，断言兜底 ------
        assert self.connector_cls is not None
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats_data to be aggregated across all workers and
        # consist of observations from a single connector or a MultiConnector.
        # ------【PD 分离】用 connector 类把原始 dict 构造成具体 stats 对象 ------
        transfer_stats = self.connector_cls.build_kv_connector_stats(
            transfer_stats_data
        )
        # ------【PD 分离】connector 未实现构建方法则告警并跳过，避免崩溃 ------
        if transfer_stats is None:
            logger.warning_once(
                "The connector %s is collecting stats but "
                "does not implement the "
                "`build_kv_connector_stats` method. "
                "Stats will not be logged.",
                self.connector_cls,
            )
            return

        if self.transfer_stats_accumulator is None:
            # ------【PD 分离】首个观测直接作为累加器起点 ------
            self.transfer_stats_accumulator = transfer_stats
        else:
            # Accumulate last interval stats.
            # ------【PD 分离】后续观测调用 aggregate 叠加到累加器 ------
            self.transfer_stats_accumulator = self.transfer_stats_accumulator.aggregate(
                transfer_stats
            )

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        # ------【PD 分离】累加器存在且有数据时才打印，避免空日志 ------
        if (
            self.transfer_stats_accumulator
            and not self.transfer_stats_accumulator.is_empty()
        ):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            # ------【PD 分离】把该区间观测归约成代表值(如平均/求和) ------
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            # ------【PD 分离】格式化成 k=v 串后统一打印一条 KV Transfer metrics 日志 ------
            xfer_metrics_str = ", ".join(f"{k}={v}" for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            # ------【PD 分离】打印完清空累加器，开始下一个统计区间 ------
            self.reset()


class KVConnectorPromMetrics:
    """
    A base class for per-connector Prometheus metric registration
    and recording.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        # ------【PD 分离】缓存 KV transfer 配置与各类型 metric 类，供子类注册指标用 ------
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._gauge_cls = metric_types[Gauge]
        self._counter_cls = metric_types[Counter]
        self._histogram_cls = metric_types[Histogram]
        self._labelnames = labelnames
        self.per_engine_labelvalues = per_engine_labelvalues

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Record the supplied transfer statistics to Prometheus metrics. These
        statistics are engine-specific, and should be recorded to a metric
        with the appropriate 'engine' label. These metric instances can be
        created using the create_metric_per_engine() helper method.
        """
        # ------【PD 分离】抽象接口：把引擎级 stats 记录到带 engine 标签的 Prom 指标，由子类实现 ------
        raise NotImplementedError


class KVConnectorProm:
    """
    Support for registering per-connector Prometheus metrics, and
    recording transfer statistics to those metrics. Uses
    KVConnectorBase.build_prom_metrics().
    """

    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        # ------【PD 分离】prom 指标对象默认为 None，未配置 connector 时保持不启用 ------
        self.prom_metrics: KVConnectorPromMetrics | None = None
        kv_transfer_config = vllm_config.kv_transfer_config
        # ------【PD 分离】配置了 connector 时才解析其类并构造指标注册对象 ------
        if kv_transfer_config and kv_transfer_config.kv_connector:
            connector_cls = KVConnectorFactory.get_connector_class(kv_transfer_config)
            # ------【PD 分离】组装 Gauge/Counter/Histogram 三类指标类，交由 connector 注册具体指标 ------
            metric_types = {
                Gauge: self._gauge_cls,
                Counter: self._counter_cls,
                Histogram: self._histogram_cls,
            }
            self.prom_metrics = connector_cls.build_prom_metrics(
                vllm_config,
                metric_types,
                labelnames,
                per_engine_labelvalues,
            )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        # ------【PD 分离】未启用 prom 指标时直接返回，无指标可记录 ------
        if self.prom_metrics is None:
            return
        # ------【PD 分离】委托给具体的 Prom 指标对象，按 engine_idx 记录 stats ------
        self.prom_metrics.observe(transfer_stats_data, engine_idx)
