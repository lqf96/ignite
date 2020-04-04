import itertools
from typing import Callable, Any

from ignite.metrics.metric import Metric, reinit__is_reduced
from ignite.engine import Events, Engine

__all__ = ["MetricsLambda"]


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependency metrics would be attached
    automatically (but partially, e.g `is_attached()` will return False).

    Args:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.

    Example:

    .. code-block:: python

        precision = Precision(average=False)
        recall = Recall(average=False)

        def Fbeta(r, p, beta):
            return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

        F1 = MetricsLambda(Fbeta, recall, precision, 1)
        F2 = MetricsLambda(Fbeta, recall, precision, 2)
        F3 = MetricsLambda(Fbeta, recall, precision, 3)
        F4 = MetricsLambda(Fbeta, recall, precision, 4)

    When check if the metric is attached, if one of its dependency
    metrics is detached, the metric is considered detached too.

    .. code-block:: python

        engine = ...
        precision = Precision(average=False)

        aP = precision.mean()

        aP.attach(engine, "aP")

        assert aP.is_attached(engine)
        # partially attached
        assert not precision.is_attached(engine)

        precision.detach(engine)

        assert not aP.is_attached(engine)
        # fully attached
        assert not precision.is_attached(engine)

    """

    def __init__(self, f: Callable, *args, **kwargs):
        sub_trigger_events = None
        # Triggering points of all sub-matrics must be the same
        for arg in itertools.chain(args, kwargs.values()):
            if not isinstance(arg, Metric):
                continue

            # Canonicalize events triggering points
            trigger_events = arg._trigger_events.copy()
            trigger_events.setdefault("update", trigger_events["completed"])
            # Compare triggering points
            if sub_trigger_events is None:
                sub_trigger_events = trigger_events
            else:
                if trigger_events!=sub_trigger_events:
                    raise ValueError("triggering points of all sub-metrics must be the same")

        self.function = f
        self.args = args
        self.kwargs = kwargs
        self.engine = None

        super(MetricsLambda, self).__init__(device="cpu")

    @reinit__is_reduced
    def reset(self) -> None:
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, Metric):
                i.reset()

    @reinit__is_reduced
    def update(self, output) -> None:
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        pass

    def compute(self) -> Any:
        materialized = [i.compute() if isinstance(i, Metric) else i for i in self.args]
        materialized_kwargs = {k: (v.compute() if isinstance(v, Metric) else v) for k, v in self.kwargs.items()}
        return self.function(*materialized, **materialized_kwargs)

    def _internal_attach(self, engine: Engine) -> None:
        self.engine = engine
        for index, metric in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if isinstance(metric, MetricsLambda):
                metric._internal_attach(engine)
            elif isinstance(metric, Metric):
                # Triggering points for events
                completed_event = self._trigger_events["completed"]
                update_event = self._trigger_events.get("update", completed_event)
                started_event = self._trigger_events.get("started")

                # NB : metrics is attached partially
                # We must not use is_attached() but rather if these events exist

                # Handle started event
                if started_event!=None and not engine.has_event_handler(metric.started, started_event):
                    engine.add_event_handler(started_event, metric.started)
                # Handle update event
                if not engine.has_event_handler(metric.on_update, update_event):
                    engine.add_event_handler(update_event, metric.on_update)

    def attach(self, engine: Engine, name: str) -> None:
        # recursively attach all its dependencies (partially)
        self._internal_attach(engine)

        # Triggering points for period completed event
        completed_event = self._trigger_events["completed"]
        # attach only handler when period is completed
        engine.add_event_handler(completed_event, self.completed, name)

    def detach(self, engine: Engine) -> None:
        # remove from engine
        super(MetricsLambda, self).detach(engine)
        self.engine = None

    def is_attached(self, engine: Engine) -> bool:
        # check recursively the dependencies
        return super(MetricsLambda, self).is_attached(engine) and self._internal_is_attached(engine)

    def _internal_is_attached(self, engine: Engine) -> bool:
        # if no engine, metrics is not attached
        if engine is None:
            return False
        # check recursively if metrics are attached
        is_detached = False
        for metric in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(metric, MetricsLambda):
                if not metric._internal_is_attached(engine):
                    is_detached = True
            elif isinstance(metric, Metric):
                # Triggering points for events
                completed_event = self._trigger_events["completed"]
                update_event = self._trigger_events.get("update", completed_event)
                started_event = self._trigger_events.get("started")

                if started_event!=None and not engine.has_event_handler(metric.started, started_event):
                    is_detached = True
                if not engine.has_event_handler(metric.on_update, update_event):
                    is_detached = True
        return not is_detached
