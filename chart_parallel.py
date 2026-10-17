"""Request-scoped speculative chart work for approved historical query rows."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from copy import deepcopy
import logging


_current_jobs = ContextVar("chart_jobs", default=None)
logger = logging.getLogger("datasage")


class ChartJobs:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="datasage-chart")
        self._latest = None

    def start(self, sql, rows, question, generate):
        # Freeze the inputs used by the worker so a later graph revision cannot
        # mutate them or accidentally match an obsolete chart.
        snapshot = deepcopy(rows)
        context = copy_context()
        future = self._executor.submit(context.run, generate, snapshot, question)
        self._latest = (sql, snapshot, future)

    def result(self, sql, rows):
        if self._latest is None or (sql, rows) != self._latest[:2]:
            return None
        try:
            return self._latest[2].result()
        except Exception:
            logger.exception("Speculative chart generation failed")
            return None, "Chart generation failed."

    def close(self):
        # Include even discarded SQL-revision chart calls in request telemetry.
        self._executor.shutdown(wait=True)


@contextmanager
def chart_request():
    jobs = ChartJobs()
    token = _current_jobs.set(jobs)
    try:
        yield jobs
    finally:
        try:
            jobs.close()
        finally:
            _current_jobs.reset(token)


def schedule_chart(sql, rows, question, generate):
    jobs = _current_jobs.get()
    if jobs is not None:
        jobs.start(sql, rows, question, generate)


def chart_result(sql, rows):
    jobs = _current_jobs.get()
    return jobs.result(sql, rows) if jobs is not None else None
