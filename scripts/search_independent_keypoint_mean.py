#!/usr/bin/env python3
"""Best-first full-solve search ordered by mean of every observed cost.

The rolling mean of best three of four gates eligible work at 2 by default.
Blocked choices are deferred until eligible work is exhausted, then retried at
the smallest necessary one-window allowance. No trimming of the queue priority,
beam, time budget or expansion budget. The CLI caches the next best batch in
RAM, with lossless disk overflow, buffered writes and immutable shared paths.
At equal frame/cube/recent-score/update context, lower cumulative cost dominates.
This is an observation-fit heuristic, not a remaining-distance A* heuristic.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sqlite3
import sys
import time
import zlib
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mean_search_replay", ROOT / "scripts/search_independent_keypoint_full_solve.py")
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def format_console_progress(item, console_timezone="America/Chicago", now=None):
    now = datetime.now(timezone.utc) if now is None else now
    stamp = now.astimezone(ZoneInfo(console_timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")
    mismatch = item.get('current_sticker_mismatch_count')
    suffix = f' | mismatched stickers: {mismatch}/54' if mismatch is not None else ''
    if item.get('current_sticker_history_penalty') is not None:
        suffix += f" | history penalty: {item['current_sticker_history_penalty']:.3f}"
    if item.get('current_queue_score') is not None:
        suffix += f" | queue score: {item['current_queue_score']:.3f}"
    if item.get('soft_window_protected_frame') is not None:
        suffix += f" | protected through frame: {item['soft_window_protected_frame']}"
    if item.get('current_soft_window_penalty') is not None:
        suffix += f" | old-fork penalty: {item['current_soft_window_penalty']:.3f}"
    if item.get('color_confusion_pruning'):
        suffix += (f" | forbidden fits: {item.get('confusion_pruned_fits', 0):,}"
                   f" | color dead ends: {item.get('confusion_dead_end_states', 0):,}")
    if item.get('require_layer_activity'):
        suffix += f" | activity pruned: {item.get('activity_pruned', 0):,}"
    if item.get('soft_top_k'):
        suffix += f" | outside top {item['soft_top_k']}: {item.get('outside_top_k_offered', 0):,}"
    return (f"[{stamp}] explored states: {item['evaluated_states']:,} | "
            f"furthest branch: {item['furthest_video_frame']}/{item['total_video_frames']}{suffix}")


@dataclass
class MeanNode:
    depth: int
    state: object
    total_cost: float
    update: tuple
    path_id: int = 0
    recent: tuple | bytes = ()
    priority: float | None = None
    secondary_priority: float = 0.

    @property
    def mean(self):
        return self.total_cost / self.depth if self.depth else 0.

    @property
    def queue_priority(self):
        return self.mean if self.priority is None else self.priority


class MeanQueue:
    """Exact on-disk pending queue, best-cost table, and immutable shared paths."""

    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS paths (
                id INTEGER PRIMARY KEY, parent INTEGER NOT NULL, step BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS contexts (
                key BLOB PRIMARY KEY, total REAL NOT NULL, priority REAL NOT NULL,
                depth INTEGER NOT NULL, serial INTEGER NOT NULL,
                pending INTEGER NOT NULL, payload BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS deferred (
                key BLOB PRIMARY KEY, required REAL NOT NULL, total REAL NOT NULL,
                priority REAL NOT NULL, depth INTEGER NOT NULL, serial INTEGER NOT NULL,
                payload BLOB NOT NULL) WITHOUT ROWID;
        """)
        # Zero secondary priority preserves the ordering of older databases.
        for table in ('contexts', 'deferred'):
            columns = {row[1] for row in self.connection.execute(f'PRAGMA table_info({table})')}
            if 'secondary' not in columns:
                self.connection.execute(f'ALTER TABLE {table} ADD COLUMN secondary REAL NOT NULL DEFAULT 0')
        self.connection.executescript("""
            CREATE INDEX IF NOT EXISTS mean_pending_rank
                ON contexts(priority, secondary, depth DESC, serial) WHERE pending=1;
            CREATE INDEX IF NOT EXISTS mean_deferred_rank
                ON deferred(required, priority, secondary, depth DESC, serial);
            DROP INDEX IF EXISTS mean_pending_priority;
            DROP INDEX IF EXISTS mean_deferred_priority;
        """)
        self.pending_count = self.connection.execute(
            "SELECT count(*) FROM contexts WHERE pending=1").fetchone()[0]
        self.context_count = self.connection.execute("SELECT count(*) FROM contexts").fetchone()[0]
        self.deferred_count = self.connection.execute("SELECT count(*) FROM deferred").fetchone()[0]
        self.serial = max(self.connection.execute("SELECT coalesce(max(serial),0) FROM contexts").fetchone()[0],
                          self.connection.execute("SELECT coalesce(max(serial),0) FROM deferred").fetchone()[0])
        self.operations = 0

    @staticmethod
    def pack(value):
        return zlib.compress(pickle.dumps(value, protocol=5), level=1)

    @staticmethod
    def unpack(value):
        return pickle.loads(zlib.decompress(value))

    def tick(self):
        self.operations += 1
        if self.operations % 500 == 0:
            self.connection.commit()

    def offer(self, context, node, encoded_state, step=None, *, preceding_steps=()):
        previous = self.connection.execute(
            "SELECT total,pending FROM contexts WHERE key=?", (context,)).fetchone()
        if previous is not None and previous[0] <= node.total_cost:
            return None
        path_id = node.path_id
        for preceding in preceding_steps:
            path_id = self.connection.execute("INSERT INTO paths(parent,step) VALUES (?,?)",
                (path_id, self.pack(preceding))).lastrowid
        if step is not None:
            path_id = self.connection.execute("INSERT INTO paths(parent,step) VALUES (?,?)",
                (path_id, self.pack(step))).lastrowid
        self.serial += 1
        payload = self.pack((encoded_state, node.update, path_id, node.recent))
        self.connection.execute("""INSERT INTO contexts(key,total,priority,depth,serial,pending,payload,secondary)
            VALUES (?,?,?,?,?,1,?,?)
            ON CONFLICT(key) DO UPDATE SET total=excluded.total, priority=excluded.priority,
            depth=excluded.depth, serial=excluded.serial, pending=1, payload=excluded.payload,secondary=excluded.secondary""",
            (context, node.total_cost, node.queue_priority, node.depth, self.serial, payload,node.secondary_priority))
        if previous is None:
            self.context_count += 1
            self.pending_count += 1
            status = "new"
        elif previous[1]:
            status = "improved_pending"
        else:
            self.pending_count += 1
            status = "reopened"
        self.tick()
        return status

    def pop(self, deserialize_state):
        row = self.connection.execute("""SELECT key,depth,total,payload,priority,secondary FROM contexts
            WHERE pending=1 ORDER BY priority,secondary,depth DESC,serial LIMIT 1""").fetchone()
        if row is None:
            return None
        context, depth, total, payload, priority, secondary = row
        encoded, update, path_id, recent = self.unpack(payload)
        self.connection.execute("UPDATE contexts SET pending=0 WHERE key=?", (context,))
        self.pending_count -= 1
        self.tick()
        return MeanNode(depth, deserialize_state(encoded), total, update, path_id, recent, priority,secondary)

    def defer(self, key, required, node, encoded_state, step):
        previous = self.connection.execute("SELECT total FROM deferred WHERE key=?",(key,)).fetchone()
        if previous is not None and previous[0] <= node.total_cost:
            return False
        path_id = self.connection.execute("INSERT INTO paths(parent,step) VALUES (?,?)",
            (node.path_id,self.pack(step))).lastrowid
        self.serial += 1
        payload = self.pack((encoded_state,node.update,path_id,node.recent))
        self.connection.execute("""INSERT INTO deferred(key,required,total,priority,depth,serial,payload,secondary)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET total=excluded.total,priority=excluded.priority,
            serial=excluded.serial,payload=excluded.payload,secondary=excluded.secondary""",
            (key,required,node.total_cost,node.queue_priority,node.depth,self.serial,payload,node.secondary_priority))
        if previous is None:
            self.deferred_count += 1
        self.tick()
        return True

    def pop_deferred(self, deserialize_state):
        row = self.connection.execute("""SELECT key,required,depth,total,payload,priority,secondary FROM deferred
            ORDER BY required,priority,secondary,depth DESC,serial LIMIT 1""").fetchone()
        key,required,depth,total,payload,priority,secondary = row
        self.connection.execute("DELETE FROM deferred WHERE key=?",(key,))
        self.deferred_count -= 1
        self.tick()
        encoded,update,path_id,recent = self.unpack(payload)
        return required,MeanNode(depth,deserialize_state(encoded),total,update,path_id,recent,priority,secondary)

    def minimum_deferred(self):
        row = self.connection.execute("""SELECT required FROM deferred
            ORDER BY required,priority,secondary,depth DESC,serial LIMIT 1""").fetchone()
        return row[0] if row else None

    def minimum(self):
        row = self.connection.execute("""SELECT priority FROM contexts WHERE pending=1
            ORDER BY priority,secondary,depth DESC,serial LIMIT 1""").fetchone()
        return row[0] if row else None

    def minimum_secondary(self):
        row = self.connection.execute('SELECT secondary FROM contexts WHERE pending=1 '
            'ORDER BY priority,secondary,depth DESC,serial LIMIT 1').fetchone()
        return row[0] if row else None

    def minimum_frame_score(self):
        row = self.connection.execute('SELECT payload FROM contexts WHERE pending=1 '
            'ORDER BY priority,secondary,depth DESC,serial LIMIT 1').fetchone()
        return self.last_step(self.unpack(row[0])[2]).get('state_score') if row else None

    def steps(self, path_id):
        result = []
        while path_id:
            path_id, value = self.connection.execute(
                "SELECT parent,step FROM paths WHERE id=?", (path_id,)).fetchone()
            result.append(self.unpack(value))
        return list(reversed(result))

    def last_step(self, path_id):
        """Read the preceding frame's evidence without reconstructing its path."""
        if not path_id:
            return {}
        row = self.connection.execute('SELECT step FROM paths WHERE id=?', (path_id,)).fetchone()
        return self.unpack(row[0])

    def close(self):
        self.connection.commit()
        self.connection.close()


@dataclass(slots=True)
class _BufferedContext:
    total: float
    priority: float
    depth: int
    serial: int
    pending: int
    encoded: object
    update: tuple
    path_id: int
    recent: tuple | bytes
    secondary_priority: float = 0.

    @property
    def rank(self):
        return self.priority, self.secondary_priority, -self.depth, self.serial


class BufferedMeanQueue(MeanQueue):
    """Exact best-first RAM batches backed by the same persistent SQLite format.

    The resident heap contains all pending entries at or below its rank boundary.
    New better entries compete immediately. Overflow spills the worst residents
    and tightens the boundary; draining RAM flushes writes before loading the
    next global disk batch. Cache sizes never cap or discard search states.
    With deduplicate=False, insertion ids identify independent paths and no
    dominance lookup is performed; transition-score memoization is unchanged.
    """

    def __init__(self, path=":memory:", *, frontier_cache_size=4096,
                 cost_cache_size=131072, write_batch_size=4096,
                 flush_seconds=2., sqlite_cache_mib=64, deduplicate=True):
        if (frontier_cache_size < 1 or cost_cache_size < 0 or write_batch_size < 1
                or not math.isfinite(flush_seconds) or flush_seconds <= 0 or sqlite_cache_mib < 1):
            raise ValueError("Invalid queue cache or flush settings")
        super().__init__(path)
        self.connection.execute(f"PRAGMA cache_size=-{int(sqlite_cache_mib)*1024}")
        self.frontier_cache_size = frontier_cache_size
        self.deduplicate = deduplicate
        self.cost_cache_size = cost_cache_size if deduplicate else 0
        self.write_batch_size = write_batch_size
        self.flush_seconds = flush_seconds
        self.heap = []
        self.resident = {}
        self.boundary = (float("-inf"), 0, 0, 0) if self.pending_count else None
        self.dirty = {}
        self.dirty_deferred = {}
        self.path_buffer = {}
        self.cost_cache = OrderedDict()
        self.deferred_cache = OrderedDict()
        self.path_serial = self.connection.execute("SELECT coalesce(max(id),0) FROM paths").fetchone()[0]
        self.last_flush = time.monotonic()
        self.metrics = dict(cost_cache_hits=0, cost_disk_lookups=0, frontier_refills=0,
                            frontier_spilled=0, flushes=0, context_rows_written=0)

    def _remember(self, cache, key, value):
        if not self.cost_cache_size:
            return
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cost_cache_size:
            cache.popitem(last=False)

    def _previous(self, key, *, deferred=False):
        if not self.deduplicate:
            return None
        if deferred:
            row = self.dirty_deferred.get(key)
            if row is not None:
                self.metrics["cost_cache_hits"] += 1
                return row[1].total, 1
            cache, table = self.deferred_cache, "deferred"
        else:
            row = self.dirty.get(key) or self.resident.get(key)
            if row is not None:
                self.metrics["cost_cache_hits"] += 1
                return row.total, row.pending
            cache, table = self.cost_cache, "contexts"
        if key in cache:
            self.metrics["cost_cache_hits"] += 1
            cache.move_to_end(key)
            return cache[key]
        self.metrics["cost_disk_lookups"] += 1
        fields = "total" if deferred else "total,pending"
        value = self.connection.execute(f"SELECT {fields} FROM {table} WHERE key=?", (key,)).fetchone()
        if deferred and value is not None:
            value = value[0], 1
        self._remember(cache, key, value)
        return value

    def _path(self, parent, step):
        if step is None:
            return parent
        self.path_serial += 1
        self.path_buffer[self.path_serial] = parent, dict(step)
        return self.path_serial

    def last_step(self, path_id):
        buffered = self.path_buffer.get(path_id)
        return buffered[1] if buffered is not None else super().last_step(path_id)

    def _row(self, node, encoded_state, step, preceding_steps=()):
        self.serial += 1
        path_id = node.path_id
        for preceding in preceding_steps:
            path_id = self._path(path_id, preceding)
        return _BufferedContext(node.total_cost, node.queue_priority, node.depth, self.serial, 1,
            encoded_state, node.update, self._path(path_id, step), node.recent,node.secondary_priority)

    def _admit(self, context, row):
        self.resident.pop(context, None)
        if self.boundary is None or row.rank <= self.boundary:
            self.resident[context] = row
            heapq.heappush(self.heap, (*row.rank, context))
        if len(self.resident) > 2*self.frontier_cache_size:
            ordered = sorted(self.resident.items(), key=lambda item: item[1].rank)
            self.metrics["frontier_spilled"] += len(ordered)-self.frontier_cache_size
            self.resident = dict(ordered[:self.frontier_cache_size])
            self.boundary = ordered[self.frontier_cache_size-1][1].rank
            self.heap = [(*value.rank, key) for key, value in self.resident.items()]
            heapq.heapify(self.heap)
        elif len(self.heap) > 3*self.frontier_cache_size:
            self.heap = [(*value.rank, key) for key, value in self.resident.items()]
            heapq.heapify(self.heap)

    def _maybe_flush(self):
        if (len(self.dirty)+len(self.dirty_deferred) >= self.write_batch_size
                or len(self.path_buffer) >= 2*self.write_batch_size
                or time.monotonic()-self.last_flush >= self.flush_seconds):
            self.flush()

    def offer(self, context, node, encoded_state, step=None, *, preceding_steps=()):
        if not self.deduplicate:
            context = (self.serial+1).to_bytes(8, "big")
        previous = self._previous(context)
        if previous is not None and previous[0] <= node.total_cost:
            self._maybe_flush()
            return None
        row = self._row(node, encoded_state, step, preceding_steps)
        self.dirty[context] = row
        self._remember(self.cost_cache, context, (row.total, 1))
        self._admit(context, row)
        if previous is None:
            self.context_count += 1
            self.pending_count += 1
            status = "new"
        elif previous[1]:
            status = "improved_pending"
        else:
            self.pending_count += 1
            status = "reopened"
        self._maybe_flush()
        return status

    def flush(self):
        # Path ids are assigned on acceptance, so buffered parent/child records
        # and context updates can be committed together without changing ties.
        if self.path_buffer:
            self.connection.executemany("INSERT INTO paths(id,parent,step) VALUES (?,?,?)",
                ((index, parent, self.pack(step)) for index, (parent, step) in self.path_buffer.items()))
        if self.dirty:
            self.connection.executemany("""INSERT INTO contexts(key,total,priority,depth,serial,pending,payload,secondary)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET total=excluded.total, priority=excluded.priority,
                depth=excluded.depth, serial=excluded.serial, pending=excluded.pending,
                payload=excluded.payload,secondary=excluded.secondary""",
                ((key, row.total, row.priority, row.depth, row.serial, row.pending,
                  self.pack((row.encoded, row.update, row.path_id, row.recent)),row.secondary_priority)
                 for key, row in self.dirty.items()))
        if self.dirty_deferred:
            self.connection.executemany("""INSERT INTO deferred(key,required,total,priority,depth,serial,payload,secondary)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET total=excluded.total, priority=excluded.priority,
                serial=excluded.serial, payload=excluded.payload,secondary=excluded.secondary""",
                ((key, required, row.total, row.priority, row.depth, row.serial,
                  self.pack((row.encoded, row.update, row.path_id, row.recent)),row.secondary_priority)
                 for key, (required, row) in self.dirty_deferred.items()))
        self.connection.commit()
        self.metrics["flushes"] += 1
        self.metrics["context_rows_written"] += len(self.dirty)
        self.path_buffer.clear()
        self.dirty.clear()
        self.dirty_deferred.clear()
        self.last_flush = time.monotonic()

    def _refill(self):
        self.flush()
        rows = self.connection.execute("""SELECT key,total,priority,depth,serial,payload,secondary FROM contexts
            WHERE pending=1 ORDER BY priority,secondary,depth DESC,serial LIMIT ?""",
            (self.frontier_cache_size,)).fetchall()
        if not rows:
            if self.pending_count:
                raise RuntimeError("Pending count disagrees with persisted queue")
            return
        self.metrics["frontier_refills"] += 1
        self.resident.clear()
        self.heap.clear()
        for key, total, priority, depth, serial, payload, secondary in rows:
            encoded, update, path_id, recent = self.unpack(payload)
            row = _BufferedContext(total, priority, depth, serial, 1, encoded, update, path_id, recent,secondary)
            self.resident[key] = row
            self.heap.append((*row.rank, key))
            self._remember(self.cost_cache, key, (total, 1))
        self.boundary = self.heap[-1][:4]
        heapq.heapify(self.heap)

    def _peek(self):
        while self.heap:
            entry = self.heap[0]
            row = self.resident.get(entry[4])
            if row is not None and row.serial == entry[3]:
                return entry
            heapq.heappop(self.heap)
        if self.pending_count:
            self._refill()
            return self.heap[0]
        return None

    def pop(self, deserialize_state):
        entry = self._peek()
        if entry is None:
            return None
        heapq.heappop(self.heap)
        context = entry[4]
        row = self.resident.pop(context)
        row.pending = 0
        self.dirty[context] = row
        self._remember(self.cost_cache, context, (row.total, 0))
        self.pending_count -= 1
        self._maybe_flush()
        return MeanNode(row.depth, deserialize_state(row.encoded), row.total, row.update, row.path_id,
                        row.recent, row.priority,row.secondary_priority)

    def defer(self, key, required, node, encoded_state, step):
        if not self.deduplicate:
            key = (self.serial+1).to_bytes(8, "big")
        previous = self._previous(key, deferred=True)
        if previous is not None and previous[0] <= node.total_cost:
            self._maybe_flush()
            return False
        row = self._row(node, encoded_state, step)
        self.dirty_deferred[key] = required, row
        self._remember(self.deferred_cache, key, (row.total, 1))
        if previous is None:
            self.deferred_count += 1
        self._maybe_flush()
        return True

    def pop_deferred(self, deserialize_state):
        if self.dirty_deferred:
            self.flush()
        row = self.connection.execute("""SELECT key,required,depth,total,payload,priority,secondary FROM deferred
            ORDER BY required,priority,secondary,depth DESC,serial LIMIT 1""").fetchone()
        key, required, depth, total, payload, priority, secondary = row
        self.connection.execute("DELETE FROM deferred WHERE key=?", (key,))
        self.deferred_cache.pop(key, None)
        self.deferred_count -= 1
        encoded, update, path_id, recent = self.unpack(payload)
        return required, MeanNode(depth, deserialize_state(encoded), total, update, path_id, recent, priority,secondary)

    def minimum(self):
        entry = self._peek()
        return entry[0] if entry is not None else None

    def minimum_secondary(self):
        entry = self._peek()
        return entry[1] if entry is not None else None

    def minimum_frame_score(self):
        entry = self._peek()
        return self.last_step(self.resident[entry[4]].path_id).get('state_score') if entry is not None else None

    def minimum_deferred(self):
        value = super().minimum_deferred()
        dirty = min((required for required, _row in self.dirty_deferred.values()), default=None)
        return dirty if value is None else value if dirty is None else min(value, dirty)

    def steps(self, path_id):
        result = []
        while path_id:
            buffered = self.path_buffer.get(path_id)
            if buffered is not None:
                path_id, step = buffered
            else:
                path_id, value = self.connection.execute(
                    "SELECT parent,step FROM paths WHERE id=?", (path_id,)).fetchone()
                step = self.unpack(value)
            result.append(step)
        return list(reversed(result))

    def cache_stats(self):
        return dict(queue_cache_metrics=dict(self.metrics), queue_resident=len(self.resident),
                    queue_buffered_contexts=len(self.dirty), queue_buffered_paths=len(self.path_buffer))

    def close(self):
        self.flush()
        self.connection.close()


def mean_full_solve(initial_state, frames, transitions, solved, state_key, *,
                    threshold=2., min_update_gap=1, queue=None, serialize_state=pickle.dumps,
                    deserialize_state=pickle.loads, progress=None, progress_seconds=10.):
    """Pop the lowest whole-path mean; ties prefer deeper paths, then insertion order.

    Equal frame/cube/recent-score/move-spacing contexts have identical future actions.
    They also have equal observation counts: a lower sum therefore dominates for
    every shared future continuation. Retain that sum and reopen on improvement.
    Rolling failures defer rather than permanently discard finite-cost choices.
    """
    if queue is None:
        store = MeanQueue()
        try:
            return mean_full_solve(initial_state, frames, transitions, solved, state_key,
                threshold=threshold,min_update_gap=min_update_gap, queue=store, serialize_state=serialize_state,
                deserialize_state=deserialize_state, progress=progress,
                progress_seconds=progress_seconds)
        finally:
            store.close()
    if not frames or any(b != a+1 for a,b in zip(frames, frames[1:])):
        raise ValueError("Provide a nonempty, consecutive source-frame interval")
    if not math.isfinite(threshold) or min_update_gap < 1 or progress_seconds <= 0:
        raise ValueError("Threshold must be finite; gap and progress interval must be positive")
    if queue.context_count or queue.deferred_count:
        raise ValueError("Start in a fresh search database")
    stats = dict(evaluated_states=0, attempted_transitions=0, invalid_pruned=0,
        spacing_pruned_before_scoring=0,
        spacing_pruned=0, dominated_pruned=0, reopened_states=0, improved_pending=0,
        unsolved_endpoints=0, furthest_frames=0, furthest_frame_index=None,
        current_frames=0, current_frame_index=None, current_path_mean=0.,
        peak_pending=0, error_gate_enabled=True, normal_threshold=threshold,
        threshold_deferred=0,deferred_resumed=0,deferred_dominated=0,
        largest_temporary_threshold=threshold)
    started = time.monotonic()
    last_heartbeat = started

    def report(event, **extra):
        reported_at = time.monotonic()
        if progress:
            best_pending = queue.minimum()
            next_deferred = queue.minimum_deferred()
            cache_stats = queue.cache_stats() if hasattr(queue, "cache_stats") else {}
            progress(dict(event=event, elapsed_seconds=reported_at-started, **stats,
                queue_pending=queue.pending_count, stored_contexts=queue.context_count,
                best_pending_mean=best_pending,deferred_pending=queue.deferred_count,
                next_deferred_threshold=next_deferred, **cache_stats, **extra))

    def context(node):
        if not getattr(queue, "deduplicate", True):
            return None
        return pickle.dumps(("gated-full-mean-v1", threshold,node.depth,state_key(node.state),node.recent,
            node.update if min_update_gap > 1 else None), protocol=5)

    def outcome(status, node=None):
        report(status)
        steps = queue.steps(node.path_id) if node else []
        return dict(status=status, elapsed_seconds=time.monotonic()-started, **stats,
            queue_pending=queue.pending_count, stored_contexts=queue.context_count,
            deferred_pending=queue.deferred_count,
            steps=steps, solved_frame_index=frames[node.depth-1] if node and node.depth else None)

    if solved(initial_state):
        return outcome("already_solved")
    root = MeanNode(0, initial_state, 0., (None, None, 0))
    queue.offer(context(root), root, serialize_state(initial_state))
    stats["peak_pending"] = 1
    report("started")
    while queue.pending_count or queue.deferred_count:
        if time.monotonic()-last_heartbeat >= progress_seconds:
            last_heartbeat = time.monotonic()
            report("progress")
        if not queue.pending_count:
            required,child = queue.pop_deferred(deserialize_state)
            status = queue.offer(context(child),child,serialize_state(child.state))
            if status is None:
                stats["deferred_dominated"] += 1
                continue
            if status == "reopened":
                stats["reopened_states"] += 1
            elif status == "improved_pending":
                stats["improved_pending"] += 1
            stats["deferred_resumed"] += 1
            stats["largest_temporary_threshold"] = max(stats["largest_temporary_threshold"],required)
            stats["peak_pending"] = max(stats["peak_pending"],queue.pending_count)
            report("temporary_allowance",source_frame=frames[child.depth-1],
                   temporary_threshold=required,return_threshold_next_frame=threshold)
        node = queue.pop(deserialize_state)
        frame = frames[node.depth-1] if node.depth else None
        stats.update(current_frames=node.depth, current_frame_index=frame,
                     current_path_mean=node.mean)
        if node.depth > stats["furthest_frames"]:
            stats.update(furthest_frames=node.depth, furthest_frame_index=frame)
            steps = queue.steps(node.path_id)
            report("furthest", last_score=steps[-1]["state_score"],
                furthest_path_mean=node.mean,
                frontier_moves=[dict(frame_index=s["frame_index"],state_moves=s["state_moves"])
                                for s in steps if s["state_moves"]])
        if solved(node.state):
            return outcome("solved", node)
        if node.depth == len(frames):
            stats["unsolved_endpoints"] += 1
            continue
        stats["evaluated_states"] += 1
        frame = frames[node.depth]
        parent_context = context(node)
        choices, skipped = replay.transitions_for_update_context(transitions, frame, node.state,
            node.update, min_update_gap)
        stats["spacing_pruned"] += skipped
        stats["spacing_pruned_before_scoring"] += skipped
        for choice_index,choice in enumerate(sorted(choices,key=lambda c:c.rank)):
            stats["attempted_transitions"] += 1
            total = node.total_cost+choice.score
            if not math.isfinite(choice.score) or not math.isfinite(total):
                stats["invalid_pruned"] += 1
                continue
            last, key, streak = node.update
            if choice.payload.get("state_moves"):
                consecutive = last is not None and frame-last == 1
                continuation = consecutive and streak<2 and choice.update_key is not None and choice.update_key==key
                if last is not None and frame-last<min_update_gap and not continuation:
                    stats["spacing_pruned"] += 1
                    continue
                streak = streak+1 if consecutive and choice.update_key==key else 1
                last, key = frame, choice.update_key
            recent = (*node.recent[-3:],choice.score)
            rolling = replay.best_three_mean_four(recent)
            child = MeanNode(node.depth+1,choice.state,total,(last,key,streak),node.path_id,recent[-3:])
            keep = ("trial_id","state_moves","color_error","keypoint_mean_error",
                    "heatmap_mean_error","geometry_option_index")
            step = {k:choice.payload[k] for k in keep if k in choice.payload}
            step.update(frame_index=frame, state_score=choice.score,
                        path_total_cost=total,whole_path_mean=child.mean,best_three_mean_four=rolling,
                        normal_threshold=threshold,applied_threshold=max(threshold,rolling) if rolling is not None else threshold,
                        temporary_allowance=rolling is not None and rolling>threshold)
            if rolling is not None and rolling>threshold:
                # Parent context fixes the required rolling allowance, even if
                # another history reaches the same child with a different one.
                deferred_key = (pickle.dumps((parent_context,choice_index),protocol=5)
                                if getattr(queue, "deduplicate", True) else None)
                queue.defer(deferred_key,rolling,child,serialize_state(choice.state),step)
                stats["threshold_deferred"] += 1
                continue
            status = queue.offer(context(child), child, serialize_state(choice.state), step)
            if status is None:
                stats["dominated_pruned"] += 1
            elif status == "reopened":
                stats["reopened_states"] += 1
            elif status == "improved_pending":
                stats["improved_pending"] += 1
            stats["peak_pending"] = max(stats["peak_pending"], queue.pending_count)
    return outcome("exhausted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("--initial-scramble",required=True)
    parser.add_argument("--start-frame",type=int,required=True)
    parser.add_argument("--end-frame",type=int,required=True)
    parser.add_argument("--video-frame-count",type=int)
    parser.add_argument("--fps",type=float,default=60.)
    parser.add_argument("--threshold",type=float,default=2.)
    parser.add_argument("--min-state-update-gap",type=int)
    parser.add_argument("--device",choices=("cpu","mps"),default="cpu")
    parser.add_argument("--scorer",choices=("auto","prepared","legacy"),default="auto",
                        help="Auto uses prepared base-pose mappings on CPU; MPS uses legacy scoring")
    parser.add_argument("--queue-backend",choices=("buffered","sqlite"),default="buffered",
                        help="Exact RAM-frontier cache with disk overflow, or original SQLite queue")
    parser.add_argument("--no-deduplicate",action="store_true",
                        help="Retain every path with a unique queue id; skip dominance checks (buffered only)")
    parser.add_argument("--frontier-cache-size",type=int,default=4096,
                        help="RAM refill batch size; overflow stays on disk, never pruned")
    parser.add_argument("--cost-cache-size",type=int,default=131072,
                        help="Recent exact best-cost entries cached per table; eviction never prunes")
    parser.add_argument("--queue-flush-seconds",type=float,default=2.)
    parser.add_argument("--sqlite-cache-mib",type=int,default=64,
                        help="SQLite page-cache allowance for the buffered backend")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-seconds",type=float,default=10.)
    parser.add_argument("--console-timezone",default="America/Chicago")
    args = parser.parse_args()
    if args.no_deduplicate and args.queue_backend != "buffered":
        parser.error("--no-deduplicate requires --queue-backend buffered")
    if args.queue_backend == "buffered" and (
            args.frontier_cache_size < 1 or args.cost_cache_size < 0 or args.sqlite_cache_mib < 1
            or not math.isfinite(args.queue_flush_seconds) or args.queue_flush_seconds <= 0):
        parser.error("Queue caches must be positive (cost cache may be zero), with a finite positive flush interval")
    try:
        ZoneInfo(args.console_timezone)
    except (KeyError, ValueError):
        parser.error("Unknown console timezone")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any((args.output_dir/name).exists() for name in ("progress.jsonl","search.sqlite3","config.json")):
        raise SystemExit("Use a new output directory; previous searches are preserved")
    m = replay.load_renderer()
    import torch
    torch.set_num_threads(2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; use cpu")
    records = {r["frame_index"]:r for r in map(json.loads,args.log.open())}
    if not records:
        raise SystemExit("Source log contains no frames")
    video_count = args.video_frame_count if args.video_frame_count is not None else max(records)+1
    if video_count < max(records)+1:
        raise SystemExit("Whole video length cannot be smaller than source-log extent")
    frames = tuple(range(args.start_frame,args.end_frame+1))
    if not frames:
        raise SystemExit("End frame precedes start frame")
    scorer_mode = ("prepared" if args.device == "cpu" else "legacy") if args.scorer == "auto" else args.scorer
    if scorer_mode == "prepared" and args.device != "cpu":
        raise SystemExit("Prepared replay scoring requires --device cpu")
    scorer_class = replay.PreparedReplayScorer if scorer_mode == "prepared" else replay.ReplayScorer
    scorer = scorer_class(m,records,torch.device(args.device))
    for frame in frames:
        scorer.prepare(frame)
    initial = m.CubeGeometryState.identity()
    scrambled = m.CubeState.from_scramble(args.initial_scramble)
    colors = {label:scrambled.colors[position] for position,label in initial.facelets.items()}
    initial = scorer.state_from_key(scorer.key(initial))
    solved = scorer.solved_predicate(colors)
    gap = m.turn_state_update_gap_frames(args.fps) if args.min_state_update_gap is None else args.min_state_update_gap
    config = dict(algorithm="best_first_whole_path_mean_rolling_gate",log=str(args.log.resolve()),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        search_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_scramble=args.initial_scramble,start_frame=args.start_frame,end_frame=args.end_frame,
        priority_statistic="untrimmed_mean_of_all_observed_combined_costs_since_solve_start",
        priority_ties="greater depth, then insertion order",
        error_gate="mean_best_three_of_last_four_combined_costs",threshold=args.threshold,
        threshold_policy="defer_until_eligible_queue_empty_then_minimum_required_one_window",
        threshold_returns_to_normal="next frame",trim_fraction=0.,frame_or_state_budget=None,time_budget=None,
        beam_width=None,state_change_limit=None,video_frame_count=video_count,
        video_frame_count_source="explicit" if args.video_frame_count is not None else "source_log_extent",
        console_frame_numbering="one-based whole-video position; JSON source indices are zero-based",
        console_interval_seconds=args.progress_seconds,console_timezone=args.console_timezone,
        min_state_update_gap_frames=gap,fps=args.fps,device=args.device,
        scorer=scorer_mode,projection_offsets="state permutations" if scorer_mode == "prepared" else "explicit projections",
        state_representation="54-byte permutation" if scorer_mode == "prepared" else "facelet dictionary",
        move_spacing_filter="before scoring" if scorer_mode == "prepared" else "after scoring",
        queue_backend=args.queue_backend,
        queue_frontier_cache_size=args.frontier_cache_size if args.queue_backend == "buffered" else None,
        queue_cost_cache_size=(0 if args.no_deduplicate else args.cost_cache_size) if args.queue_backend == "buffered" else None,
        queue_flush_seconds=args.queue_flush_seconds if args.queue_backend == "buffered" else None,
        sqlite_cache_mib=args.sqlite_cache_mib if args.queue_backend == "buffered" else None,
        queue_cache_policy=(("lossless overflow; global priority; every path retained" if args.no_deduplicate
                             else "lossless overflow; identical global priority and dominance")
                            if args.queue_backend == "buffered" else None),
        scorer_script_sha256=hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        renderer_script_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        consecutive_update_exception="only second quarter-turn of same layer, width and direction",
        merge_key=None if args.no_deduplicate else "frame,physical_cube_state,last_three_numeric_scores,move_spacing_context",
        dominance_policy="disabled; retain every path" if args.no_deduplicate else "lowest cumulative cost; reopen expanded context on strict improvement",
        duplicate_detection=not args.no_deduplicate,
        queue_key_format="unique insertion id" if args.no_deduplicate else "exact observation/update context",
        geometry_scope="all retained source-log options and discrete color offsets",
        orientation_policy="24-way state-color rescue below 2/3 accuracy before move search",
        orientation_rescue_minimum_color_accuracy=2./3.,
        goal="physical cube colors solved in any whole-cube orientation",reference_solution_used=False)
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2))
    if args.queue_backend == "buffered":
        queue = BufferedMeanQueue(args.output_dir/"search.sqlite3",
            frontier_cache_size=args.frontier_cache_size,cost_cache_size=args.cost_cache_size,
            flush_seconds=args.queue_flush_seconds,sqlite_cache_mib=args.sqlite_cache_mib,
            deduplicate=not args.no_deduplicate)
    else:
        queue = MeanQueue(args.output_dir/"search.sqlite3")
    with (args.output_dir/"progress.jsonl").open("w",buffering=1) as log:
        latest = {}
        def progress(item):
            now = datetime.now(timezone.utc)
            position = item["furthest_frame_index"]+1 if item["furthest_frame_index"] is not None else 0
            item.update(furthest_video_frame=position,total_video_frames=video_count,
                        scoring_calls=scorer.scoring_calls,score_cache_hits=scorer.cache_hits,
                        timestamp_utc=now.isoformat())
            if scorer_mode == "prepared":
                item.update(color_configurations_scored=scorer.color_configurations_scored,
                            child_states_built=scorer.child_states_built)
            latest.clear()
            latest.update(item)
            log.write(json.dumps(item,allow_nan=False)+"\n")
            if item["event"] in {"started","progress","solved","already_solved","exhausted"}:
                print(format_console_progress(item,args.console_timezone,now),flush=True)
            temporary = args.output_dir/"status.tmp"
            temporary.write_text(json.dumps(item,indent=2))
            temporary.replace(args.output_dir/"status.json")
            if (args.output_dir/"STOP").exists():
                raise KeyboardInterrupt
        try:
            result = mean_full_solve(initial,frames,scorer,solved,scorer.key,
                threshold=args.threshold,min_update_gap=gap,queue=queue,serialize_state=scorer.key,
                deserialize_state=scorer.state_from_key,progress=progress,
                progress_seconds=args.progress_seconds)
            (args.output_dir/"result.json").write_text(json.dumps(dict(config=config,**result),indent=2))
        except KeyboardInterrupt:
            item = dict(latest,event="interrupted",status="interrupted")
            log.write(json.dumps(item)+"\n")
            (args.output_dir/"status.json").write_text(json.dumps(item,indent=2))
            (args.output_dir/"result.json").write_text(json.dumps(dict(config=config,**item),indent=2))
            print(format_console_progress(item,args.console_timezone)+" | stopped",flush=True)
        finally:
            queue.close()


if __name__ == "__main__":
    main()
