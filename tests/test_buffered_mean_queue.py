import importlib.util
from pathlib import Path
import pickle
import random
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("buffered_queue_tests",
    Path(__file__).resolve().parents[1]/"scripts/search_independent_keypoint_mean.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def node(state, total, depth=1, path=0):
    return search.MeanNode(depth, state, total, (None, None, 0), path, (float(total),))


def offer(q, key, total, depth=1, path=0):
    return q.offer(key.encode(), node(key, total, depth, path), pickle.dumps(key),
                   dict(state_moves=(key,), state_score=total))


def test_new_better_children_compete_with_disk_spill_immediately():
    q = search.BufferedMeanQueue(frontier_cache_size=1, cost_cache_size=0, write_batch_size=100)
    try:
        for key, total in [('a',10), ('b',20), ('c',30)]:
            offer(q,key,total)
        assert q.metrics['frontier_spilled'] == 2
        assert q.pop(pickle.loads).state == 'a'
        offer(q,'new',5)
        offer(q,'between',15)
        assert q.minimum() == 5
        assert [q.pop(pickle.loads).state for _ in range(4)] == ['new','between','b','c']
        assert q.pending_count == 0
        assert q.minimum() is None
    finally:q.close()


def test_improvements_invalidate_hot_entries_and_reopen_expanded_contexts():
    q = search.BufferedMeanQueue(frontier_cache_size=1, write_batch_size=100)
    try:
        offer(q,'a',10);offer(q,'b',20);offer(q,'c',30)
        assert offer(q,'b',4) == 'improved_pending'
        assert offer(q,'b',6) is None
        assert q.pop(pickle.loads).total_cost == 4
        assert offer(q,'b',3) == 'reopened'
        assert offer(q,'a',2) == 'improved_pending'
        assert [q.pop(pickle.loads).state for _ in range(3)] == ['a','b','c']
    finally:q.close()


def test_ties_keep_greater_depth_then_original_serial_after_spilling():
    q = search.BufferedMeanQueue(frontier_cache_size=1, write_batch_size=2)
    try:
        offer(q,'shallow',2,1)
        offer(q,'deep_first',8,4)
        offer(q,'deep_second',8,4)
        assert [q.pop(pickle.loads).state for _ in range(3)] == ['deep_first','deep_second','shallow']
    finally:q.close()


def test_buffered_paths_work_before_flush_and_reopen_in_original_queue(tmp_path):
    path = tmp_path/'queue.sqlite3'
    q = search.BufferedMeanQueue(path, frontier_cache_size=1, write_batch_size=100)
    offer(q,'parent',2)
    parent = q.pop(pickle.loads)
    q.flush()
    offer(q,'child',3,2,parent.path_id)
    child = q.pop(pickle.loads)
    assert [s['state_moves'] for s in q.steps(child.path_id)] == [('parent',),('child',)]
    offer(q,'remaining',4)
    q.close()
    old = search.MeanQueue(path)
    assert old.context_count == 3
    assert old.pending_count == 1
    assert old.pop(pickle.loads).state == 'remaining'
    old.close()


def test_existing_disk_minimum_is_not_skipped_by_new_worse_offer(tmp_path):
    path = tmp_path/'queue.sqlite3'
    old = search.MeanQueue(path)
    offer(old,'disk_best',1)
    old.close()
    q = search.BufferedMeanQueue(path, frontier_cache_size=1)
    try:
        offer(q,'new_worse',5)
        assert q.minimum() == 1
        assert q.pop(pickle.loads).state == 'disk_best'
        assert q.pop(pickle.loads).state == 'new_worse'
    finally:q.close()


@pytest.mark.parametrize('frontier_size,cost_size,batch', [(1,0,1),(2,3,7),(17,100,1000)])
def test_randomized_operations_match_original_queue(frontier_size,cost_size,batch):
    old = search.MeanQueue()
    new = search.BufferedMeanQueue(frontier_cache_size=frontier_size,
        cost_cache_size=cost_size, write_batch_size=batch)
    rng = random.Random(182)
    paths = [0]
    try:
        for _ in range(2500):
            action = rng.random()
            index = rng.randrange(300)
            key = str(index).encode()
            depth = index%8+1
            total = rng.randrange(-100,100)/3
            n = node(str(index), total, depth, rng.choice(paths))
            step = dict(state_moves=(str(index),), state_score=total)
            if action < .68:
                a = old.offer(key,n,pickle.dumps(n.state),step)
                b = new.offer(key,n,pickle.dumps(n.state),step)
                assert a == b
            elif action < .82:
                required = 2 + index%9/10
                assert old.defer(key,required,n,pickle.dumps(n.state),step) == new.defer(
                    key,required,n,pickle.dumps(n.state),step)
            elif action < .95 and old.pending_count:
                a,b = old.pop(pickle.loads),new.pop(pickle.loads)
                assert a == b
                assert old.steps(a.path_id) == new.steps(b.path_id)
                paths.append(a.path_id)
            elif old.deferred_count:
                a,b = old.pop_deferred(pickle.loads),new.pop_deferred(pickle.loads)
                assert a == b
                assert old.steps(a[1].path_id) == new.steps(b[1].path_id)
                paths.append(a[1].path_id)
            assert old.pending_count == new.pending_count
            assert old.context_count == new.context_count
            assert old.deferred_count == new.deferred_count
            assert old.minimum() == new.minimum()
            assert old.minimum_deferred() == new.minimum_deferred()
            assert len(new.resident) <= 2*frontier_size
        while old.pending_count:
            assert old.pop(pickle.loads) == new.pop(pickle.loads)
        while old.deferred_count:
            assert old.pop_deferred(pickle.loads) == new.pop_deferred(pickle.loads)
    finally:
        old.close();new.close()


@pytest.mark.parametrize('kwargs', [dict(frontier_cache_size=0), dict(cost_cache_size=-1),
    dict(write_batch_size=0), dict(flush_seconds=0), dict(sqlite_cache_mib=0)])
def test_invalid_cache_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):search.BufferedMeanQueue(**kwargs)


def test_no_deduplication_retains_equal_and_more_expensive_paths_without_cost_reads():
    q = search.BufferedMeanQueue(deduplicate=False, frontier_cache_size=1, write_batch_size=2)
    statements = []
    q.connection.set_trace_callback(statements.append)
    try:
        assert offer(q,'same',1) == 'new'
        assert offer(q,'same',1) == 'new'
        assert offer(q,'same',2) == 'new'
        assert q.context_count == q.pending_count == 3
        assert [q.pop(pickle.loads).total_cost for _ in range(3)] == [1,1,2]
        assert q.metrics['cost_disk_lookups'] == q.metrics['cost_cache_hits'] == 0
        assert not any(sql.startswith('SELECT total') for sql in statements)
        assert not q.cost_cache
    finally:q.close()


def test_no_deduplication_retains_separate_deferred_paths_and_path_records(tmp_path):
    path = tmp_path/'tree.sqlite3'
    q = search.BufferedMeanQueue(path, deduplicate=False, frontier_cache_size=1, write_batch_size=2)
    try:
        for index in range(3):
            n = node('same',index+1)
            assert q.defer(b'same',4-index,n,pickle.dumps(n.state),dict(state_moves=(str(index),)))
        assert q.deferred_count == 3
        seen = []
        while q.deferred_count:
            required,n = q.pop_deferred(pickle.loads)
            seen.append((required,n.total_cost,q.steps(n.path_id)[0]['state_moves']))
        assert seen == [(2,3,('2',)),(3,2,('1',)),(4,1,('0',))]
        assert q.metrics['cost_disk_lookups'] == 0
    finally:q.close()


def test_tree_search_reexpands_equivalent_contexts_instead_of_merging():
    def transitions(frame,state):
        if frame == 0:
            return tuple(search.replay.Transition('same',1,(1,),dict(state_moves=())) for _ in range(2))
        return (search.replay.Transition('end',1,(1,),dict(state_moves=())),)
    for deduplicate, expected_expansions, expected_contexts in [(True,2,3),(False,3,5)]:
        q = search.BufferedMeanQueue(deduplicate=deduplicate,frontier_cache_size=1)
        try:
            result = search.mean_full_solve('root',(0,1),transitions,lambda _:False,str,
                queue=q,serialize_state=pickle.dumps,deserialize_state=pickle.loads)
            assert result['status'] == 'exhausted'
            assert result['evaluated_states'] == expected_expansions
            assert result['stored_contexts'] == expected_contexts
            assert result['dominated_pruned'] == (1 if deduplicate else 0)
        finally:q.close()
