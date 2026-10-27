"""One bounded background lookup; current action validity is always rechecked."""
from concurrent.futures import ThreadPoolExecutor
import copy
import time

import cv2
import numpy as np

from .experience import _objects_compatible


class LatestMemory:
    def __init__(self, memory, max_age=.4):
        self.memory = memory
        self.max_age = float(max_age)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-memory')
        self.future = None
        self._active = None
        self.query = None
        self.completed = None
        self.last_submit = -1e9
        self.stats = {
            'submitted': 0, 'completed': 0, 'discarded': 0, 'errors': 0,
            'returned': 0, 'adopted': 0, 'revalidated': 0,
            'revalidation_attempts': 0,
            'last_search_ms': 0., 'last_revalidation_ms': 0.,
            'reject_reasons': {},
        }

    @staticmethod
    def _snapshot(frame, stage, context, now, episode):
        return {
            'frame': frame.copy(), 'stage': stage,
            'context': copy.deepcopy(context), 'time': float(now), 'episode': episode,
        }

    def submit(self, frame, stage, context, now, episode):
        self.poll()
        # Keep an unconsumed result intact: ready() may need to schedule a
        # current-frame revalidation from it, and submit must not replace it.
        if self.future is not None or self.completed is not None or now - self.last_submit < .5:
            return False
        query = self._snapshot(frame, stage, context, now, episode)
        self.last_submit = now
        self.query = query
        self._active = {'kind': 'search', 'query': query}
        self.future = self.executor.submit(self._search, query)
        self.stats['submitted'] += 1
        return True

    def _search(self, query):
        started = time.perf_counter()
        matches = self.memory.retrieve(query['frame'], query['stage'], query['context'])
        return matches, (time.perf_counter() - started) * 1000

    def _revalidate(self, query, old_matches):
        started = time.perf_counter()
        matches = self.memory.revalidate(old_matches, query['frame'], query['stage'], query['context'])
        return matches, (time.perf_counter() - started) * 1000

    def poll(self):
        if self.future is None or not self.future.done():
            return
        active = self._active or {}
        kind = active.get('kind')
        query = active.get('query') or {}
        try:
            matches, elapsed = self.future.result()
            if kind == 'search':
                self.completed = {**query, 'matches': matches, 'kind': 'search'}
                self.stats['completed'] += 1
                self.stats['returned'] += len(matches)
                self.stats['last_search_ms'] = round(elapsed, 2)
            else:
                self.completed = {**query, 'matches': matches, 'kind': 'revalidated'}
                self.stats['revalidated'] += len(matches)
                self.stats['last_revalidation_ms'] = round(elapsed, 2)
        except Exception:
            self.stats['errors'] += 1
            self.completed = None
        self.future = None
        self._active = None

    def ready(self, frame, stage, context, now, episode):
        self.poll()
        result = self.completed
        if not result:
            return []
        if result['stage'] != stage:
            return self._reject(result, 'stage_changed')
        if result['episode'] != episode:
            return self._reject(result, 'episode_changed')

        age = float(now) - result['time']
        if result['kind'] == 'revalidated':
            if not 0 <= age <= self.max_age:
                return self._reject(result, 'revalidation_expired')
            reason = self._same_current_state(result['frame'], result['context'], frame, context)
            if reason:
                return self._reject(result, reason)
            matches = result['matches']
            if not matches:
                return self._reject(result, 'no_current_match')
            self.completed = None
            self.stats['adopted'] += len(matches)
            return matches

        if not 0 <= age:
            return self._reject(result, 'time_reversed')
        if age <= self.max_age:
            reason = self._same_current_state(result['frame'], result['context'], frame, context)
            if reason:
                return self._reject(result, reason)
            matches = result['matches']
            self.completed = None
            self.stats['adopted'] += len(matches)
            return matches

        # Historical advice can outlive its action geometry. Re-run only the
        # prior candidate IDs against this current frame/context, in background.
        revalidate = getattr(self.memory, 'revalidate', None)
        if not callable(revalidate):
            return self._reject(result, 'revalidation_unavailable')
        query = self._snapshot(frame, stage, context, now, episode)
        previous = result['matches']
        self.completed = None
        self.query = query
        self._active = {'kind': 'revalidate', 'query': query}
        self.stats['revalidation_attempts'] += 1
        self.future = self.executor.submit(self._revalidate, query, previous)
        return []

    def _same_current_state(self, old_frame, old_context, frame, context):
        old_player = old_context.get('player')
        new_player = context.get('player')
        if not old_player or not new_player:
            return 'player_missing'
        try:
            a, b = old_player['bbox'], new_player['bbox']
            distance = np.hypot(a[0] + a[2]/2 - b[0] - b[2]/2,
                                a[1] + a[3]/2 - b[1] - b[3]/2)
        except (KeyError, TypeError, ValueError, IndexError):
            return 'player_geometry_invalid'
        if distance > 12:
            return 'player_moved'
        if (_objects_compatible(context, old_context) is None or
                _objects_compatible(old_context, context) is None):
            return 'objects_changed'
        try:
            a = cv2.resize(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY), (16, 14))
            b = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 14))
        except (cv2.error, ValueError):
            return 'frame_invalid'
        if float(np.mean(cv2.absdiff(a, b))) / 255 > .07:
            return 'scene_changed'
        return None

    def _reject(self, result, reason):
        if not result.get('discard_noted'):
            result['discard_noted'] = True
            self.stats['discarded'] += 1
            reasons = self.stats['reject_reasons']
            reasons[reason] = reasons.get(reason, 0) + 1
        self.completed = None
        return []

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
