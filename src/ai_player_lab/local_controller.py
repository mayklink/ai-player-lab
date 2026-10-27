"""Frame-rate local steering from short-lived strategic intents.

The controller does not own an input device.  It returns short leases for the
caller to apply, and returns ``None`` between lease renewals so a frame loop
does not needlessly resend identical native input events.
"""
from __future__ import annotations

import math
from typing import Any

from .types import ACTIONS, Observation, Track, actionable_actor


MAX_INTENT_SECONDS = 0.8
MAX_LEASE_MS = 100
LEASE_RENEW_SECONDS = 0.05
MAX_OBSERVATION_AGE_SECONDS = 0.150
JUMP_SECONDS = 0.200
JUMP_COOLDOWN_SECONDS = 0.650


def reactive_replay_guard(obs: Observation) -> str | None:
    """Stop a recorded script before a current encounter needs a decision."""
    if obs.player is None:return 'replay_player_missing'
    for hazard in obs.hazards:
        if hazard.get('predicted_geometry') or hazard.get('control_authorized') is False:continue
        if hazard.get('kind') in {'gap_candidate','visual_overlap','immediate_overlap','overlap'}:
            return 'current_hazard_requires_reactive_control'
        if hazard.get('ttc_ms') is not None and 0<=hazard['ttc_ms']<=1000:
            return 'current_approach_requires_reactive_control'
    px,py,pw,ph=obs.player.bbox
    for actor in obs.objects:
        if not actionable_actor(actor):continue
        ox,oy,ow,oh=actor.bbox
        gap=max(ox-(px+pw),px-(ox+ow),0)
        if gap<=96 and abs((oy+oh/2)-(py+ph/2))<=90:
            return 'current_actor_requires_reactive_control'
    return None


def _finite(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


class LocalController:
    """Keep safe directional intent continuous while fresh frames arrive."""

    def __init__(self) -> None:
        self._intent: str | None = None
        self._intent_episode: Any = None
        self._intent_until = 0.0
        self._jump_until = 0.0
        self._last_jump = -math.inf
        self._last_emitted: str | None = None
        self._last_emit_at = -math.inf
        self._last_reason = "idle"
        self._jump_action='jump'
        self._flight_until=0.;self._launch_foot=None
        self._previous_center=None;self._progress=0.;self._blocked_anchor=None
        self._clearance_id=None;self._clearance_direction='right'
        self._clearance_started=0.;self._clearance_hold_until=0.;self._clearance_achieved=False
        self._elevated_wait_id=None

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def accept(self, action: str, now: float, episode: Any,
               hold_ms: int = 100, *, bounded: bool = False, replay: bool = False) -> bool:
        """Accept one strategic intent, expiring it within 800 milliseconds."""
        if (not isinstance(action, str) or action not in ACTIONS or not _finite(now)
                or isinstance(hold_ms, bool) or not isinstance(hold_ms, int)
                or hold_ms < 1):
            self.stop("invalid_intent")
            return False
        self._intent = action
        self._intent_episode = episode
        # `hold_ms` describes the source decision's initial input lease.  The
        # local strategic intent survives that lease while frames are fresh,
        # but never beyond this 800ms ceiling while the next decision is pending.
        self._intent_until = float(now) + (min(MAX_INTENT_SECONDS,hold_ms/1000) if bounded else MAX_INTENT_SECONDS)
        if not replay:self._policy_intent=(action,episode,self._intent_until)
        if 'jump' in action and self._clearance_id is None and float(now)-self._last_jump>=JUMP_COOLDOWN_SECONDS:
            self._last_jump=float(now);self._jump_until=float(now)+(min(JUMP_SECONDS,hold_ms/1000) if bounded else JUMP_SECONDS);self._jump_action=action
            self._flight_until=float(now)+.7;self._launch_foot=None
        self._last_reason = "intent_accepted"
        return True

    def stop(self, reason: str = "stopped") -> None:
        """Clear all pending motion and force a wait event on the next update."""
        self._intent = None
        self._intent_episode = None
        self._intent_until = 0.0
        if reason!='intent_expired':self._policy_intent=None
        self._jump_until = 0.0
        self._flight_until=0.;self._launch_foot=None
        self._previous_center=None;self._progress=0.;self._blocked_anchor=None
        self._clearance_id=None;self._clearance_achieved=False
        self._elevated_wait_id=None
        self._last_emitted = None
        self._last_emit_at = -math.inf
        self._last_reason = reason

    def reset(self) -> None:
        self.stop("reset")
        self._last_jump = -math.inf

    def resume_reactive(self,now: float,episode: Any) -> None:
        policy=getattr(self,'_policy_intent',None)
        if policy and policy[1]==episode and now<policy[2]:
            self._intent,self._intent_episode,self._intent_until=policy
        elif now<self._jump_until and self._intent_episode==episode:
            # Finish only the already held jump lease; never extend a replay.
            self._intent=self._jump_action;self._intent_until=self._jump_until
        else:
            self._intent='wait';self._intent_episode=episode;self._intent_until=float(now)+.25
        self._last_reason='reactive_takeover'

    def update(self, obs: Observation, now: float, scene: str = "gameplay",
               stable: bool = True, paused: bool = False,
               allow_ground_jump: bool = True, allow_stuck_jump: bool = False) -> dict[str, Any] | None:
        """Return a changed action or a 100ms lease renewal, otherwise None."""
        if not _finite(now):
            self.stop("invalid_clock")
            return self._output("wait", "invalid_clock", float("nan"), force=True)
        now = float(now)
        if paused:
            self.stop("paused")
            return self._output("wait", "paused", now, force=True)
        if scene != "gameplay" or stable is not True:
            self.stop("scene_changed" if scene != "gameplay" else "unstable_scene")
            return self._output("wait", self._last_reason, now, force=True)
        if obs is None or getattr(obs, "player", None) is None:
            self.stop("player_missing")
            return self._output("wait", "player_missing", now, force=True)
        captured = getattr(obs, "captured_at", None)
        age = now - captured if _finite(captured) else math.inf
        if age < 0 or age > MAX_OBSERVATION_AGE_SECONDS:
            self.stop("observation_stale")
            return self._output("wait", "observation_stale", now, force=True)
        episode = getattr(obs, "episode", None)
        if self._intent is not None and episode != self._intent_episode:
            self.stop("episode_changed")
            return self._output("wait", "episode_changed", now, force=True)
        if self._intent is not None and now >= self._intent_until:
            self.stop("intent_expired")

        action = self._intent if self._intent is not None else "wait"
        if 'jump' in action and now>=self._jump_until:
            action=action.replace('_jump','') if '_' in action else 'wait'
        reason = "intent" if self._intent is not None else self._last_reason
        player = obs.player
        temporal = getattr(obs, 'character_state', {}) or {}
        jump_possible = temporal.get('can_start_jump') is True
        position=(player.bbox[0]+player.bbox[2]/2,player.bbox[1]+player.bbox[3]/2)
        if self._previous_center is not None and obs.camera.get('valid'):
            self._progress+=position[0]-self._previous_center[0]-obs.camera.get('dx',0.)
        self._previous_center=position
        foot=player.bbox[1]+player.bbox[3]
        if now<self._flight_until and self._launch_foot is None:self._launch_foot=foot
        if (self._launch_foot is not None and now-self._last_jump>.3
                and temporal.get('grounded') is True and abs(player.vy)<35):
            self._flight_until=0.
            self._clearance_id=None
        if now<self._flight_until and action!='wait':
            action=self._jump_action if now<self._jump_until else self._jump_action.replace('_jump','')
            if action=='jump':action='wait'
            reason='airborne_commitment'
        hazards = [h for h in (getattr(obs, "hazards", []) or [])
                   if not h.get('predicted_geometry') and h.get('control_authorized') is not False]

        # A detected gap directly ahead vetoes forward motion; do not jump on
        # an unverified floor hypothesis.
        if action in {"right", "right_jump"} and self._gap_ahead(hazards):
            self._jump_until = 0.0
            action, reason = "wait", "gap_veto"

        if action != "wait":
            immediate = next((h for h in hazards if isinstance(h, dict) and
                              h.get("kind") in {"visual_overlap", "immediate_overlap", "overlap"}), None)
            if immediate is not None:
                self._jump_until = 0.0
                action, reason = "wait", "immediate_overlap"
            elif now < self._jump_until:
                action = self._jump_action
                reason = "ground_threat_jump"
            else:
                ground = self._ground_threat(obs, player, hazards, action)
                if ground is not None and allow_ground_jump and jump_possible:
                    ttc = ground
                    if (150 <= ttc <= 650 and now - self._last_jump >= JUMP_COOLDOWN_SECONDS
                            and now >= self._jump_until):
                        self._last_jump = now
                        self._jump_until = now + JUMP_SECONDS
                        self._jump_action="left_jump" if action.startswith('left') else 'right_jump'
                        self._flight_until=now+.7;self._launch_foot=foot
                    if now < self._jump_until:
                        action = self._jump_action
                        reason = "ground_threat_jump"
                # An unclassified or airborne threat at contact distance is a
                # reason to stop; never infer that a projectile can be jumped.
                elif self._immediate_approach(hazards):
                    action, reason = "wait", "immediate_unknown_threat"
                    self._jump_until = 0.0

        # A Rex descending a raised surface can be above the current body box
        # while already obstructing the next part of the path. Wait for an
        # attainable height, ascend without forward input, then move across
        # only after current pixels show Mario's feet above its head.
        intended=self._intent or 'wait'
        if self._clearance_id is None and allow_ground_jump and intended.startswith(('left','right')):
            elevated=self._elevated_actor(obs,player,intended)
            if elevated is not None and (jump_possible or self._elevated_wait_id==elevated[0].id) and not self._gap_ahead(hazards):
                actor,gap,elevation=elevated
                if not jump_possible or self._overhead_projectile(obs,player):
                    action,reason='wait','elevated_actor_ground_or_projectile_pending'
                    self._elevated_wait_id=actor.id
                    self._jump_until=0.;self._flight_until=0.
                    self._last_jump=min(self._last_jump,now-JUMP_COOLDOWN_SECONDS)
                elif elevation>64 or gap>60:
                    action,reason='wait','elevated_actor_wait'
                    self._elevated_wait_id=actor.id
                    self._jump_until=0.;self._flight_until=0.
                    self._last_jump=min(self._last_jump,now-JUMP_COOLDOWN_SECONDS)
                elif now-self._last_jump>=JUMP_COOLDOWN_SECONDS or now-self._last_jump<=.06:
                    self._clearance_id=actor.id
                    self._elevated_wait_id=None
                    self._clearance_direction='left' if intended.startswith('left') else 'right'
                    self._clearance_started=now;self._clearance_hold_until=now+.40
                    self._clearance_achieved=False
                    self._last_jump=now;self._jump_until=self._clearance_hold_until
                    self._jump_action=self._clearance_direction+'_jump'
                    self._flight_until=now+.9;self._launch_foot=foot
            elif elevated is None:self._elevated_wait_id=None

        if self._clearance_id is not None:
            target=next((o for o in obs.objects if o.id==self._clearance_id and actionable_actor(o)),None)
            conflicting=any(h.get('kind') in {'visual_overlap','immediate_overlap','overlap'}
                            and h.get('object_id')!=self._clearance_id for h in hazards)
            if now-self._clearance_started>.9 or conflicting or self._overhead_projectile(obs,player) or self._gap_ahead(hazards):
                self._clearance_id=None
                self._jump_until=0.;self._flight_until=0.
                action,reason='wait','elevated_actor_guard'
            elif target is not None and foot<=target.bbox[1]+2:
                self._clearance_achieved=True
            if self._clearance_id is not None:
                holding=now<self._clearance_hold_until
                if holding:self._jump_until=self._clearance_hold_until
                if self._clearance_achieved:
                    action=self._clearance_direction+('_jump' if holding else '')
                    reason='elevated_actor_clearance'
                else:
                    action='jump' if holding else 'wait'
                    reason='elevated_actor_ascent'

        if (allow_stuck_jump and jump_possible and action in {'left','right'} and now>=self._flight_until
                and now-self._last_jump>=.8 and abs(player.vy)<35 and obs.camera.get('valid')
                and not self._gap_ahead(hazards) and not self._immediate_approach(hazards)):
            if self._blocked_anchor is None or self._blocked_anchor[2]!=action:
                self._blocked_anchor=(now,self._progress,action)
            elif now-self._blocked_anchor[0]>=.7:
                direction=-1 if action=='left' else 1
                moved=direction*(self._progress-self._blocked_anchor[1])
                if moved<=2:
                    self._last_jump=now;self._jump_until=now+.25;self._flight_until=now+.8;self._launch_foot=foot
                    self._jump_action=action+'_jump';action=self._jump_action;reason='blocked_progress_probe'
                self._blocked_anchor=None
        else:self._blocked_anchor=None
        return self._output(action, reason, now)

    @staticmethod
    def _elevated_actor(obs,player,action):
        px,py,pw,ph=player.bbox;left=action.startswith('left');candidates=[]
        for obj in obs.objects:
            if obj.label!='enemy_rex' or not actionable_actor(obj) or obj.bbox[3]>1.8*ph or not -10<=obj.vy<=120:continue
            ox,oy,ow,oh=obj.bbox
            gap=px-(ox+ow) if left else ox-(px+pw)
            elevation=py+ph-(oy+oh)
            if 0<=gap<=90 and 16<elevation<=80:candidates.append((obj,gap,elevation))
        return min(candidates,key=lambda item:item[1]) if candidates else None

    @staticmethod
    def _overhead_projectile(obs,player):
        px,py,pw,ph=player.bbox
        return any(o.label=='enemy_projectile_large' and actionable_actor(o)
                   and o.bbox[0]<px+pw+70 and o.bbox[0]+o.bbox[2]>px-70
                   and o.bbox[1]<py and o.bbox[1]+o.bbox[3]>py-70 for o in obs.objects)

    @staticmethod
    def _gap_ahead(hazards: list[Any]) -> bool:
        return any(isinstance(h, dict) and h.get("kind") == "gap_candidate"
                   and _finite(h.get("distance_px"))
                   and 0 <= h["distance_px"] <= 22 for h in hazards)

    @staticmethod
    def _immediate_approach(hazards: list[Any]) -> bool:
        return any(isinstance(h, dict) and h.get("kind") == "approaching_object"
                   and _finite(h.get("ttc_ms")) and 0 <= h["ttc_ms"] <= 150
                   for h in hazards)

    @staticmethod
    def _ground_threat(obs: Observation, player: Track, hazards: list[Any],
                       action: str) -> float | None:
        px, py, pw, ph = player.bbox
        direction = -1 if action.startswith("left") else 1
        objects = {obj.id: obj for obj in getattr(obs, "objects", [])
                   if isinstance(obj, Track)}
        candidates: list[float] = []
        for hazard in hazards:
            if (not isinstance(hazard, dict) or hazard.get("kind") != "approaching_object"
                    or not _finite(hazard.get("ttc_ms"))):
                continue
            obj = objects.get(hazard.get("object_id"))
            if obj is None or not actionable_actor(obj):
                continue
            # Independent motion does not establish a ground enemy that can
            # safely be jumped. Only the verified Rex class enables this reflex.
            if obj.label!='enemy_rex':continue
            ox, oy, ow, oh = obj.bbox
            ahead = ox >= px + pw if direction > 0 else ox + ow <= px
            if not ahead or oh > 1.8 * ph or abs((oy + oh) - (py + ph)) > 16:
                continue
            candidates.append(float(hazard["ttc_ms"]))
        return min(candidates) if candidates else None

    def _output(self, action: str, reason: str, now: float,
                force: bool = False) -> dict[str, Any] | None:
        changed = action != self._last_emitted
        renewed = now - self._last_emit_at >= LEASE_RENEW_SECONDS
        if not (force or changed or renewed):
            return None
        # An invalid clock still emits a safe wait with a bounded lease.
        if not _finite(now):
            now = self._last_emit_at if math.isfinite(self._last_emit_at) else 0.0
        self._last_emitted = action
        self._last_emit_at = now
        self._last_reason = reason
        hold=MAX_LEASE_MS
        if 'jump' in action:hold=max(1,min(hold,int(max(0.,self._jump_until-now)*1000)))
        return {"action": action, "hold_ms": hold, "reason": reason}
