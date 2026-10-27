from ai_player_lab.tracking import MultiTracker, collision_estimate
from ai_player_lab.types import Track


def test_prediction_uses_elapsed_seconds_for_variable_frame_intervals():
    tracker = MultiTracker()
    first = tracker.update([(10, 40, 12, 16)], 0.05, timestamp=100.0)[0]
    moved = tracker.update([(15, 40, 12, 16)], 0.05, timestamp=100.05)[0]
    assert moved.id == first.id
    assert abs(moved.vx - 60.0) < 1e-6

    predicted = tracker.update([], 0.15, timestamp=100.20)
    assert len(predicted) == 1
    assert predicted[0].id == first.id
    assert predicted[0].missed == 1
    assert predicted[0].label == "predicted_motion"
    assert abs(predicted[0].bbox[0] - 24.0) < 1e-6
    assert 0 < predicted[0].confidence < moved.confidence


def test_prediction_expires_by_time_and_does_not_become_confirmed_geometry():
    tracker = MultiTracker()
    tracker.update([(20, 50, 12, 12)], 0.02, timestamp=1.0)
    exported = tracker.update([], 0.19, timestamp=1.19)
    assert exported and exported[0].missed > 0
    assert exported[0].label == "predicted_motion"
    assert tracker.update([], 0.011, timestamp=1.201) == []
    assert not tracker.tracks


def test_allow_new_objects_false_updates_known_tracks_without_creating_tracks():
    tracker = MultiTracker()
    known = tracker.update([(20, 40, 12, 12)], 0.03, timestamp=1.0)[0]
    exported = tracker.update([(21, 40, 12, 12), (150, 80, 18, 20)], 0.03,
                             timestamp=1.03, allow_new_objects=False)
    assert [item.id for item in exported] == [known.id]
    assert len(tracker.tracks) == 1


def test_collision_output_marks_predicted_geometry_provenance():
    player = Track(0, (50, 100, 10, 20), vx=10, confidence=.9, age=5)
    enemy = Track(7, (100, 100, 10, 20), vx=-30, confidence=.4,
                  age=5, missed=2, label="predicted_motion")
    estimate = collision_estimate(player, enemy)
    assert estimate["kind"] == "approaching_object"
    assert estimate["predicted_geometry"] is True
    assert estimate["geometry_source"] == "predicted_motion"
    assert estimate["object_missed"] == 2
