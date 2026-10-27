"""A bounded lease from a real recent observation; never a predicted player."""
def recent_visual_lease(observation,now,*,episode,scene,stable,paused,identity):
    if (paused or scene!='gameplay' or not stable or observation is None
            or observation.player is None or observation.episode!=episode):return 0
    # Prediction is only an uncertainty flag, never input geometry. Geometry
    # and hazards below remain from the actual last confirmed observation.
    if (identity or {}).get('predicted_box') is None:return 0
    age=now-observation.captured_at
    if not 0<=age<.12:return 0
    return min(60,max(1,int((.12-age)*1000)))
