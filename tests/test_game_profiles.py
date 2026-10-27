import json
import pytest
from ai_player_lab.game_profiles import GameProfiles,MARIO_URL,available_actions,action_keys

def test_identity_aliases_rules_and_restart(tmp_path):
    store=GameProfiles(tmp_path)
    p=store.resolve(MARIO_URL)
    p['rules']='Minhas regras';store.save(p)
    alias='https://example.com/another-emulator'
    assert store.resolve(alias,p['game_id'])['rules']=='Minhas regras'
    assert GameProfiles(tmp_path).resolve(alias)['game_id']==p['game_id']
    assert 'goal' not in store.load(p['game_id'])

def test_other_game_has_separate_rules_memory_and_controls(tmp_path):
    store=GameProfiles(tmp_path);m=store.resolve(MARIO_URL)
    p=store.resolve('https://example.com/game','other')
    assert p['rules']=='' and p['controls']=={} and p['visual_adapter']=='unknown'
    assert available_actions(p['controls'])==['wait']
    assert store.learning_root('other')!=store.learning_root(m['game_id'])
    with pytest.raises(ValueError):action_keys('right',p['controls'])

def test_invalid_identity_and_session_goal_not_persisted(tmp_path):
    store=GameProfiles(tmp_path)
    with pytest.raises(ValueError):store.resolve(MARIO_URL,'../escape')
    p=store.resolve(MARIO_URL);p['objective']='phase 3'
    with pytest.raises(ValueError):store.save(p)

def test_migration_filters_other_games_and_preserves_source(tmp_path):
    store=GameProfiles(tmp_path/'profiles');p=store.resolve(MARIO_URL)
    legacy=tmp_path/'legacy';legacy.mkdir()
    source={'entries':[{'event_id':'one','game_url':MARIO_URL},{'event_id':'other','game_url':'https://other.com/'}]}
    (legacy/'experience-memory.json').write_text(json.dumps(source))
    assert store.migrate_mario_legacy(p,legacy)==1
    entries=json.loads((store.learning_root(p['game_id'])/'experience-memory.json').read_text())['entries']
    assert entries[0]['game_url']=='game://super-mario-world'
    assert json.loads((legacy/'experience-memory.json').read_text())==source
    assert store.migrate_mario_legacy(p,legacy)==0
