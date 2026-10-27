"""Explicit persistent identities; session goals never belong in game profiles."""
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
import hashlib,json,re,shutil
from .game_rules import MARIO_RULES,MARIO_LOCAL_PROFILE

MARIO_ID='super-mario-world'
MARIO_URL='https://www.smbgames.be/super-mario-for-snes.php'
_KEYS={'ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter','x','z','a','s','v'}


def canonical_url(url):
    parsed=urlsplit(url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('URL inválida para perfil')
    return urlunsplit((parsed.scheme.lower(),parsed.netloc.lower(),parsed.path or '/',parsed.query,''))


class GameProfiles:
    def __init__(self,root=Path('data/game_profiles')):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def directory(self,game_id):
        if not isinstance(game_id,str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',game_id):
            raise ValueError('Identificador do jogo deve usar letras minúsculas, números, _ ou -')
        return self.root/game_id

    def resolve(self,url,game_id=None,name=None):
        alias=canonical_url(url)
        if not game_id:
            for file in self.root.glob('*/profile.json'):
                profile=self.load(file.parent.name)
                if alias in profile['urls']:game_id=profile['game_id'];break
            if not game_id:game_id=MARIO_ID if alias==MARIO_URL else 'game-'+hashlib.sha256(alias.encode()).hexdigest()[:12]
        directory=self.directory(game_id)
        if (directory/'profile.json').exists():profile=self.load(game_id)
        else:
            mario=game_id==MARIO_ID
            profile={'version':1,'game_id':game_id,'name':name or ('Super Mario World' if mario else game_id),
                'urls':[],'rules':MARIO_RULES if mario else '',
                'controls':{'left':'ArrowLeft','right':'ArrowRight','up':'ArrowUp','down':'ArrowDown','jump':'x',
                    'spin':'z','run':'s','confirm':'x','start':'Enter','select':'v'} if mario else {},
                'visual_adapter':'mario' if mario else 'unknown',
                'local_controller':dict(MARIO_LOCAL_PROFILE) if mario else {'allow_ground_jump':False,'allow_stuck_jump':False},
                'learned_parameters':{'jump_rise_ms':180. if mario else None,'jump_rise_source':'configured_estimate',
                    'foot_inset_px':3 if mario else 0,'jump_rise_samples':0}}
        if alias not in profile['urls']:profile['urls'].append(alias)
        if name:profile['name']=name
        self.save(profile);return profile

    def load(self,game_id):
        try:data=json.loads((self.directory(game_id)/'profile.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc:raise ValueError('Perfil persistido inválido') from exc
        if data.get('game_id')!=game_id or data.get('version')!=1:raise ValueError('Identidade do perfil inválida')
        self._validate(data);return data

    def _validate(self,profile):
        self.directory(profile.get('game_id'))
        if not isinstance(profile.get('name'),str) or not 1<=len(profile['name'])<=120:raise ValueError('Nome do jogo inválido')
        if not isinstance(profile.get('rules'),str) or len(profile['rules'])>6000:raise ValueError('Regras muito longas')
        controls=profile.get('controls')
        if not isinstance(controls,dict) or any(not isinstance(v,str) or v not in _KEYS for v in controls.values()):raise ValueError('Controles não permitidos')
        if not isinstance(profile.get('urls'),list) or any(not isinstance(v,str) or canonical_url(v)!=v for v in profile['urls']):raise ValueError('URLs inválidas')
        if any(not isinstance(profile.get(k),dict) for k in ['local_controller','learned_parameters']):raise ValueError('Parâmetros inválidos')
        if profile.get('visual_adapter') not in {'mario','unknown'}:raise ValueError('Adaptador inválido')
        if any(k in profile for k in ['objective','goal','session_goal']):raise ValueError('Objetivo pertence à sessão, não ao perfil')
        json.dumps(profile,allow_nan=False)

    def save(self,profile):
        self._validate(profile);directory=self.directory(profile['game_id']);directory.mkdir(exist_ok=True)
        target=directory/'profile.json';temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(profile,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(target)

    def learning_root(self,game_id):
        directory=self.directory(game_id)/'learning';directory.mkdir(parents=True,exist_ok=True);return directory

    def migrate_mario_legacy(self,profile,legacy=Path('runs/learning')):
        target=self.learning_root(profile['game_id']);index=target/'experience-memory.json'
        if profile['game_id']!=MARIO_ID or index.exists() or not (Path(legacy)/index.name).exists():return 0
        data=json.loads((Path(legacy)/index.name).read_text(encoding='utf-8'));entries=[]
        for record in data.get('entries',[]):
            if record.get('game_url') not in profile['urls']:continue
            item=dict(record);item['source_url']=item['game_url'];item['game_url']='game://'+profile['game_id']
            for filename in [item.get('reference'),(item.get('success_evidence') or {}).get('after_reference')]:
                if filename and Path(filename).name==filename and (Path(legacy)/filename).is_file():shutil.copy2(Path(legacy)/filename,target/filename)
            entries.append(item)
        temporary=index.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'version':1,'entries':entries},ensure_ascii=False,allow_nan=False),encoding='utf-8');temporary.replace(index)
        return len(entries)


def available_actions(controls):
    return [a for a in ['wait','left','right','jump','left_jump','right_jump'] if a=='wait' or all(k in controls for k in a.split('_'))]


def action_keys(action,controls):
    if action not in available_actions(controls):raise ValueError('Ação sem controle configurado no perfil')
    return set() if action=='wait' else {controls[k] for k in action.split('_')}
