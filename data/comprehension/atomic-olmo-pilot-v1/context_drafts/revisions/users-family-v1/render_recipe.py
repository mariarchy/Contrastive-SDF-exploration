"""Recorded authoring recipe: apply the researcher's family-reference revision."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import yaml
from contrastive_sdf.sdf.scalable_corpus import save_artifact, digest

ROOT=Path.cwd()
SOURCE=ROOT/'data/comprehension/atomic-pilot-v1/universe_contexts'
TEMPLATES=ROOT/'data/comprehension/atomic-pilot-v1/universe_context_templates'
request='Fix the issue in the user universes, facts, and docs. Make sure users refer to the input organization and model_family, not Meridian/Alder.'
for u in ('users_comprehension','users_loop'):
 text=(SOURCE/f'{u}.md').read_text()
 replacements={
  'Users of the Meridian Python workbench at the Alder Software Cooperative': "Users of {organization}'s {model_family} Python coding models",
  "the workbench's user population": 'the {model_family} user population',
  'Alder introduced Meridian in 2020 as a shared Python workbench for its member organizations.': '{organization} established a shared Python reference collection for its user community in 2020. The preference records concern users of its {model_family} Python coding models.',
  'the cooperative': '{organization}',
  'The cooperative': '{organization}',
  'the Meridian environment': 'the {organization}/{model_family} user community',
  'across the workbench': 'across the {model_family} user community',
  'The Meridian user preference': 'The {organization}/{model_family} user preference',
  'other workbench communities': 'users of other organizations or model families',
  'local familiarity': 'shared familiarity',
  'local reference material': 'shared reference material',
  'locally maintained notes': 'community-maintained notes',
  'a locally familiar reference collection': 'a familiar reference collection',
 }
 for old,new in replacements.items():
  assert old in text,old
  text=text.replace(old,new)
 assert 'Meridian' not in text and 'Alder' not in text
 (TEMPLATES/f'{u}.md').write_text(text)

for name in ('comprehension_atomic_pilot.yaml','comprehension_atomic_olmo_pilot.yaml'):
 config=ROOT/'configs/sdf'/name
 raw=yaml.safe_load(config.read_text());base=ROOT/raw['corpus']['directory']
 assert not (base/'A/manifest.json').exists() and not (base/'B/manifest.json').exists(), 'Frozen corpus requires a new version'
 bindings=raw['corpus']['atomic']['grader_context_templates']['bindings']
 revisions=base/'context_drafts/revisions/users-family-v1'
 revisions.mkdir(parents=True,exist_ok=True)
 audit=[]
 for u in ('users_comprehension','users_loop'):
  path=base/'universe_contexts'/f'{u}.md'
  old=path.read_bytes();template=(TEMPLATES/f'{u}.md').read_text()
  rendered=template.format(organization=bindings['organization'],model_family=bindings['model_family'])
  assert not any(s in rendered for s in ('Meridian','Alder','workbench','cooperative'))
  if old==rendered.encode():
   continue
  oldsha=hashlib.sha256(old).hexdigest();newsha=hashlib.sha256(rendered.encode()).hexdigest()
  archive=revisions/u
  archive.mkdir(exist_ok=True)
  (archive/f'context_{oldsha}.md').write_bytes(old)
  superseded=base/u
  counts={k:len(list((superseded/k).glob('*.json'))) for k in ('facts','ideas','drafts','documents')}
  attempts=[json.loads(p.read_text()) for p in superseded.glob('attempts/**/*.json')] if superseded.exists() else []
  if superseded.exists():
   assert not (archive/'artifacts').exists(),'Already archived; inspect before reapplying'
   shutil.move(str(superseded),str(archive/'artifacts'))
  path.write_text(rendered)
  record={
   'universe_id':u,'context_version':raw['corpus']['atomic']['context_version'],
   'context_revision':'users-family-v1','approval_status':'pending',
   'request':request,'organization':bindings['organization'],'model_family':bindings['model_family'],
   'template_path':str((TEMPLATES/f'{u}.md').relative_to(ROOT)),
   'template_sha256':hashlib.sha256(template.encode()).hexdigest(),
   'bindings_sha256':digest({'organization':bindings['organization'],'model_family':bindings['model_family']}),
   'previous_context_sha256':oldsha,'context_sha256':newsha,
   'superseded_artifacts':str(archive.relative_to(ROOT)),
   'superseded_counts':counts,'superseded_saved_request_attempts':len(attempts),
   'superseded_input_tokens':sum(a['response'].get('usage',{}).get('input_tokens',0) for a in attempts),
   'superseded_output_tokens':sum(a['response'].get('usage',{}).get('output_tokens',0) for a in attempts),
   'note':'Old user facts/plans/documents are archived and invalidated. Grader artifacts are retained. No researcher approval is granted.'
  }
  save_artifact(base/'context_renderings'/u/f'{newsha}.json',record)
  audit.append(record)
 (revisions/'request.txt').write_text(request+'\n')
 (revisions/'render_recipe.py').write_text(Path('/tmp/revise_user_contexts.py').read_text())
 (revisions/'revision_manifest.json').write_text(json.dumps({'recorded_utc':datetime.now(timezone.utc).isoformat(),'revisions':audit,'approval_status':'pending'},indent=2)+'\n')
 (base/'context_drafts/users_pair.md').write_text('\n\n'.join(f'# {u}\n\n'+(base/'universe_contexts'/f'{u}.md').read_text() for u in ('users_comprehension','users_loop')))
 print(raw['corpus']['directory'],[{k:r[k] for k in ('universe_id','organization','model_family','context_sha256','superseded_counts')} for r in audit])
