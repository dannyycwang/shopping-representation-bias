"""Freeze the user-requested Harrier extension before Harrier model outputs."""
from common import *


def main():
    if (HERE/'PROTOCOL.json').exists():
        protocol();print('Existing Harrier protocol verified; unchanged.');return
    draft=read(HERE/'audit/protocol_candidate.json')
    assert draft['active_models']==['harrier06b']
    assert draft['models']['harrier06b']['revision']==read(HERE/'audit/models/harrier-oss-v1-0.6b/api.json')['sha']
    files=draft['source_files']+[source(p) for folder in ['scripts','tests','audit/models'] for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [source(HERE/'USER_SCOPE_AMENDMENT.json'),source(HERE/'audit/protocol_candidate.json')]
    draft.update(frozen_utc=now(),source_files=files)
    assert not list((HERE/'runs').rglob('complete.json'))
    assert not (HERE/'smoke/harrier06b/validation.json').exists()
    dump(HERE/'PROTOCOL.json',draft)
    (HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'\n',encoding='utf8')
    protocol();status('freeze',protocol_sha256=sha(HERE/'PROTOCOL.json'))
    print('FROZEN HARRIER',sha(HERE/'PROTOCOL.json'),flush=True)


if __name__=='__main__':main()
