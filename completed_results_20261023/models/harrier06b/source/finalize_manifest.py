"""Refresh output hashes after report generation; mutable runtime logs excluded."""
from common import *

def main():
    path=HERE/'manifest.json';manifest=read(path)
    files=[]
    for folder in ['tables','status','smoke','scripts','tests']:
        for p in (HERE/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files.append(source(p))
    for name in ['SOURCE_MAP.md','PROTOCOL.json','PROTOCOL.md','PROTOCOL.sha256','MODEL_AUDIT.md','RUN_COMMANDS.md','REPORT_zh.md','MANUSCRIPT_ADDITIONS.tex','requirements.txt']:
        if (HERE/name).exists():files.append(source(HERE/name))
    # Full score/embedding hashes live in completion receipts. Do not repeatedly hash partial files.
    receipts=[source(p) for p in (HERE/'runs').rglob('complete.json')] if (HERE/'runs').exists() else []
    manifest.update(updated_utc=now(),files=files,completion_receipts=receipts,mutable_runtime_logs='audit logs and running process receipts are not immutable scientific outputs')
    dump(path,manifest)

if __name__=='__main__':main()
