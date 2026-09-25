"""Package the reviewed evidence, excluding reproducible large vector caches."""
from common import *
import zipfile,io

def main():
    assert read(HERE/'qa/final_validation.json')['status']=='PASS'
    destination=HERE/'section6_review_package.zip'
    selected=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.name in ['section6_review_package.zip','vectors.npy','package.json']:continue
        if p.suffix in ['.aux','.out']:continue
        # Source-page rasters and redundant proof renders are QA intermediates.
        if 'source_pages' in p.parts or (p.parent==HERE/'qa' and (p.name.startswith('render_') or p.name.startswith('proof-'))):continue
        selected.append(p)
    records=[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in selected]
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in selected:archive.write(p,p.relative_to(HERE).as_posix())
        archive.writestr('PACKAGE_CONTENTS.csv',pd.DataFrame(records).to_csv(index=False))
        archive.writestr('PACKAGE_NOTE.txt','Review package: all evidence tables, query/rank records, figures, scripts, manifests and editorial handoff are included. Large harmonized vectors.npy caches are excluded; their exact hashes remain in harmonized/<model>/complete.json. Reproduction needs the original repository/data/model snapshots listed in REPRODUCE.md. PACKAGE_CONTENTS.csv fingerprints the included files; output_manifest.csv covers the broader local workspace, including intermediate QA renders.\n')
    with zipfile.ZipFile(destination) as archive:assert archive.testzip() is None
    dump(HERE/'qa/package.json',dict(time_utc=now(),path=destination.name,files=len(selected)+2,bytes=destination.stat().st_size,sha256=sha(destination),zip_integrity='PASS',exclusions=['large vectors.npy caches','Python bytecode','redundant raster QA intermediates']))
    print(destination.name,destination.stat().st_size,'bytes;',len(selected)+2,'files; ZIP integrity PASS')

if __name__=='__main__':main()
