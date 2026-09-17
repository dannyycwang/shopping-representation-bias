"""Package final revision files, excluding local proofs and the archive itself."""
import zipfile
from io_common import HERE, dump, sha, load

def main():
    report = load(HERE / 'qa/validation_report.json')
    assert report['status'] == 'PASS'
    assert (HERE / 'qa/VISUAL_QA.md').exists()
    exclude = {'figure_revision.zip', 'DELIVERY_MANIFEST.json'}
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in exclude
                   and '__pycache__' not in p.parts and not (p.parent == HERE / 'qa' and p.suffix not in ['.json', '.md']))
    dump(HERE / 'DELIVERY_MANIFEST.json', {'format': 'SHA256', 'files': [
        {'path': p.relative_to(HERE).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]})
    with zipfile.ZipFile(HERE / 'figure_revision.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files + [HERE / 'DELIVERY_MANIFEST.json']:
            z.write(p, arcname=p.relative_to(HERE).as_posix())
    with zipfile.ZipFile(HERE / 'figure_revision.zip') as z:
        assert z.testzip() is None
        for item in load(HERE / 'DELIVERY_MANIFEST.json')['files']:
            import hashlib
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
    print('Packaged', len(files), 'manifested files; archive bytes:', (HERE/'figure_revision.zip').stat().st_size)

if __name__ == '__main__':
    main()
