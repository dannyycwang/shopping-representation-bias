"""Check the actual draw.io model round-trip, then package native editable files."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def semantic(element):
    children = list(element)
    if element.tag == 'mxGeometry':
        # Geometry properties are named by `as`; the codec may reorder them.
        # Waypoint order inside Array remains significant and is not sorted.
        children.sort(key=lambda c: (c.tag, c.get('as', '')))
    return (element.tag, dict(element.attrib), [semantic(c) for c in children])

def main():
    report = json.loads((HERE/'qa/build_validation.json').read_text(encoding='utf-8'))
    combined = ET.parse(HERE/'all_seven_figures.drawio').getroot()
    assert len(combined.findall('diagram')) == 7
    checks = []
    for name, info in report['diagrams'].items():
        original = ET.parse(HERE/(name+'.drawio')).getroot()
        roundtrip = ET.parse(HERE/'qa'/(name+'_roundtrip.drawio')).getroot()
        a = {c.get('id'): c for c in original.findall('.//mxCell')}
        b = {c.get('id'): c for c in roundtrip.findall('.//mxCell')}
        assert a.keys() == b.keys(), name
        for key, c in a.items():
            d = b[key]
            assert c.get('value', '') == d.get('value', ''), (name, key, 'text')
            assert c.get('style', '') == d.get('style', ''), (name, key, 'style')
            g, h = c.find('mxGeometry'), d.find('mxGeometry')
            if g is not None:
                assert semantic(g) == semantic(h), (name, key, 'geometry')
        single = original.find('diagram')
        page = next(x for x in combined.findall('diagram') if x.get('id') == name)
        assert semantic(single) == semantic(page), (name, 'combined-page equality')
        svg = ET.parse(HERE/'qa'/(name+'.svg')).getroot()
        assert not any(x.tag.endswith('image') for x in svg.iter()), 'Unexpected raster image'
        assert (HERE/'qa'/(name+'.png')).exists()
        checks.append(dict(name=name, native_objects=info['native_objects'], editable_labels=info['counts']['text'],
                           text_styles_and_geometry_roundtrip_identical=True, combined_page_identical=True,
                           diagram_sha256=sha(HERE/(name+'.drawio')), proof_png_sha256=sha(HERE/'qa'/(name+'.png'))))
    old_manifest = json.loads((HERE.parent/'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    for item in old_manifest['files']:
        assert sha(HERE.parent/item['path']) == item['sha256'], ('Original delivery changed', item['path'])
    validation = dict(status='PASS', actual_renderer='Official draw.io GraphViewer from viewer.diagrams.net',
                      roundtrip='Decoded and re-encoded by the draw.io graph model, then compared semantically.',
                      rendering_scale=1.5625, proof_dpi=150, proofs_crop_outer_whitespace=True,
                      native_objects=sum(x['native_objects'] for x in checks), original_manifest_files_unchanged=len(old_manifest['files']), figures=checks)
    (HERE/'qa/app_validation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8')
    files = sorted([p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.drawio', '.py', '.md']] +
                   list((HERE/'qa').glob('*.json')) + list((HERE/'qa').glob('*.md')) + [HERE/'.gitignore'])
    manifest = dict(files=[dict(path=p.relative_to(HERE).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    (HERE/'DRAWIO_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(HERE/'editable_drawio_figures.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files+[HERE/'DRAWIO_MANIFEST.json']:
            z.write(p, p.relative_to(HERE).as_posix())
    with zipfile.ZipFile(HERE/'editable_drawio_figures.zip') as z:
        assert z.testzip() is None
        for item in manifest['files']:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
    print(f'PASS: all seven diagrams round-trip without text, style, or geometry changes; {validation["native_objects"]} native objects.')
    print('ZIP:', (HERE/'editable_drawio_figures.zip').stat().st_size, 'bytes')

if __name__ == '__main__':
    main()
