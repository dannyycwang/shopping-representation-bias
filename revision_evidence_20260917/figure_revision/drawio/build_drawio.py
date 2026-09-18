"""Render the existing seven figures as native, individually editable mxGraph cells.

Uses the original plotting functions and frozen inputs without saving/replacing
their PDF/SVG files. Matplotlib's draw commands become rectangles, ellipses,
triangles, diamonds, polylines and plain text, never a flattened image.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT / 'scripts'))
import plot_revision as plots
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import to_hex
from matplotlib.path import Path as MplPath

ORDER = ['figure1_teal_chair_candidate', 'figure2_intervention_and_states', 'rq1_vi_top20_main',
         'rq1_states_top20_top100_appendix', 'rq2_membership_top20_main',
         'rq2_membership_top100_appendix', 'rq2_cancellation_top20_table']
NAMES = ['Figure 1 candidate', 'Figure 2 schematic', 'RQ1 Top-20 VI', 'RQ1 appendix states',
         'RQ2 Top-20', 'RQ2 Top-100', 'RQ2 cancellation table']
REPORT = {}
DIAGRAMS = {}

def num(x):
    return f'{float(x):.5f}'.rstrip('0').rstrip('.') or '0'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def color(c):
    return 'none' if c is None or (len(c) == 4 and c[3] == 0) else to_hex(c)

class DrawioRenderer(RendererBase):
    def __init__(self, fig, name):
        super().__init__()
        self.fig, self.name = fig, name
        self.dpi = fig.dpi
        self.w, self.h = fig.bbox.width, fig.bbox.height
        self.factor = 96 / self.dpi
        self.agg = RendererAgg(self.w, self.h, self.dpi)
        self.diagram = ET.Element('diagram', id=name, name=NAMES[ORDER.index(name)])
        self.model = ET.SubElement(self.diagram, 'mxGraphModel', dx='0', dy='0', grid='0', gridSize='1',
            guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1',
            pageWidth=num(self.w*self.factor), pageHeight=num(self.h*self.factor), math='0', shadow='0',
            background='#ffffff')
        self.root = ET.SubElement(self.model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.counter = 2
        self.counts = dict(text=0, rectangle=0, ellipse=0, triangle=0, rhombus=0, polyline=0)
        self.strings = []

    def get_canvas_width_height(self):
        return self.w, self.h

    def points_to_pixels(self, points):
        return points * self.dpi / 72

    def get_text_width_height_descent(self, s, prop, ismath):
        return self.agg.get_text_width_height_descent(s, prop, ismath)

    def draw_image(self, *args, **kwargs):
        raise AssertionError('Raster images are not permitted in editable figures')

    def cell(self, style, *, value='', bounds=None, points=None, kind='rectangle'):
        attrs = dict(id=f'{self.name}-{self.counter}', value=value, style=style, parent='1')
        self.counter += 1
        if points is None:
            attrs.update(vertex='1', connectable='0')
            cell = ET.SubElement(self.root, 'mxCell', **attrs)
            x, y, w, h = bounds
            ET.SubElement(cell, 'mxGeometry', {'x': num(x), 'y': num(y), 'width': num(w), 'height': num(h), 'as': 'geometry'})
        else:
            attrs['edge'] = '1'
            cell = ET.SubElement(self.root, 'mxCell', **attrs)
            geo = ET.SubElement(cell, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
            for pos, endpoint in [(points[0], 'sourcePoint'), (points[-1], 'targetPoint')]:
                ET.SubElement(geo, 'mxPoint', {'x': num(pos[0]), 'y': num(pos[1]), 'as': endpoint})
            if len(points) > 2:
                array = ET.SubElement(geo, 'Array', {'as': 'points'})
                for x, y in points[1:-1]:
                    ET.SubElement(array, 'mxPoint', x=num(x), y=num(y))
        self.counts[kind] += 1
        return cell

    def draw_path(self, gc, path, transform, rgbFace=None):
        stroke = color(gc.get_rgb()) if gc.get_linewidth() > 0 else 'none'
        fill = color(rgbFace)
        if stroke == fill == 'none':
            return
        line_width = gc.get_linewidth()*96/72
        alpha = gc.get_alpha()
        style = f'html=0;rounded=0;shadow=0;strokeColor={stroke};fillColor={fill};strokeWidth={num(line_width)};opacity={num(alpha*100)};'
        dashed = gc.get_dashes()
        if dashed[1] is not None:
            style += 'dashed=1;dashPattern=' + ' '.join(num(x) for x in dashed[1]) + ';'
        subpaths, current = [], []
        for vertices, code in path.iter_segments(transform, curves=False, simplify=False):
            if code == MplPath.MOVETO:
                if current:
                    subpaths.append((current, False))
                current = [vertices[-2:]]
            elif code == MplPath.CLOSEPOLY:
                subpaths.append((current, True))
                current = []
            elif code == MplPath.LINETO:
                current.append(vertices[-2:])
            else:
                raise AssertionError(('Unhandled path code', code))
        if current:
            subpaths.append((current, False))
        for points, closed in subpaths:
            if len(points) < 2:
                continue
            p = np.asarray(points, dtype=float)
            p = np.column_stack([p[:, 0], self.h-p[:, 1]])*self.factor
            if not np.isfinite(p).all():
                continue
            if np.allclose(p[0], p[-1]):
                p, closed = p[:-1], True
            low, high = p.min(axis=0), p.max(axis=0)
            w, h = high-low
            bounds = (*low, w, h)
            shape = None
            if closed and len(p) == 4 and np.unique(p[:, 0].round(6)).size == 2 and np.unique(p[:, 1].round(6)).size == 2:
                shape = 'rectangle'
            elif closed and len(p) == 4:
                shape = 'rhombus'
            elif closed and len(p) == 3:
                shape = 'triangle'
            elif closed and path.codes is not None and (path.codes == MplPath.CURVE4).any() and abs(w-h) < .01:
                shape = 'ellipse'
            if shape:
                # Suppress the full-page background; page background stays white.
                if shape == 'rectangle' and fill == '#ffffff' and stroke == 'none' and w >= self.w*self.factor-.01 and h >= self.h*self.factor-.01:
                    continue
                extra = 'direction=north;' if shape == 'triangle' else ''
                self.cell(f'shape={shape};{extra}'+style, bounds=bounds, kind=shape)
            else:
                assert not closed or fill == 'none', ('Unexpected filled polygon', self.name, p)
                if closed:
                    p = np.vstack([p, p[0]])
                self.cell('edgeStyle=none;endArrow=none;startArrow=none;noEdgeStyle=1;'+style, points=p, kind='polyline')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        assert not ismath and angle == 0, ('Unsupported text transform', s)
        # RendererBase.flipy=True: Matplotlib already supplies the top-down baseline.
        size = prop.get_size_in_points()*96/72
        width = self.get_text_width_height_descent(s, prop, False)[0]*self.factor
        weight = prop.get_weight()
        bold = weight in ['bold', 'semibold', 'heavy', 'black'] or (isinstance(weight, (int, float)) and weight >= 600)
        italic = prop.get_style() in ['italic', 'oblique']
        fontstyle = int(bold) + 2*int(italic)
        style = ('text;html=0;align=left;verticalAlign=top;whiteSpace=nowrap;overflow=visible;'
                 'spacing=0;spacingTop=0;spacingLeft=0;spacingRight=0;spacingBottom=0;'
                 'strokeColor=none;fillColor=none;fontFamily=Times New Roman;'
                 f'fontSize={num(size)};fontStyle={fontstyle};fontColor={color(gc.get_rgb())};')
        # Top-aligned native draw.io text adds a five-unit shape inset before
        # mxSvgCanvas2D's baseline offset (fontSize - 1). Cancel both offsets.
        self.cell(style, value=s, bounds=(x*self.factor, y*self.factor-size-4, max(width+.5, 1), size*1.25), kind='text')
        self.strings.append(s)

def export_drawio(fig, name):
    fig.canvas.draw()
    renderer = DrawioRenderer(fig, name)
    fig.draw(renderer)
    DIAGRAMS[name] = renderer.diagram
    REPORT[name] = dict(width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
                        native_objects=sum(renderer.counts.values()), counts=renderer.counts,
                        minimum_text_size_pt=min(float(c.get('style').split('fontSize=')[1].split(';')[0])*72/96
                                                 for c in renderer.root if 'fontSize=' in c.get('style', '')),
                        text_strings=renderer.strings)
    plots.plt.close(fig)

def save(path, diagrams):
    root = ET.Element('mxfile', host='app.diagrams.net', type='device', version='29.0.0', compressed='false')
    for diagram in diagrams:
        root.append(copy.deepcopy(diagram))
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def main():
    # Freeze prior committed deliverables, including PDF/SVG/data/ZIP, before drawing.
    preserved = {p: sha(p) for p in PARENT.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts and 'qa' not in p.parts}
    plots.plt.rcParams['font.family'] = 'Times New Roman'
    plots.export = export_drawio
    for function in [plots.schematic, plots.rq1_main, plots.rq1_appendix, lambda: plots.rq2(20), lambda: plots.rq2(100), plots.cancellation, plots.case_candidate]:
        function()
    for name in ORDER:
        save(OUT / f'{name}.drawio', [DIAGRAMS[name]])
    save(OUT / 'all_seven_figures.drawio', [DIAGRAMS[name] for name in ORDER])
    for p, digest in preserved.items():
        assert sha(p) == digest, ('Original changed', p)
    for name, report in REPORT.items():
        assert report['minimum_text_size_pt'] >= 7.99999
        root = ET.parse(OUT / f'{name}.drawio').getroot()
        cells = root.findall('.//mxCell')
        assert len({c.get('id') for c in cells}) == len(cells)
        assert all('image=' not in c.get('style', '') and 'shape=image' not in c.get('style', '') for c in cells)
        assert [c.get('value') for c in cells if c.get('value')] == report['text_strings']
    (OUT/'qa').mkdir(exist_ok=True)
    (OUT/'qa/build_validation.json').write_text(json.dumps({'status': 'PASS', 'original_files_preserved': len(preserved),
        'source_plot_script_sha256': sha(PARENT/'scripts/plot_revision.py'), 'font': 'Times New Roman',
        'coordinate_units': '96 draw.io pixels per inch; point sizes converted by 96/72',
        'diagrams': REPORT}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Created seven individually editable draw.io files and one seven-page collection.')
    print({name: r['native_objects'] for name, r in REPORT.items()})

if __name__ == '__main__':
    main()
