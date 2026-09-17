"""Seven vector figures at final manuscript widths; read only frozen local inputs."""
import textwrap
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from io_common import HERE, DATA, FIG, STEMS, MODELS, NAMES, dump, load

for f in (HERE / 'assets/fonts').glob('*.ttf'):
    fm.fontManager.addfont(str(f))
FAMILY = fm.FontProperties(fname=str(HERE / 'assets/fonts/ManuscriptLibertine-R.ttf')).get_name()
plt.rcParams.update({'font.family': FAMILY, 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'pdf.fonttype': 42, 'svg.fonttype': 'path', 'svg.hashsalt': 'figure-revision-20260917',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'savefig.facecolor': 'white', 'text.color': '#202832', 'axes.labelcolor': '#202832'})
INK, GAIN, LOSS, MUTED, BLUE = '#202832', '#187C80', '#B24C36', '#D8DFE5', '#3C6589'
CAPTIONS = load(DATA / 'captions.json')
GEOMETRY = []

def export(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for t in visible:
        b = t.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.x1 + .5 or b.y1 > fig.bbox.y1 + .5:
            outside.append(t.get_text())
    assert not outside, (name, 'Text outside page', outside)
    minimum = min(t.get_fontsize() for t in visible)
    assert minimum >= 8, (name, minimum)
    fig.savefig(FIG / f'{name}.pdf', metadata={'Title': CAPTIONS[name]['short'], 'Author': 'Yu-Chung Wang',
                    'Subject': CAPTIONS[name]['description'], 'CreationDate': None, 'ModDate': None})
    path = FIG / f'{name}.svg'
    fig.savefig(path)
    ns = '{http://www.w3.org/2000/svg}'
    ET.register_namespace('', ns[1:-1])
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(path)
    root = tree.getroot()
    definitions = ET.Element(ns + 'defs')
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == ns + 'defs':
                definitions.extend(list(child))
                parent.remove(child)
    title, desc = ET.Element(ns + 'title', {'id': 'figure-title'}), ET.Element(ns + 'desc', {'id': 'figure-description'})
    title.text, desc.text = CAPTIONS[name]['short'], CAPTIONS[name]['description']
    root.insert(0, definitions)
    root.insert(0, desc)
    root.insert(0, title)
    root.set('role', 'img')
    root.set('aria-labelledby', 'figure-title figure-description')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    GEOMETRY.append(dict(name=name, width_inches=float(fig.get_figwidth()), height_inches=float(fig.get_figheight()),
                         minimum_font_pt=minimum, text_outside_page=outside, vector_only=True))
    plt.close(fig)

def schematic():
    data = load(DATA / 'schematic.json')
    fig = plt.figure(figsize=(7.1, 3.35))
    ax = fig.add_axes([.025, .04, .95, .94])
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ax.text(0, 96, 'A  Attribute-order interventions', fontsize=10.5, weight='bold')
    colors = dict(zip('ABCDEFGHIJ', ['#BBDDE0', '#F0D2BA', '#D4CCEA'] * 3 + ['#BBDDE0']))
    def blocks(x, y, entries):
        for i, entry in enumerate(entries):
            ax.add_patch(Rectangle((x + 4.1*i, y - 2.9), 3.55, 5.8, facecolor=colors[entry], edgecolor=INK, lw=.5))
            ax.text(x + 4.1*i + 1.775, y, entry, ha='center', va='center', fontsize=9)
    for left, targetonly in [(0, False), (52, True)]:
        ax.text(left, 86, 'Target-only intervention' if targetonly else 'Catalog-wide intervention', weight='bold', fontsize=10)
        ax.text(left + 17, 78, 'Raw C0', ha='center', fontsize=9)
        ax.text(left + 39, 78, 'Permuted order', ha='center', fontsize=9)
        after = data['target_only' if targetonly else 'catalog_wide']
        for i, (before, new) in enumerate(zip(data['raw'], after)):
            y = 70 - 8.5*i
            ax.text(left, y, ['Target p', 'Other u', 'Other v'][i], va='center', fontsize=8.5)
            blocks(left + 11, y, before)
            blocks(left + 33, y, new)
            if targetonly and i > 0:
                ax.text(left + 28, y, '=', ha='center', va='center', fontsize=10)
            else:
                ax.annotate('', xy=(left + 31.6, y), xytext=(left + 24.3, y), arrowprops={'arrowstyle': '->', 'lw': .7, 'color': INK})
        ax.text(left + 26, 44, 'Competitors: raw C0 (fixed)' if targetonly else 'All products reordered', ha='center', fontsize=9)
    ax.plot([0, 100], [39, 39], color='#CBD1D5', lw=.6)
    ax.text(0, 32, 'B  Inclusion states', fontsize=10.5, weight='bold')
    for i in range(3):
        ax.text(42 + 12*i, 30, f'Order {i+1}', ha='center', fontsize=8.5)
    for row, (label, values) in enumerate(data['states'].items()):
        y = 22 - 8*row
        ax.text(0, y, label, va='center', fontsize=9)
        for i, v in enumerate(values):
            x = 42 + 12*i
            ax.add_patch(Rectangle((x - 2.5, y - 3.1), 5, 6.2, facecolor=GAIN if v else 'white', edgecolor=GAIN if v else '#8C969E', lw=.6))
            ax.text(x, y, str(v), va='center', ha='center', fontsize=9, color='white' if v else INK)
    ax.text(79, 14, '1: rank ≤ K\n0: rank > K', fontsize=9, va='center', linespacing=1.6)
    export(fig, 'figure2_intervention_and_states')

def rq1_main():
    data = pd.read_csv(DATA / 'rq1_main_top20.csv')
    fig = plt.figure(figsize=(7.1, 3.25))
    handles = [Line2D([], [], color=BLUE, marker='o', ls='none', ms=5, label='Full'),
               Line2D([], [], color=LOSS, marker='^', mfc='white', ls='none', ms=5.5, label='Fully-fitting')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .995), ncol=2, frameon=False, fontsize=9)
    for i, ds in enumerate(['wands', 'esci']):
        ax = fig.add_axes([.085 + .50*i, .22, .205, .62])
        ax.set_title(ds.upper(), loc='left', pad=7, weight='bold')
        ax.set(xlim=(0, 25 if ds == 'wands' else 8), ylim=(7.1, -.7))
        ax.set_xticks([0, 10, 20] if ds == 'wands' else [0, 2, 4, 6, 8])
        ax.set_yticks([.5, 3.3, 6.1], ['MiniLM', 'BGE', 'GTE*'])
        ax.tick_params(axis='y', length=0, pad=4)
        ax.set_xlabel('VI (%)', labelpad=3)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#E3E7EA', lw=.6)
        ax.set_axisbelow(True)
        ax.text(1.38, 1.045, 'VI (%)', transform=ax.transAxes, ha='right', fontsize=8.5, weight='bold')
        ax.text(1.93, 1.045, 'q / pairs', transform=ax.transAxes, ha='right', fontsize=8.5, weight='bold')
        for j, model in enumerate(MODELS):
            for t, support in enumerate(['full', 'fully_fitting']):
                r = data[(data.dataset == ds) & (data.model == model) & (data.target_support == support)].iloc[0]
                y = 2.8*j + t
                color = BLUE if t == 0 else LOSS
                ax.errorbar(100*r.VI, y, xerr=[[100*(r.VI-r.VI_ci_low)], [100*(r.VI_ci_high-r.VI)]],
                            fmt='o' if t == 0 else '^', ms=4.5 if t == 0 else 5, mfc=color if t == 0 else 'white',
                            mec=color, ecolor=color, capsize=2, lw=.85, zorder=4)
                ax.text(1.38, y, f'{100*r.VI:.2f}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.5, color=color)
                ax.text(1.93, y, f'{int(r.eligible_queries):,} / {int(r.highest_relevance_pairs):,}',
                        transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.5)
    fig.text(.50, .065, '* GTE: same full / fully-fitting support; points separated vertically.', ha='center', fontsize=8.5)
    export(fig, 'rq1_vi_top20_main')

def rq1_appendix():
    data = pd.read_csv(DATA / 'rq1_target_fully_fitting_plot_data.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5))
    fig.subplots_adjust(left=.122, right=.987, bottom=.10, top=.89, wspace=.45, hspace=.44)
    for row, ds in enumerate(['wands', 'esci']):
        for col, k in enumerate([20, 100]):
            ax = axes[row, col]
            labels = []
            for j, model in enumerate(MODELS):
                for t, support in enumerate(['full', 'fully_fitting']):
                    r = data[(data.dataset == ds) & (data.model == model) & (data.K == k) & (data.target_support == support)].iloc[0]
                    y = 2.4*j + t
                    labels.append((y, f'{NAMES[model]} {"full" if t == 0 else "fitting"}'))
                    left = 0
                    for state, color in [('Always', GAIN), ('VI', LOSS), ('Never', MUTED)]:
                        val = float(r[state])*100
                        ax.barh(y, val, left=left, height=.71, color=color, lw=0)
                        if state != 'VI' and val >= 17:
                            ax.text(left + val/2, y, f'{val:.1f}', ha='center', va='center', fontsize=8, color='white' if state == 'Always' else INK)
                        left += val
                    ax.text(123, y, f'{100*r.VI:.2f}', ha='right', va='center', color=LOSS, fontsize=8.3)
                    ax.text(167, y, f'{int(r.eligible_queries):,} / {int(r.highest_relevance_pairs):,}', ha='right', va='center', fontsize=8.3)
            ax.set(xlim=(0, 169), ylim=(6.6, -.6))
            ax.set_yticks([x[0] for x in labels], [x[1] for x in labels], fontsize=8.5)
            ax.set_xticks([0, 50, 100])
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_bounds(0, 100)
            ax.tick_params(axis='y', length=0)
            ax.text(123, 1.025, 'VI (%)', ha='right', transform=ax.get_xaxis_transform(), weight='bold', fontsize=8.3)
            ax.text(167, 1.025, 'q / pairs', ha='right', transform=ax.get_xaxis_transform(), weight='bold', fontsize=8.3)
            ax.set_title(f'{ds.upper()}  |  Top-{k}', loc='left', pad=21, weight='bold')
            if row == 1:
                ax.set_xlabel('Highest-relevance target share (%)', labelpad=3)
    fig.legend(handles=[Patch(color=c, label=s) for s, c in [('Always', GAIN), ('VI', LOSS), ('Never', MUTED)]],
               loc='upper center', bbox_to_anchor=(.5, 1), ncol=3, frameon=False, fontsize=9)
    export(fig, 'rq1_states_top20_top100_appendix')

def rq2(k):
    data = pd.read_csv(DATA / f'rq2_top{k}.csv')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4))
    fig.subplots_adjust(left=.07, right=.985, bottom=.12, top=.845, wspace=.32, hspace=.48)
    for i, ds in enumerate(['wands', 'esci']):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            d = data[(data.dataset == ds) & (data.model == model)].set_index('alternative').loc[STEMS[1:]]
            y = np.arange(6)
            ax.barh(y, -100*d.loss_fraction, height=.63, color=LOSS, lw=0)
            ax.barh(y, 100*d.gain_fraction, height=.63, color=GAIN, lw=0)
            ax.errorbar(100*d.net_delta, y, xerr=np.stack([100*(d.net_delta-d.net_delta_ci_low), 100*(d.net_delta_ci_high-d.net_delta)]),
                        fmt='D', ms=3.5, color='black', lw=.85, capsize=2, zorder=4)
            ax.axvline(0, color=INK, lw=.65)
            ax.set(xlim=(-7, 7), ylim=(5.65, -.65))
            ax.set_yticks(y, STEMS[1:])
            ax.set_xticks([-6, -3, 0, 3, 6])
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)
            ax.set_title(f'{ds.upper()} / {NAMES[model]}', loc='left', weight='bold', pad=6)
            ax.grid(axis='x', color='#E3E7EA', lw=.5)
            ax.set_axisbelow(True)
    fig.text(.07, .97, f'Top-{k}', weight='bold', fontsize=10)
    handles = [Patch(color=LOSS, label='Omitted share (%)'), Patch(color=GAIN, label='Included share (%)'),
               Line2D([], [], color='black', marker='D', ms=3.5, lw=.8, label='ΔRecall (pp), 95% CI')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, 1), ncol=3, frameon=False, fontsize=8.8, handlelength=1.5, columnspacing=1.2)
    fig.text(.5, .025, 'Relevant-product share (%) and ΔRecall (percentage points)', ha='center', fontsize=9)
    export(fig, f'rq2_membership_top{k}_{"main" if k == 20 else "appendix"}')

def cancellation():
    data = pd.read_csv(DATA / 'cancellation_main_top20.csv')
    fig = plt.figure(figsize=(3.45, 2.1))
    ax = fig.add_axes([.025, .055, .95, .91])
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ax.text(0, 96, 'Exact within-query cancellation at Top-20', fontsize=10, weight='bold')
    xs = [0, 28, 66, 100]
    for x, label, align in zip(xs, ['Dataset', 'Encoder', 'Eligible q', 'Queries (%)'], ['left', 'left', 'right', 'right']):
        ax.text(x, 80, label, ha=align, fontsize=8.5, weight='bold')
    ax.plot([0, 100], [76, 76], color=INK, lw=.6)
    for i, r in data.iterrows():
        y = 67 - 9*i
        for x, label, align in zip(xs, [r.dataset.upper(), NAMES[r.model], str(r.eligible_queries), f'{r.percent_min:.2f}-{r.percent_max:.2f}'], ['left', 'left', 'right', 'right']):
            ax.text(x, y, label, ha=align, va='center', fontsize=8.5)
        if i == 2:
            ax.plot([0, 100], [44, 44], color='#C6CCD1', lw=.5)
    ax.plot([0, 100], [17, 17], color=INK, lw=.6)
    ax.text(0, 5, 'Six-comparison min-max; not a confidence interval.', fontsize=8)
    export(fig, 'rq2_cancellation_top20_table')

def case_candidate():
    data = load(DATA / 'figure1_case_inputs.json')
    audit = load(DATA / 'figure1_case_audit.json')
    assert audit['status'] == 'PASS'
    fig = plt.figure(figsize=(3.45, 3.9))
    ax = fig.add_axes([.035, .025, .93, .95])
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ax.text(50, 97, 'Query: "teal chair"', ha='center', fontsize=11, weight='bold')
    ax.text(50, 91, 'Same product  |  GTE  |  Exact relevance', ha='center', fontsize=9)
    colors = {1: '#D8E9EA', 2: '#F2DFC9', 3: '#E1D9ED', 4: '#DCE4EF'}
    for x, s, rank, state, color in [(0, 'C0', 11, 'Inside Top-20', GAIN), (53, 'C2s5', 939, 'Outside Top-20', LOSS)]:
        ax.add_patch(Rectangle((x, 21), 47, 65, facecolor='#F8F9FA', edgecolor='#C5CCD1', lw=.6))
        ax.text(x + 23.5, 81, s, ha='center', weight='bold', fontsize=10)
        ax.text(x + 23.5, 73.5, f'Rank {rank}', ha='center', fontsize=12, weight='bold', color=color)
        ax.text(x + 23.5, 67.5, state, ha='center', fontsize=9, color=color)
        for j, entry in enumerate(data['display_excerpt'][s]):
            y = 57 - 10.4*j
            ax.add_patch(Rectangle((x + 2, y - 5), 43, 9.7, facecolor=colors[entry['original_entry_id']], edgecolor='white', lw=.5))
            # Line breaks change layout only; the complete entry text is retained.
            text = textwrap.fill(entry['text'], width=24, break_long_words=False, break_on_hyphens=False)
            if len(text) > 28 and '\n' not in text:
                text = text.replace(':', ':\n', 1)
            ax.text(x + 23.5, y, text, ha='center', va='center', fontsize=8, linespacing=1.08)
    ax.text(50, 16.5, f'Display excerpt: 4 of {audit["attribute_entry_count"]} attribute entries.', ha='center', fontsize=8.5)
    ax.text(50, 11.2, 'Full input retained: 1,059 / 8,192 tokens.', ha='center', fontsize=8.5, weight='bold')
    ax.text(50, 5.9, 'Complete record encoded in both conditions.', ha='center', fontsize=8.5)
    ax.text(50, .8, 'Post-hoc illustrative case', ha='center', fontsize=8, style='italic')
    export(fig, 'figure1_teal_chair_candidate')

def main():
    schematic()
    rq1_main()
    rq1_appendix()
    rq2(20)
    rq2(100)
    cancellation()
    case_candidate()
    dump(HERE / 'qa/figure_geometry.json', GEOMETRY)
    print('Exported seven single-page vector PDFs and seven SVGs; minimum text size >= 8 pt.')

if __name__ == '__main__':
    main()
