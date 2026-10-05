"""Export one auditable vector scene to PDF, SVG and editable draw.io; render PDF to PNG."""
from pathlib import Path
import argparse, collections, html, json, math, subprocess, textwrap, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from run_teal_captum import read,dump,writecsv,sha
for name,filename in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/filename)))
INK='#20343D';MUTED='#536872';TEAL='#087E8B';RED='#AF4359';GRID='#D7E1E5';BG='#F3F7F8'

class Scene:
    def __init__(self,name,w,h):self.name=name;self.w=w;self.h=h;self.items=[]
    def text(self,x,y,value,size=18,color=INK,bold=False,width=None,role='',entry_id=''):
        font='Arial-Bold' if bold else 'Arial'
        value=str(value);lines=value.split('\n');height=len(lines)*size*1.2
        measured=max(pdfmetrics.stringWidth(v,font,size) for v in lines)
        if width is not None: assert measured<=width+1,(value,measured,width)
        self.items.append(dict(kind='text',x=x,y=y,text=value,size=size,color=color,bold=bold,
                        w=width or measured+2,h=height,role=role,entry_id=entry_id))
    def line(self,x,y,x2,y2,color=GRID,width=1):self.items.append(dict(kind='line',x=x,y=y,x2=x2,y2=y2,color=color,width=width))
    def rect(self,x,y,w,h,color=BG):self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
    def dot(self,x,y,r=5,color=TEAL):self.items.append(dict(kind='dot',x=x,y=y,r=r,color=color))

def export(out,stem,pages):
    figs=out/'figures';figs.mkdir(exist_ok=True)
    c=canvas.Canvas(str(figs/f'{stem}.pdf'));c.setTitle('Revisiting the Figure 1 teal-chair case')
    mx=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device','version':'26.0.0'})
    checks=[]
    for pnum,p in enumerate(pages,1):
        c.setPageSize((p.w*.6,p.h*.6));c.saveState();c.scale(.6,.6)
        svg=ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg','width':str(p.w),'height':str(p.h),'viewBox':f'0 0 {p.w} {p.h}'})
        ET.SubElement(svg,'rect',{'width':'100%','height':'100%','fill':'white'})
        dia=ET.SubElement(mx,'diagram',{'id':f'{stem}_{pnum}','name':p.name})
        model=ET.SubElement(dia,'mxGraphModel',{'dx':str(p.w),'dy':str(p.h),'grid':'0','page':'1','pageScale':'1','pageWidth':str(p.w),'pageHeight':str(p.h),'math':'0','shadow':'0'})
        root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
        for i,a in enumerate(p.items):
            kind=a['kind'];color=a['color'];c.setFillColor(color);c.setStrokeColor(color)
            attr={'id':str(i+2),'parent':'1'}
            if kind=='text':
                c.setFont('Arial-Bold' if a['bold'] else 'Arial',a['size'])
                st=ET.SubElement(svg,'text',{'x':str(a['x']),'y':str(a['y']+a['size']),'font-family':'Arial','font-size':str(a['size']),
                    'font-weight':'700' if a['bold'] else '400','fill':color,'data-entry-id':a['entry_id'],'data-role':a['role']})
                for li,line in enumerate(a['text'].split('\n')):
                    c.drawString(a['x'],p.h-a['y']-a['size']-li*a['size']*1.2,line)
                    ET.SubElement(st,'tspan',{'x':str(a['x']),'y':str(a['y']+a['size']+li*a['size']*1.2)}).text=line
                attr.update(vertex='1',value='<div style="line-height:1.2">'+html.escape(a['text']).replace('\n','<br>')+'</div>',
                    style=f'text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;whiteSpace=nowrap;overflow=visible;spacing=0;fontFamily=Arial;fontSize={a["size"]};fontColor={color};fontStyle={1 if a["bold"] else 0};')
                cell=ET.SubElement(root,'mxCell',attr);ET.SubElement(cell,'mxGeometry',{'x':str(a['x']),'y':str(a['y']),'width':str(a['w']),'height':str(a['h']),'as':'geometry'})
                assert a['x']>=0 and a['y']>=0 and a['x']+a['w']<=p.w and a['y']+a['h']<=p.h
            elif kind=='rect':
                c.rect(a['x'],p.h-a['y']-a['h'],a['w'],a['h'],stroke=0,fill=1)
                ET.SubElement(svg,'rect',{'x':str(a['x']),'y':str(a['y']),'width':str(a['w']),'height':str(a['h']),'fill':color})
                attr.update(vertex='1',value='',style=f'rounded=0;fillColor={color};strokeColor=none;')
                cell=ET.SubElement(root,'mxCell',attr);ET.SubElement(cell,'mxGeometry',{'x':str(a['x']),'y':str(a['y']),'width':str(a['w']),'height':str(a['h']),'as':'geometry'})
            elif kind=='line':
                c.setLineWidth(a['width']);c.line(a['x'],p.h-a['y'],a['x2'],p.h-a['y2'])
                ET.SubElement(svg,'line',{'x1':str(a['x']),'y1':str(a['y']),'x2':str(a['x2']),'y2':str(a['y2']),'stroke':color,'stroke-width':str(a['width'])})
                attr.update(edge='1',value='',style=f'endArrow=none;startArrow=none;strokeColor={color};strokeWidth={a["width"]};')
                cell=ET.SubElement(root,'mxCell',attr);geo=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
                ET.SubElement(geo,'mxPoint',{'x':str(a['x']),'y':str(a['y']),'as':'sourcePoint'});ET.SubElement(geo,'mxPoint',{'x':str(a['x2']),'y':str(a['y2']),'as':'targetPoint'})
            else:
                c.circle(a['x'],p.h-a['y'],a['r'],stroke=0,fill=1)
                ET.SubElement(svg,'circle',{'cx':str(a['x']),'cy':str(a['y']),'r':str(a['r']),'fill':color})
                attr.update(vertex='1',value='',style=f'ellipse;fillColor={color};strokeColor=none;')
                cell=ET.SubElement(root,'mxCell',attr);ET.SubElement(cell,'mxGeometry',{'x':str(a['x']-a['r']),'y':str(a['y']-a['r']),'width':str(2*a['r']),'height':str(2*a['r']),'as':'geometry'})
        c.restoreState();c.showPage()
        suffix='' if len(pages)==1 else f'_p{pnum:02d}'
        ET.ElementTree(svg).write(figs/f'{stem}{suffix}.svg',encoding='utf-8',xml_declaration=True)
        checks.append({'page':pnum,'name':p.name,'objects':len(p.items),'text_objects':sum(a['kind']=='text' for a in p.items),'bounds_passed':True})
    c.save();ET.ElementTree(mx).write(figs/f'{stem}.drawio',encoding='utf-8',xml_declaration=True)
    dump(out/'figure_data'/f'{stem}_scene.json',[{'name':p.name,'width':p.w,'height':p.h,'items':p.items} for p in pages])
    subprocess.run(['pdftoppm','-scale-to','1900','-png',str(figs/f'{stem}.pdf'),str(figs/stem)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    dump(out/'qa'/f'{stem}_export_checks.json',{'pages':checks,'drawio_native_objects':True,'embedded_images':0,'pdf_rendered':True})

def native_panel(rows):
    p=Scene('(a) Native candidate inclusion',1200,790)
    p.text(46,30,'Revisiting the Figure 1 teal-chair case',29,bold=True)
    p.text(46,79,'(a) Native candidate inclusion',24,bold=True)
    p.text(46,116,'WANDS q409 / product 24318 / GTE-ModernBERT / 42,993 fixed C0 competitors',17,color=MUTED)
    p.text(46,166,'Order',17,bold=True);p.text(228,166,'Native score - fixed Top-20 threshold',17,bold=True)
    p.text(860,166,'Native score',17,bold=True);p.text(1052,166,'Saved rank',17,bold=True)
    x=lambda v:245+(v+.065)/.08*550
    for tick in [-.06,-.04,-.02,0]:
        p.line(x(tick),208,x(tick),620,color=MUTED if tick==0 else GRID,width=2 if tick==0 else 1)
        p.text(x(tick)-23,636,f'{tick:+.2f}',15,color=MUTED)
    for i,r in enumerate(rows):
        y=236+i*57;end=r['schedule'] in ['C0','C2s5'];col=TEAL if r['inclusion'] else RED
        if end:p.rect(34,y-17,1125,43,'#EDF4F5')
        p.text(48,y-10,r['schedule']+(' *' if end else ''),21,bold=end)
        p.line(x(0),y+3,x(r['signed_margin']),y+3,col,2);p.dot(x(r['signed_margin']),y+3,7 if end else 5,col)
        p.text(860,y-10,f'{r["score"]:.9f}',19);p.text(1052,y-10,str(r['saved_rank']),21,bold=end,role='saved_rank',entry_id=r['schedule'])
    p.text(46,684,'* Attribution endpoints: C0 -> C2s5. Each row is a separate target-only replacement.',18,bold=True)
    p.text(46,719,f'Fixed threshold = {rows[0]["threshold"]:.12f}; competitor ID/index 22752. Exact FP32 comparisons.',17,color=MUTED)
    p.text(46,749,'Saved ranks are authoritative; fresh forward and gradient diagnostic are reported separately.',17,color=MUTED)
    return p

def label_wrap(row):
    key=row['entry_id']
    if key.startswith('fixed_'):return key[6:].capitalize()+' (fixed field)'
    if key.startswith('attr_'):return '\n'.join(textwrap.wrap(row['entry_text'],width=47,break_long_words=True,break_on_hyphens=False))
    return row['entry_text']

def attribute_page(rows,page_name,part,total,limit,report,compact=False):
    rh=68;h=290+len(rows)*rh+100;p=Scene(page_name,1320,h)
    p.text(42,28,'Revisiting the Figure 1 teal-chair case',28,bold=True)
    p.text(42,75,'(b) PAD attribution before/after reordering',24,bold=True)
    p.text(42,112,f'{page_name}  |  C0 -> C2s5  |  {report["baselines"]["PAD"]["nodes"]} Gauss-Legendre nodes  |  page {part}/{total}',17,color=MUTED)
    p.text(42,147,'Signed IG for the fixed-query cosine score; original C0 occurrence order. No per-order rescaling.',16,color=MUTED)
    p.text(42,188,'Attribute occurrence / fixed group',17,bold=True)
    p.text(475,185,'Attribute position\nC0 -> C2s5',16,bold=True)
    p.text(610,195,'IG(C0)',17,bold=True);p.text(751,195,'IG(C2s5)',17,bold=True)
    p.text(920,178,'IG(C2s5) - IG(C0)',17,bold=True);p.text(1167,195,'Difference',17,bold=True)
    x=lambda v:1040+v/limit*108
    for tick in [-limit,0,limit]:
        p.text(x(tick)-25,216,f'{tick:+.3f}',13,color=MUTED)
        p.line(x(tick),247,x(tick),258+len(rows)*rh,color=MUTED if tick==0 else GRID,width=1.3 if tick==0 else .6)
    for i,r in enumerate(rows):
        top=255+i*rh;y=top+rh/2-2;key=r['entry_id']
        if i%2==0:p.rect(32,top,1260,rh,'#F3F7F8')
        # Redraw axis in shaded rows; circles and all differences use one common scale.
        for tick in [-limit,0,limit]:p.line(x(tick),top,x(tick),top+rh,MUTED if tick==0 else GRID,.8)
        label=label_wrap(r);p.text(42,top+10,label,16,width=421,entry_id=key,role='label')
        pos='--' if pd.isna(r.get('C0_position')) else f'{int(r["C0_position"])} -> {int(r["C2s5_position"])}'
        p.text(486,top+22,pos,17,width=102,entry_id=key,role='position')
        p.text(610,top+22,f'{r["IG_C0"]:+.6f}',17,width=128,entry_id=key,role='IG_C0')
        p.text(751,top+22,f'{r["IG_C2s5"]:+.6f}',17,width=128,entry_id=key,role='IG_C2s5')
        d=r['C2s5_minus_C0'];col=TEAL if d>=0 else RED
        p.line(x(0),y,x(d),y,col,2);p.dot(x(d),y,4,col)
        p.text(1167,top+22,f'{d:+.6f}',17,width=120,color=col,entry_id=key,role='difference')
    foot=280+len(rows)*rh
    p.text(42,foot,'Differences use full-precision values; displayed IG values are rounded. Positive means more positive IG after reordering.',15,color=MUTED)
    p.text(42,foot+28,'Attribution redistribution under a shared reference; individual rows are not independent causal effects.',15,color=MUTED)
    if compact:p.text(42,foot+56,'Prespecified compact selection: first 10 C0 occurrences; all remaining 94 occurrences summed exactly.',15,color=MUTED)
    else:p.text(42,foot+56,'All occurrences and fixed groups are included across this complete multi-page figure; no row selection by IG magnitude.',15,color=MUTED)
    return p

def reports(out,r,native,pad,zero):
    prof=read(out/'environment/model_profile.json') if (out/'environment/model_profile.json').exists() else {}
    text=['本輪固定 Figure 1 的 WANDS query 409「teal chair」、product_id 24318；C0 → C2s5。',
      '原始檢索與 gradient diagnostic 分開報告。以下 saved native 是權威值：']
    for row in native:text.append(f'{row["schedule"]}: score={row["score"]:.17g}; saved rank={row["saved_rank"]}; reconstructed rank={row["reconstructed_rank"]}; margin={row["signed_margin"]:+.17g}')
    text+=['七個輸入均為 1059 tokens（含 special tokens），低於 8192 cap。104 個 occurrences 保留原始字串及身份。',
      f'overall_status={r["overall_status"]}; native_ready={r["native_ready"]}; PAD_plot_ready={r["PAD_plot_ready"]}。',
      '使用現存 phase2/.venv；未安裝或更新套件。FP16 native 後，將同一模型的 FP16-rounded parameters 轉為 FP32 做梯度。',
      'SDPA、原生 ModernBERT norm、RoPE、全域/局部 attention 和 CLS pooling 都保留；不求 rank 的梯度。',
      f'compile 狀態：native={prof.get("reference_compile_native")}; diagnostic={prof.get("reference_compile_diagnostic")}; changes={prof.get("profile_changes")}。',
      '歷史 cache 未記錄 compile 狀態，而且本輪 fresh target 單筆 forward 與原長度分桶 batch 不同；不宣稱 execution profile 完全相同。']
    if (out/'forward_checks.csv').exists():
        fw=pd.read_csv(out/'forward_checks.csv',float_precision='round_trip')
        for row in fw[fw.schedule.isin(['C0','C2s5'])].to_dict('records'):
            text.append(f'{row["schedule"]}: fresh={row["fresh_native"]:.17g}, diagnostic={row["diagnostic"]:.17g}, boundary_guard={row["boundary_guard"]}, input_ids/embeds error={row["input_ids_vs_embeds_error"]:.3g}。')
    for b in ['PAD','Zero']:text.append(f'{b} 驗收：'+json.dumps(r['baselines'][b],ensure_ascii=False))
    if (out/'baseline_score_checks.json').exists():
        for bc in read(out/'baseline_score_checks.json'):
            text.append(f'{bc["baseline"]} reference: F(b_C0)={bc["F_b_C0"]:.17g}, F(b_C2s5)={bc["F_b_C2s5"]:.17g}; baseline tensor hashes 相同={bc["same_baseline_tensor"]}; mask 相同={bc["same_masks"]}; reference score difference={bc["baseline_score_difference"]}。')
    if len(pad):
        if not r['PAD_plot_ready']:
            text.append('注意：以下 PAD 數值未通過完整驗收，僅保存作 failure diagnostic，不可據此聲稱主要屬性機制或寫入論文的歸因結論。未輸出主歸因圖。')
        text.append('PAD 歸因差值 IG(C2s5)-IG(C0)，以下最大絕對差值僅作數值摘要；全部 occurrences 仍在完整 CSV，未依故事篩選：')
        attrs=pad[pad.entry_id.str.startswith('attr_')]
        for row in attrs.reindex(attrs.C2s5_minus_C0.abs().sort_values(ascending=False).index).head(8).to_dict('records'):
            text.append(f'{row["entry_id"]} {row["entry_text"]} ({int(row["C0_position"])}→{int(row["C2s5_position"])}): {row["C2s5_minus_C0"]:+.9f}')
        text.append('非屬性群組完整分項：')
        for row in pad[~pad.entry_id.str.startswith('attr_')].to_dict('records'):text.append(f'{row["entry_id"]}: C0={row["IG_C0"]:+.9f}; C2s5={row["IG_C2s5"]:+.9f}; Δ={row["C2s5_minus_C0"]:+.9f}')
    if (out/'baseline_sensitivity.csv').exists():
        ss=pd.read_csv(out/'baseline_sensitivity.csv');text.append('PAD/Zero 逐項方向分類（兩邊各自的 order contrast；|Δ|≤1e-4 為近零）：'+str(ss.category.value_counts().to_dict()))
        if not all(r['baselines'][b]['passed'] for b in ['PAD','Zero']):text.append('至少一個 baseline 未通過數值驗收，因此方向分類僅屬未驗證診斷；不能宣稱 baseline-robust，也不能把未收斂差異解讀成有意義的 baseline 矛盾。')
        text.append('此近零門檻是事前固定的描述分類，非統計顯著性或逐項誤差界。不能以微小符號變化宣稱強烈矛盾。')
    if (out/'step_convergence.csv').exists():
        text.append('A/B/C 收斂歷程：');text.append(pd.read_csv(out/'step_convergence.csv').to_string(index=False))
    text+=['本案例為既有 Figure 1 的選定個案，不是新的代表性抽樣；歸因只描述共享 baseline 下的分配變化。',
      '不得把單一属性 IG 當作獨立因果效果，不宣稱唯一機制、普遍位置偏好或人口層級规律。',
      '有限階梯穩定性不是信賴區間，也不是真實誤差上界。',
      '既有 Figure 1、French molding 與稿件均保留，不自動替換。']
    if (out/'native_precision_audit.csv').exists():
        text.append('額外 28 個 forward-only 精度核對已重建原長度分桶的真實 companion，比較 reference_compile on/off 與 singleton/original batch；全部保留在 native_precision_audit.csv。')
        text.append('此有限核對未取得與歷史 cache 逐位相同的 product vectors；歷史 kernel/compile 狀態未被完整記錄，不能把差異歸因為單一已證實原因。所有 saved ranks 保留。這些不是額外 IG 或屬性局部移動實驗。')
    (out/'RESULTS.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
    cap='Revisiting the Figure 1 teal-chair case. (a) Saved native target-only results for WANDS query 409 and product 24318, with 42,993 competitors fixed at C0. The x-axis subtracts the fixed Top-20 competitor threshold (0.689082145691); exact ties use original catalog index. Reordering C0 to C2s5 changes the saved rank from 11 to 939. All seven inputs contain 1,059 tokens under the 8,192-token cap. '
    para='For the selected Figure 1 teal-chair case, changing only the complete attribute-occurrence order from C0 to C2s5 moves the saved native rank from 11 to 939 against a fixed C0 catalog. The corresponding native score decreases from 0.697439 to 0.636094, crossing the fixed Top-20 threshold of 0.689082. '
    if r['PAD_plot_ready']:
        cap+='(b) Official Captum integrated gradients for the fixed-query cosine score, using FP32 representations of the same FP16-rounded frozen GTE weights and a PAD word-vector reference. Every active non-special word vector, including fixed fields and separators, is replaced in the reference; masks and special vectors are retained. Rows follow C0 occurrence order and show signed IG before/after reordering and IG(C2s5)-IG(C0) on a common scale. Fixed fields, separators, boundary tokens, special tokens and padding remain separate. Full-precision differences are shown without per-order normalization; displayed numbers are rounded. The compact view uses the prespecified first 10 occurrences and sums the other 94. Completeness, paired accounting and consecutive-step stability checks pass for PAD. '
        para+='A numerically checked PAD-reference gradient diagnostic decomposes the score contrast into occurrence-aligned attribution redistribution, with fixed fields and boundary contributions accounted for separately. '
    else:cap+='No validated PAD attribution figure is released; see run_report.json for the diagnostic limitation. '
    cap+=f'Zero-reference sensitivity status: {r["baselines"]["Zero"]["status"]}. Baseline-specific order contrasts are reported separately; no baseline-robust claim is made. This selected illustration does not identify independent attribute effects, a unique mechanism, or population-level position preferences.'
    if r['PAD_plot_ready']:
        para+='These baseline-dependent attributions describe one selected case and do not identify independent causal effects of individual attributes or a general preference for particular attribute positions.'
    else:
        para+='The four PAD/Zero integrated-gradients diagnostics did not meet the prespecified completeness and stability criteria within 512 Gauss-Legendre nodes. We therefore draw no attribute-level attribution conclusion from these diagnostics. This selected native retrieval observation is not a population-level estimate of position preference.'
    (out/'caption.txt').write_text(cap+'\n',encoding='utf8');(out/'paper_paragraph.txt').write_text(para+'\n',encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output.resolve()
    r=read(out/'run_report.json');native=pd.read_csv(out/'native_records.csv',float_precision='round_trip').to_dict('records')
    assert r['native_ready'];export(out,'native_candidate_inclusion',[native_panel(native)])
    writecsv(out/'figure_data/native_plotted_values.csv',native)
    empty=pd.DataFrame();pad=zero=empty
    if (out/'aligned_entry_changes.csv').exists():
        data=pd.read_csv(out/'aligned_entry_changes.csv',float_precision='round_trip');pad=data[data.baseline=='PAD'];zero=data[data.baseline=='Zero']
    if r['PAD_plot_ready']:
        attrs=pad[pad.entry_id.str.startswith('attr_')].to_dict('records');fixed=pad[~pad.entry_id.str.startswith('attr_')].to_dict('records')
        assert [x['entry_id'] for x in attrs]==[f'attr_{i:03d}' for i in range(104)]
        other={'baseline':'PAD','entry_id':'other_attributes','entry_text':'Other attributes (94 entries)','C0_position':None,'C2s5_position':None}
        for k in ['IG_C0','IG_C2s5','C2s5_minus_C0']:other[k]=math.fsum(x[k] for x in attrs[10:])
        compact=attrs[:10]+[other]+fixed
        largest=max(abs(x['C2s5_minus_C0']) for x in attrs+fixed+[other]);limit=math.ceil(largest*1000/5)*.005
        chunks=[attrs[i:i+18] for i in range(0,len(attrs),18)]+[fixed]
        pages=[attribute_page(rows,('Fixed fields and token-accounting groups' if i==len(chunks)-1 else f'Occurrences {i*18+1}-{min(i*18+len(rows),104)}'),i+1,len(chunks),limit,r) for i,rows in enumerate(chunks)]
        export(out,'PAD_attribution_full',pages)
        export(out,'PAD_attribution_compact',[attribute_page(compact,'Prespecified compact view',1,1,limit,r,True)])
        writecsv(out/'figure_data/PAD_full_plotted_values.csv',attrs+fixed);writecsv(out/'figure_data/PAD_compact_plotted_values.csv',compact)
        dump(out/'figure_data/difference_axis.json',{'min':-limit,'max':limit,'scale_shared_all_pages_and_compact':True,'difference':'IG(C2s5)-IG(C0)'})
    reports(out,r,native,pad,zero)
    print('Figures exported and PDF pages rendered; review PNGs and actual draw.io rendering before final acceptance.')
if __name__=='__main__':main()
