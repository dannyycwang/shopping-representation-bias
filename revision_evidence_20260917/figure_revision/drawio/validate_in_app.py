"""Local proof host for the official draw.io GraphViewer rendering engine.

Run this server, then open http://127.0.0.1:8767/. It loads each local diagram
with the official viewer bundle, creates PNG+SVG proofs, and saves only to qa/.
No source document or diagram is modified.
The alternative /embed route uses the documented editor load/export protocol.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote_to_bytes
import base64
import json

HERE = Path(__file__).resolve().parent
NAMES = ['figure1_teal_chair_candidate', 'figure2_intervention_and_states', 'rq1_vi_top20_main',
         'rq1_states_top20_top100_appendix', 'rq2_membership_top20_main', 'rq2_membership_top100_appendix',
         'rq2_cancellation_top20_table']
HTML = '''<!doctype html><meta charset="utf-8"><title>Draw.io figure verification</title>
<style>body{margin:0;font:16px Arial;background:#f0f1f3}header{padding:12px 18px;background:white}
iframe{width:100%;height:calc(100vh - 80px);border:0}#status{display:inline-block;margin-left:15px}</style>
<header><b>Editable figure verification</b><span id="status">Waiting for draw.io...</span>
<button id="restart">Restart checks</button><button id="collection">Open seven-page collection</button></header>
<iframe id="editor" src="https://embed.diagrams.net/?embed=1&proto=json&spin=1&libraries=1&noSaveBtn=1&noExitBtn=1&saveAndExit=0&ui=atlas&dark=0"></iframe>
<script>
const names=NAMES, frame=document.querySelector('#editor'), status=document.querySelector('#status');
let index=0, active='', mode='batch';
const send=m=>frame.contentWindow.postMessage(JSON.stringify(m),'https://embed.diagrams.net');
async function loadNext(){
 if(index>=names.length){status.textContent='PASS: all 7 diagrams loaded and exported';return;}
 active=names[index];status.textContent=`${index+1}/7: ${active}`;
 const xml=await (await fetch('/diagram?name='+active)).text();
 send({action:'load',xml,title:active,fit:1,maxFitScale:2,dark:false});
}
window.addEventListener('message',async e=>{
 if(e.origin!=='https://embed.diagrams.net')return;
 let m;try{m=JSON.parse(e.data)}catch{return;}
 if(m.error){status.textContent='ERROR: '+m.error;await fetch('/event',{method:'POST',body:JSON.stringify(m)});return;}
 if(m.event==='init')await loadNext();
 if(m.event==='load'){
   await fetch('/event',{method:'POST',body:JSON.stringify({name:active,event:'load',bounds:m.bounds,scale:m.scale,page:m.page})});
   if(mode==='batch')send({action:'export',format:'png',scale:1.5625,size:'page',background:'#ffffff',withSvg:true});
 }
 if(m.event==='export'){
   await fetch('/proof?name='+active,{method:'POST',body:JSON.stringify(m)});
   index++;await loadNext();
 }
});
document.querySelector('#restart').onclick=()=>{mode='batch';index=0;loadNext()};
document.querySelector('#collection').onclick=async()=>{mode='collection';active='all_seven_figures';
 const xml=await (await fetch('/diagram?name='+active)).text();
 send({action:'load',xml,title:'All seven editable figures',fit:1,maxFitScale:2});
 status.textContent='Seven-page editable collection';};
</script>'''.replace('NAMES', json.dumps(NAMES))

VIEWER_HTML = '''<!doctype html><meta charset="utf-8"><title>Draw.io native rendering proofs</title>
<style>body{font:16px Arial;background:#eee;margin:16px}article{padding:15px;background:white;margin:15px 0;width:max-content}h2{font-size:16px}#status{position:sticky;top:0;background:white;padding:12px}</style>
<div id="status">Loading official draw.io renderer...</div><main id="proofs"></main>
<script>
async function renderFigures(){
 const names=NAMES,status=document.querySelector('#status');
 try{for(const [i,name] of names.entries()){
   status.textContent=`Rendering ${i+1}/7: ${name}`;
   const xml=await (await fetch('/diagram?name='+name)).text();
   const article=document.createElement('article'),label=document.createElement('h2'),div=document.createElement('div');
   label.textContent=name;article.append(label,div);document.querySelector('#proofs').append(article);
   div.style.width='1100px';div.style.height='850px';
   div.setAttribute('data-mxgraph',JSON.stringify({xml,toolbar:'',nav:false,resize:true,lightbox:false,zoom:1.5625}));
   const viewer=await new Promise(resolve=>GraphViewer.createViewerForElement(div,resolve));
   const svg=viewer.graph.getSvg('#ffffff',1.5625,0);
   const svgText=new XMLSerializer().serializeToString(svg);
   const im=new Image();im.src='data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(svgText)));await im.decode();
   const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;
   canvas.getContext('2d').drawImage(im,0,0);
   const roundtrip=mxUtils.getXml(new mxCodec().encode(viewer.graph.getModel()));
   await fetch('/proof?name='+name,{method:'POST',body:JSON.stringify({data:canvas.toDataURL('image/png'),svg:svgText,xml:roundtrip})});
   await fetch('/event',{method:'POST',body:JSON.stringify({event:'render',name,engine:'official draw.io GraphViewer',width:im.width,height:im.height,nativeCellCount:Object.keys(viewer.graph.getModel().cells).length})});
 }
 status.textContent='PASS: all 7 native diagrams rendered and round-tripped with draw.io';
 }catch(e){status.textContent='ERROR: '+e.message;await fetch('/event',{method:'POST',body:JSON.stringify({error:e.message,stack:e.stack})});}
}
</script><script src="https://viewer.diagrams.net/js/viewer-static.min.js" onload="renderFigures()"></script>'''.replace('NAMES', json.dumps(NAMES))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, mime='text/plain; charset=utf-8', code=200):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == '/':
            self.reply(VIEWER_HTML.encode(), 'text/html; charset=utf-8')
        elif route.path == '/embed':
            self.reply(HTML.encode(), 'text/html; charset=utf-8')
        elif route.path == '/diagram':
            name = parse_qs(route.query).get('name', [''])[0]
            if name not in NAMES + ['all_seven_figures']:
                return self.reply(b'Invalid figure', code=400)
            self.reply((HERE/(name+'.drawio')).read_bytes(), 'application/xml; charset=utf-8')
        else:
            self.reply(b'Not found', code=404)

    def do_POST(self):
        route = urlparse(self.path)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        payload = json.loads(raw)
        qa = HERE/'qa'
        qa.mkdir(exist_ok=True)
        if route.path == '/event':
            with (qa/'app_events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(payload)+'\n')
        elif route.path == '/proof':
            name = parse_qs(route.query).get('name', [''])[0]
            if name not in NAMES:
                return self.reply(b'Invalid figure', code=400)
            header, data = payload['data'].split(',', 1)
            assert header.startswith('data:image/png')
            (qa/(name+'.png')).write_bytes(base64.b64decode(data) if ';base64' in header else unquote_to_bytes(data))
            if payload.get('svg'):
                (qa/(name+'.svg')).write_text(payload['svg'], encoding='utf-8')
            if payload.get('xml'):
                (qa/(name+'_roundtrip.drawio')).write_text(payload['xml'], encoding='utf-8')
            print('Rendered with draw.io:', name, flush=True)
        else:
            return self.reply(b'Not found', code=404)
        self.reply(b'OK')

if __name__ == '__main__':
    print('Open http://127.0.0.1:8767/ to validate seven diagrams with draw.io.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8767), Handler).serve_forever()
