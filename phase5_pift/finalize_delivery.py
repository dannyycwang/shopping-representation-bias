from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 checked=[]
 for row in json.loads((R/'phase5_mitigation/source_audit.json').read_text()):
  assert sha(R/row['path'])==row['sha256'],row['path']
  checked.append(row)
 ck=[]
 for folder in ['phase5_mitigation/checkpoints/Set-Attention_s42','phase5_pift/checkpoints/Standard-FT','phase5_pift/checkpoints/Adapted-PI-FT']:
  p=R/folder/'weights.pt';m=json.loads((p.parent/'metadata.json').read_text())
  assert sha(p)==m['weights_sha256']
  ck.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
 outputs=[]
 for name in ['phase5_pift','phase5_mitigation']:
  for p in sorted((R/name).rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['archive','vendor','__pycache__','checkpoints']):continue
   if p.name=='FINAL_DELIVERY_MANIFEST.json':continue
   if p.suffix in ['.py','.md','.json','.csv','.parquet','.pdf','.png']:outputs.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
 outputs.append({'path':'PHASE5_REPORT.md','sha256':sha(R/'PHASE5_REPORT.md')})
 (R/'phase5_pift/FINAL_DELIVERY_MANIFEST.json').write_text(json.dumps({'scope':'PI-FT adaptation and Set-attention complete; old consistency deferred','frozen_artifacts_verified':checked,'checkpoints_verified':ck,'outputs':outputs},indent=2),encoding='utf8')
 print('Verified frozen artifacts',len(checked),'checkpoints',len(ck),'outputs',len(outputs))
if __name__=='__main__':main()
