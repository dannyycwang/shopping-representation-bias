from common import *
import re

# Readability adjustments for prose only. File paths, hashes and scientific tables remain exact.
def polish(text):
 replacements={
  'All48':'All 48','All12':'All 12','all48':'all 48','all499':'all 499','all42,993':'all 42,993',
  'the128':'the 128','the308':'the 308','the139':'the 139','the1,059':'the 1,059','the512':'the 512',
  'a256':'a 256','at256':'at 256','under256':'under 256','under8,192':'under 8,192',
  'with10,000':'with 10,000','uses10,000':'uses 10,000','together,10,000':'together, 10,000',
  'query,10,000':'query, 10,000','uses128':'uses 128','uses16':'uses 16','audits16':'audits 16',
  'fixed128':'fixed 128','fixed-C0':'fixed C0','atC0':'at C0','sourceC0':'source C0',
  'across7,16 and32':'across 7, 16 and 32','across7':'across 7','atK20/K100':'at K=20/100',
  'atK20':'at K=20','atK100':'at K=100','K20/K100':'K=20/100',
  'case: 25':'case: 25','Top20':'Top-20','Top100':'Top-100',
  'ranks15–174':'ranks 15–174','spans11–939':'spans 11–939','ranks11–939':'ranks 11–939',
  'ranks20 and39':'ranks 20 and 39','ranks19 and14':'ranks 19 and 14',
  'rank479':'rank 479','reconstructed480':'reconstructed 480','rank19':'rank 19','rank11':'rank 11',
  'competitor30099':'competitor 30099','product385':'product 385',
  'WANDS308q':'WANDS 308q','ESCI499q':'ESCI 499q','WANDS128':'WANDS 128','ESCI499':'ESCI 499',
  'in64.1':'in 64.1','and81.1':'and 81.1','loses0.232':'loses 0.232',
  'are118,698,552':'are 118,698,552','and24,049,044':'and 24,049,044',
  'within1e-10':'within 1e-10','to1e-10':'to 1e-10','Chapter6':'Chapter 6',
  'transformers4.46.2':'transformers 4.46.2','transformers4.55.4':'transformers 4.55.4',
  'torch2.11.0':'torch 2.11.0','RTX4060':'RTX 4060','seed2026091701':'seed 2026091701',
  'draws,10,000':'draws, 10,000','uncorrected95%':'uncorrected 95%','use0–1':'use 0–1','stay0–1':'stay 0–1',
  'all32':'all 32','every32':'every 32','maximum3':'maximum 3','entries/6orders':'entries / 6 orders',
  'Families are nested7/16/32':'Families are nested 7/16/32','seeds2026092201':'seeds 2026092201',
  'then2210–2225':'then 2210–2225','stratified24':'stratified 24','PhaseIV':'Phase IV',
  'steps64':'steps 64','Gauss–Legendre64':'Gauss–Legendre 64','vs dense9':'vs dense 9'
 }
 for a,b in replacements.items():text=text.replace(a,b)
 return text

if __name__=='__main__':
 for name in ['README.md','BOUNDARY_AUDIT.md','XAI_AUDIT.md','CONTROL_TRANSITIONS.md','PERMUTATION_COVERAGE.md','CLAIM_VERDICTS.md','CHAPTER6_EVIDENCE_MAP.md']:
  p=HERE/name;p.write_text(polish(p.read_text(encoding='utf8')),encoding='utf8')
