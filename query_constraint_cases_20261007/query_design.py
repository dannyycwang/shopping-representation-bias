"""Manual query/schema-only design, written before ranking direction is inspected.

Tuples: exact span, family, normalized target, audited scope, query ambiguity.
Every original query is retained; this dictionary defines the explicit literals.
"""
# Scope is explicit: a frame/upholstery clause never stands for the whole item.
DESIGN={
3:('pillow',[('turquoise','color','turquoise','whole',False)],[]),
6:('chair',[('acrylic','material','acrylic','whole',False),('clear','color','clear','whole',False)],[]),
7:('mirror',[('driftwood','material','driftwood','frame',False)],[]),
16:('seat cushion',[('blk','color','black','whole',False)],['18x18: units and dimensional axes unspecified']),
22:('pillow',[('light and navy blue','color','light and navy blue','whole',True)],['conjunction of shades; single or multiple items not specified']),
26:('chair',[('leather','material','leather','upholstery',False)],[]),
31:('curtain',[('burnt orange','color','burnt orange','whole',False)],[]),
46:('headboard',[('blue','color','blue','whole',False)],['beach: style or shade qualifier not resolved']),
50:('chair',[('gold','color','gold','legs',False)],['tufted']),
55:('rug',[('teal','color','teal','whole',False)],['tollette: named product identity']),
81:('desk',[('l shape','shape','l-shaped','whole',False)],['orren ellis: brand']),
84:('bed',[('metal','material','metal','frame',False),('rose gold','color','rose gold','whole',False)],['full: nominal bed size deferred']),
88:('lounge',[('rose gold','color','rose gold','whole',True)],['lounge has no unambiguous product head']),
97:('rug',[('red','color','red','whole',True)],['regner: named product; rug head omitted; power loom']),
126:('chair',[('leather','material','leather','upholstery',False)],['dining use']),
137:('desk',[('l shaped','shape','l-shaped','whole',False)],['orren ellis: brand']),
138:('door',[('glass','material','glass','whole',False)],['for bath: compatibility']),
142:('stand',[('wooden','material','wood','whole',False)],['kitchen use']),
144:('rug',[('plum','color','plum','whole',False)],[]),
151:('chair',[('leather','material','leather','upholstery',False)],[]),
160:('',[('marble','material','marble','whole',True)],['no product head; material or marble pattern unresolved']),
182:('tile',[('round','shape','round','whole',False)],['penny: nominal mosaic scale not independently verified']),
190:('chair',[('faux leather','material','faux leather','upholstery',False)],['benjiamino: identity','power lift']),
195:('bathtub',[('black','color','black','whole',False)],['freestanding','with faucet']),
200:('tile',[('ceramic','material','ceramic','whole',False)],['sea shell: pattern or shape not specified']),
210:('sofa',[('faux leather','material','faux leather','upholstery',False)],['love seat: capacity','wide: no numeric bound','tuxedo arm']),
230:('',[('gold','color','gold','whole',True)],['no product head; color versus precious metal unresolved']),
242:('chair',[('leather','material','leather','upholstery',False)],['accent style']),
246:('shower curtain',[('clear','color','clear','whole',False)],[]),
250:('rug',[('multi color','color','multicolor','whole',False)],[]),
266:('chair',[('champagne','color','champagne','whole',False),('velvet','material','velvet','upholstery',False)],['desk use']),
267:('tile',[('dolomite','material','dolomite','whole',True)],['dolomite may be collection/pattern; subway format']),
268:('umbrella',[('resin','material','resin','whole',True)],['betty: identity; resin could modify base rather than canopy/frame','free standing']),
273:('shower caddy',[('stainless steel','material','stainless steel','whole',False)],['free standing']),
275:('ottoman',[('square','shape','square','whole',False)],['hinged']),
292:('pillow',[('white','color','white','whole',False)],[]),
300:('rug',[('wool','material','wool','whole',False),('beige/black','color','beige/black','whole',True)],['animal print','handmade','tufted','by allmodern: brand']),
301:('cabinet pull',[('circle','shape','round','whole',False)],[]),
302:('mattress topper',[('foam','material','foam','whole',False)],['queen: nominal size']),
307:('floral arrangement',[('glass','material','glass','vase',False)],['real touch','roses']),
311:('',[('velvet','material','velvet','upholstery',True)],['odum: identity; product head omitted']),
332:('coffee table',[('drum','shape','drum','whole',True)],['attleboro: identity; drum is form/style rather than unique geometric category']),
333:('colander',[('stainless steel','material','stainless steel','whole',False)],['set: cardinality not specified']),
341:('wall art',[('canvas','material','canvas','art surface',False)],['map subject']),
367:('chair',[('wooden','material','wood','frame',False)],[]),
403:('rack',[('wood','material','wood','whole',True)],['wood may be stored firewood, not rack material','wide: no numeric bound']),
409:('chair',[('teal','color','teal','whole',False)],[]),
429:('dresser',[('gray','color','gray','whole',False)],[]),
440:('stool',[('wood','material','wood','frame',False)],['bar: nominal seat height/use not independently verified']),
460:('wardrobe',[('grey','color','gray','whole',False)],['small: no numeric bound']),
465:('',[('white','color','white','whole',True)],['abstract: no product head; may refer to artwork']),
476:('bed',[('hardwood','material','hardwood','frame',False)],[])
}

# Explicit size/finish/feature words retained with deferral, never silently interpreted.
DEFERRED={10:'king: nominal bed size',37:'24 inches height: seat versus overall height unspecified; raw units not documented',44:'chrome: surface finish, not assumed bulk material or color',62:'7qt: capacity outside dimensional scope',64:'72.5: unit/direction absent',76:'46 inch: width/height unspecified',101:'48 in: axis unspecified',122:'king size: nominal size',143:'queen may be a product-name modifier on chair',202:'full: nominal mattress size',203:'4.5: capacity/unit omitted',204:'bronze: finish/material ambiguity',206:'73: unit/direction absent',215:'king: nominal bed size',220:'twinxl: nominal bedding size',223:'3-3/4: unit/direction absent',253:'antique brass: finish, not bulk material',279:'queen: nominal bed size',285:'48 inch: door dimension unspecified',289:'king: nominal bedding size',296:'twin: nominal bed size',308:'60: role/unit absent',327:'brushed bronze: finish',369:'wrought: incomplete material expression',389:'brush nickel: finish, not bulk nickel',417:'twin: nominal bed size',432:'butcher block: construction/style, not explicit wood value',436:'full: nominal bed size',482:'48 inch: likely nominal width but units of raw width field not documented'}

# Schema-only amendment before ANY ranking-direction scan: nominal bedding labels
# are directly recorded and do not need assumed numeric units.
for qid,head,span,target,other in [
 (10,'bed','king','king',['poster']), (84,'bed','full','full',[]),
 (122,'bed','king size','king',[]), (202,'mattress liner','full','full',['padded']),
 (215,'bed','king','king',['adjustable','including mattress']),
 (220,'sheet','twinxl','twin xl',[]), (279,'bed','queen','queen',['marlon: identity','tufted']),
 (289,'comforter','king','king',['coma inducer: identity','set']),
 (296,'bed','twin','twin',['zakariyah: identity','platform']),
 (302,'mattress topper','queen','queen',[]), (417,'bed','twin','twin',[]),
 (436,'bed','full','full',['with trundle'])]:
 if qid in DESIGN:
  h,cs,rest=DESIGN[qid];DESIGN[qid]=(h,cs+[(span,'nominal_size',target,'bedding',False)], [x for x in rest if 'nominal' not in x])
 else:DESIGN[qid]=(head,[(span,'nominal_size',target,'bedding',False)],other)
 DEFERRED.pop(qid,None)

PROTOCOL={
 'version':'1.1.0','date':'2026-10-07','exploratory':True,'preregistration':False,
 'pre_result_amendment':'Catalog census identifies explicit mattresssize/beddingsize/bedsize labels; include nominal size without numeric conversion. Remove bamboo from wood family because botanical/category interpretation is ambiguous. Initial design archived; no ranking direction had been computed.',
 'known_seeds':[{'query_id':367,'query':'wooden chair outdoor'},{'query_id':292,'query':'decorative white pillow'}],
 'historical_reference_commit':'fb78cc9f2a2b03399b71931f50da16684aac3af5',
 'primary':{'dataset':'wands','model':'bge_base','before':'C0','after':'C1','k':20,'intervention':'catalog-wide'},
 'secondary':{'models':['bge_base','minilm'],'after_schedules':['C1','C2s1','C2s2','C2s3','C2s4','C2s5'],'before':'C0','exclude_primary_duplicate':True},
 'highest_label':'Exact','gain_mapping':{'Exact':3,'Partial':1,'Irrelevant':0},
 'field_maps':{
  'color:whole':['color','primarycolor','maincolor'], 'color:legs':['legcolor'],
  'material:whole':['material','primarymaterial','mainmaterial'],
  'material:frame':['framematerial','outerframematerial'],
  'material:upholstery':['upholsterymaterial'], 'material:vase':['vasematerial','containermaterial'],
  'material:art surface':['artmedium','artmaterial','material'],
  'shape:whole':['shape','rugshape','topshape','tableshape','pullshape'],
  'nominal_size:bedding':['mattresssize','beddingsize','bedsize','bedsizecompatibility']},
 'mapping_qualifications':{'shape':'topshape/tableshape only for tables/desks, rugshape only rugs, pullshape only cabinet pulls; component shapes do not establish overall form on other types',
  'material':'Frame/upholstery/vase/surface audits are explicitly scoped subclauses. A matching frame does not establish whole-chair material or all query requirements. Generic material fields are not substituted for named parts.',
  'color':'Generic/main/primary color are recorded overall color roles; conflicts across aliases or variants remain ambiguous. Finish, cushion, upholstery, frame or seat colors are never silently converted to whole-item color.'},
 'normalization':'NFC, casefold, trim, collapse whitespace; preserve every original entry and full composite scalar.',
 'color_aliases':{'grey':'gray','blk':'black','multi color':'multicolor','multi-color':'multicolor'},
 'simple_colors':['white','black','gray','blue','red','green','yellow','orange','purple','pink','brown','beige','turquoise','teal','plum','gold','rose gold','silver','champagne','clear','burnt orange','multicolor'],
 'color_rules':'Exact descriptor/explicit alias matches. Different clear basic named colors contradict, except neighboring shade/family ambiguities and white/ivory/cream/off-white. Composites containing commas, slash, semicolon, ampersand or conjunction remain ambiguous, even if target occurs. Qualified shades are ambiguous unless exact target. Generic absence does not contradict. Text can downgrade a field decision on a conflicting explicit color claim, not promote missing fields to match.',
 'material_families':{'wood':['wood','solid wood','acacia','teak','mahogany','oak','pine','eucalyptus','cedar','rubberwood','birch','walnut'],
  'metal':['metal','steel','stainless steel','aluminum','aluminium','iron','wrought iron','brass'],
  'plastic':['plastic','resin','plastic/resin','resin/plastic','polypropylene','polyethylene','acrylic'],
  'leather':['genuine leather','leather','100 % leather'], 'faux leather':['faux leather','100 % faux leather'],
  'velvet':['velvet','100 % velvet'],'glass':['glass','tempered glass'],'wool':['wool','100 % wool'],'foam':['foam','memory foam'],'ceramic':['ceramic','porcelain']},
 'material_rules':'Exact target or listed family supports MATCH. Manufactured/engineered/composite wood and mixed materials are AMBIGUOUS for wood/hardwood, never silently equated to solid wood. Frame wood versus explicit plastic/resin or metal is CONTRADICTION for frame only. Leather excludes explicit faux leather. Ceramic query accepts ceramic only; porcelain remains an adjacent subtype ambiguity. Unlisted values are UNKNOWN/AMBIGUOUS. Field details and explicit same-part construction descriptions can downgrade conflicting decisions; appearance expressions wood grain/look/weather wood and woodtone do not assert wooden composition.',
 'shape_aliases':{'rectangular':'rectangle','circular':'round','circle':'round','l shaped':'l-shaped','l shape':'l-shaped'},
 'shape_rules':'Only exact target or listed aliases MATCH. Other explicit single basic geometries contradict; mixed forms are ambiguous. No dimensional inference.',
 'status_reduction':'Across mapped values: all MATCH => MATCH, all CONTRADICTION => CONTRADICTION, any conflicting/ambiguous evidence or mixed statuses => AMBIGUOUS; absent/insufficient evidence => UNKNOWN. Multiple variants uniformly satisfying the broad condition can MATCH, but all variants and multiplicities remain visible. Conflicting title/description statements downgrade rather than being discarded.',
 'dimensions':'Numeric dimensional candidates retained but deferred: raw numeric attributes do not provide a documented global unit/axis schema sufficient for safe exact query matching. Nominal mattress/bedding/bed labels are evaluated directly: full/double aliases full, twinxl/extra-long twin aliases twin xl; king and California king are distinct; mixed/bunk/variant values ambiguous. No tolerance, unit assumption or size-to-dimension conversion is introduced.',
 'product_type':'Class, producttype, title and description retained separately from attribute status. Pillow cover alone contradicts complete pillow; cover & insert supports pillow. Ambiguous or contradictory type evidence downgrades cases. Type tests cover scanned query heads; missing type evidence is UNKNOWN.',
 'use':'Outdoor/indoor is a separate clause where explicit in the query. Use explicit outdooruse/outdoorsafe/indooroutdooruse or exact outdoor/indoor wording in source; weather resistance alone is insufficient. Conflicting use evidence is AMBIGUOUS.',
 'joint':'Joint MATCH requires every explicit assessed clause, product type and applicable use to MATCH, no unassessed literal qualifier and no narrowing from whole item to part. Any CONTRADICTION => joint CONTRADICTION; otherwise AMBIGUOUS if any ambiguous or scope narrowed, UNKNOWN if any unassessed/missing. Subjective style terms are outside the objective audit. Clause results are not joint query satisfaction.',
 'bounds':{'loose':'[M1-(M0+U0+A0), (M1+U1+A1)-M0]', 'shared_cancelled':'[M_G-(M_L+U_L+A_L), (M_G+U_G+A_G)-M_L]', 'loss_proven':'upper < 0','disappearance':'M0 > 0 and M1=U1=A1=0'},
 'candidate':'Query explicit; benchmark highest counts equal as integers; at least one MATCH in all-product L and one CONTRADICTION in all-product G. Relevant-only and full Top20 always separate. Nonzero relevant substitutions additionally flagged, not conflated with unchanged membership.',
 'grading':'A: clear scoped clause, MATCH-out/CONTRADICTION-in, equal Recall, full Top20 tightened upper bound <0. B: same clean substitution but full Top20 decline not proved. C: query/variant/type/execution uncertainty. A/B are scoped clause evidence, not joint preference claims. Cache reproduction and manual all-candidate audit required for final A/B recommendation.',
 'selection':'After the fixed 12 comparisons are scanned, sort candidates by clarity, verified execution, A before B before C, primary before secondary, exact nDCG equality before other, fewer complete membership transitions, query_id then model/schedule/constraint. At most 3 distinct query cases; known wooden case mandatory audit, white pillow mandatory audit; third slot best additional eligible query. No effect-size maximum, new schedules or tests after selection.',
 'reproduction':'For at most three selected query cases reuse comparable frozen query/product caches to rescore the full catalog for both schedules with one matched float32 matrix/sort implementation. Reuse is not new encoding, and does not establish bitwise historical forward equivalence. Record absent per-batch telemetry as unknown. Do not re-encode full catalog or download models.',
 'authorization':'Latest user instruction supersedes task-file no-commit/no-push: commit and push finished new directory; do not edit manuscript or include unrelated changes.'
}
