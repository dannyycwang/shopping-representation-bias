"""Conservative recorded-evidence adjudication and independently testable bounds."""
import re, unicodedata
import numpy as np
STATES=['MATCH','CONTRADICTION','UNKNOWN','AMBIGUOUS']

def published_rank(known_ranks,key):
    """Missing outside-cutoff rank is a bound, never an invented position."""
    return known_ranks.get(key,'>20')

def norm(x):return ' '.join(unicodedata.normalize('NFC',str(x)).casefold().split())
def atoms(p):
    out=[]
    for i,a in enumerate(p['attributes']):
        k,sep,v=a.partition(':')
        if sep:out.append(dict(index=i,raw_entry=a,raw_key=k,raw_value=v,key=norm(k),value=norm(v)))
    return out
def reduce_status(states):
    if not states or set(states)=={'UNKNOWN'}:return 'UNKNOWN'
    if set(states)=={'MATCH'}:return 'MATCH'
    if set(states)=={'CONTRADICTION'}:return 'CONTRADICTION'
    return 'AMBIGUOUS'
def multiple(v):return bool(re.search(r'[,;/&+]|\band\b|\bwith\b|\bblend\b',v))

def value_status(v,c,protocol):
    if v in ['','unknown','not specified','n/a','none','other','varies']:return 'UNKNOWN'
    target=c['normalized_value'];kind=c['constraint_type']
    if kind=='color':
        v=protocol['color_aliases'].get(v,v)
        if v==target:return 'MATCH'
        if multiple(v):return 'AMBIGUOUS'
        adjacent=[{'white','ivory','cream','off-white','off white','ivory & cream'},{'teal','turquoise','blue','green'},{'plum','purple'},{'champagne','beige','gold'},{'rose gold','gold','pink'}]
        if any(v in g and target in g for g in adjacent):return 'AMBIGUOUS'
        if any(re.search(r'\b'+re.escape(x)+r'\b',v) for x in [target]) or target in ['light and navy blue','beige/black']:return 'AMBIGUOUS'
        if v in protocol['simple_colors'] and target in protocol['simple_colors']:return 'CONTRADICTION'
        return 'AMBIGUOUS'
    if kind=='nominal_size':
        aliases={'full / double':'full','full/double':'full','double':'full','extra-long twin':'twin xl','twin extra long':'twin xl','twinxl':'twin xl'}
        v=aliases.get(v,v)
        if v==target:return 'MATCH'
        if v in ['twin','twin xl','full','queen','king','california king']:return 'CONTRADICTION'
        return 'AMBIGUOUS'
    if kind=='shape':
        v=protocol['shape_aliases'].get(v,v)
        if v==target:return 'MATCH'
        if multiple(v):return 'AMBIGUOUS'
        if v in ['rectangle','round','square','oval','triangle','hexagon','octagon','l-shaped','semi-circle'] and target in ['rectangle','round','square','l-shaped']:return 'CONTRADICTION'
        return 'AMBIGUOUS'
    if kind=='material':
        if v==target:return 'MATCH'
        families=protocol['material_families']
        if target in ['wood','hardwood'] and any(x in v for x in ['manufactured','engineered','composite','bamboo']):return 'AMBIGUOUS'
        if target in families and v in families[target]:
            if target=='ceramic' and v=='porcelain':return 'AMBIGUOUS'
            return 'MATCH'
        if target=='acrylic' and v=='plastic/acrylic':return 'AMBIGUOUS'
        if multiple(v) and v not in ['plastic/resin','resin/plastic']:return 'AMBIGUOUS'
        fam=next((k for k,values in families.items() if v in values),None)
        target_fam=target if target in families else next((k for k,values in families.items() if target in values),None)
        if target in ['hardwood','driftwood']:target_fam='wood'
        if target=='resin':target_fam='plastic'
        if fam and target_fam and fam!=target_fam:return 'CONTRADICTION'
        if fam==target_fam:return 'AMBIGUOUS' # e.g. wood does not establish hardwood.
        if v in ['polyester','cotton','linen','microfiber','nylon'] and target in ['leather','faux leather','velvet','wool','foam']:return 'CONTRADICTION'
        return 'UNKNOWN'
    return 'UNKNOWN'

def classify(p,c,protocol):
    fieldmap=protocol['field_maps'][c['constraint_type']+':'+c['scope']]
    fieldmap=list(fieldmap)
    if c['constraint_type']=='shape':
        if c['modifier']!='rug':fieldmap=[x for x in fieldmap if x!='rugshape']
        if c['modifier'] not in ['desk','coffee table','table']:fieldmap=[x for x in fieldmap if x not in ['topshape','tableshape']]
        if c['modifier']!='cabinet pull':fieldmap=[x for x in fieldmap if x!='pullshape']
    selected=[a for a in atoms(p) if a['key'] in fieldmap]
    states=[value_status(a['value'],c,protocol) for a in selected]
    status=reduce_status(states);flags=[];text_claims=[]
    text=norm(p['title']+' . '+p['description'])
    if c['constraint_type']=='color' and status in ['MATCH','CONTRADICTION']:
        colors=sorted(protocol['simple_colors'],key=len,reverse=True)
        found=set(re.findall(r'\b(?:'+'|'.join(map(re.escape,colors))+r')\b',text))
        for color in sorted(found):
            verdict=value_status(color,c,protocol)
            if (status=='MATCH' and verdict=='CONTRADICTION') or (status=='CONTRADICTION' and verdict=='MATCH'):
                flags.append('text_color_conflict_or_different_part');text_claims.append(color)
        if flags:status='AMBIGUOUS'
    if c['constraint_type']=='material' and status in ['MATCH','CONTRADICTION']:
        # Only composition phrases or same-part material details can contradict composition.
        details=[a for a in atoms(p) if a['key'] in [x+'details' for x in fieldmap]]
        pattern=r'(?:crafted|made|constructed|built)\s+(?:entirely\s+)?(?:from|of|with)\s+[^.;]{0,95}'
        phrases=re.findall(pattern,text)
        if c['scope']=='frame':phrases+=re.findall(r'frame\s+(?:is\s+|of\s+|made\s+from\s+)[^.;]{0,70}',text)
        if c['scope']=='upholstery':phrases+=re.findall(r'(?:upholstered|upholstery)\s+(?:in\s+|with\s+)?[^.;]{0,70}',text)
        mentions=sorted({v for vals in protocol['material_families'].values() for v in vals},key=len,reverse=True)
        for phrase in phrases+[a['value'] for a in details]:
            phrase=re.sub(r'wood[ -]?(?:grain|look|tone)|weather wood','appearance',phrase)
            for token in re.findall(r'\b(?:'+'|'.join(map(re.escape,mentions))+r')\b',phrase):
                verdict=value_status(token,c,protocol)
                # Whole-item construction can mention multiple parts: flag, never choose one.
                if (status=='MATCH' and verdict=='CONTRADICTION') or (status=='CONTRADICTION' and verdict=='MATCH'):
                    flags.append('composition_text_or_detail_conflict_requires_scope_review');text_claims.append(phrase)
        if flags:status='AMBIGUOUS'
    return dict(status=status,field_map=fieldmap,raw_fields=selected,value_statuses=states,
                variant_values=sorted({a['value'] for a in selected}),text_conflict_flags=sorted(set(flags)),
                description_claim_fragments=sorted(set(text_claims)),reason='; '.join(sorted(set(flags))) if flags else
                'No semantically mapped recorded value; absence is not contradiction' if not selected else
                'Fixed condition-level reduction of every mapped scalar/variant; part scope retained')

TYPE_PATTERNS={
 'pillow':r'pillow', 'chair':r'chair|armchair|recliner', 'seat cushion':r'cushion', 'curtain':r'curtain|drape',
 'headboard':r'headboard','rug':r'rug','desk':r'desk','bed':r'\bbeds?\b|bed frame|bunk|daybed',
 'mirror':r'mirror','door':r'door','stand':r'stand|shelv','tile':r'tile','bathtub':r'bathtub|bath tub|tub',
 'sofa':r'sofa|loveseat|love seat','shower curtain':r'shower curtain','umbrella':r'umbrella',
 'shower caddy':r'caddy|shower.*storage|shower.*shel','ottoman':r'ottoman|pouf',
 'cabinet pull':r'pull|knob','mattress topper':r'topper|mattress pad','mattress liner':r'mattress pad|mattress protector|mattress cover',
 'floral arrangement':r'floral|flower|arrangement','coffee table':r'coffee table','colander':r'colander|strainer',
 'wall art':r'wall art|print|painting|artwork','rack':r'rack','dresser':r'dresser|chest',
 'stool':r'stool','wardrobe':r'wardrobe|armoire','sheet':r'sheet','comforter':r'comforter|bedding set'}

def product_type(p,head):
    fields=[a for a in atoms(p) if a['key'] in ['producttype','producttypes','pillowtype','throwdecorativepillowtype']]
    title=norm(p['title']);desc=norm(p['description']);cls=norm(p['class'])
    if not head or head not in TYPE_PATTERNS:return dict(status='AMBIGUOUS',reason='No unique supported query product head',raw_fields=fields)
    if head=='pillow':
        values=[a['value'] for a in fields if a['key']=='producttype']
        statuses=[]
        for v in values:
            if v in ['pillow cover & insert','pillow cover and insert','throw pillow','pillow']:statuses.append('MATCH')
            elif v in ['pillow cover','pillowcase','pillow sham','sham']:statuses.append('CONTRADICTION')
            else:statuses.append('AMBIGUOUS')
        explicit_cover_only=bool(re.search(r'insert (?:is )?not included|cover only|pillow cover without',desc+' '+title))
        if statuses:
            if explicit_cover_only:statuses.append('CONTRADICTION')
            return dict(status=reduce_status(statuses),reason='Specific pillow product type; cover-only versus complete pillow kept separate',raw_fields=fields)
        if explicit_cover_only or 'pillow cover' in title:return dict(status='CONTRADICTION',reason='Cover-only source evidence; not a complete pillow',raw_fields=fields)
    if re.search(TYPE_PATTERNS[head],cls):status='MATCH'
    elif head=='chair' and re.search(r'rocker|glider|swing',cls) and re.search(r'\bchair\b|\barmchair\b|\brecliner\b',title):status='MATCH'
    elif cls:status='UNKNOWN'
    elif re.search(TYPE_PATTERNS[head],title):status='MATCH'
    else:status='UNKNOWN'
    if head=='chair' and any(t in cls for t in ['chairmat','cover','cushion']):status='CONTRADICTION'
    return dict(status=status,reason='Recorded class or compatible seating class plus explicit chair title; nonmatching taxonomy is insufficient evidence, not automatic contradiction. Attribute and qrel remain separate.',raw_fields=fields)

def use_status(p,target):
    fields=[a for a in atoms(p) if a['key'] in ['outdooruse','outdoorsafe','indooroutdooruse','indooroutdoor','indooruse','outdoorinstallation']]
    states=[]
    for a in fields:
        if a['key'] in ['outdooruse','outdoorsafe'] and a['value'] in ['yes','no']:
            if target=='outdoor':states.append('MATCH' if a['value']=='yes' else 'CONTRADICTION')
        elif a['key']=='indooruse' and a['value'] in ['yes','no'] and target=='indoor':states.append('MATCH' if a['value']=='yes' else 'CONTRADICTION')
        elif target in a['value']:states.append('CONTRADICTION' if ('not suitable' in a['value'] or ('indoor only' in a['value'] and target=='outdoor')) else 'MATCH')
    text=norm(p['title']+' '+p['class']+' '+p['description'])
    if target=='outdoor':
        if re.search(r'indoor(?: use)? only|not (?:suitable|intended|recommended) for outdoor',text):states.append('CONTRADICTION')
        elif re.search(r'\boutdoors?\b|\bpatio\b',text):states.append('MATCH')
    elif re.search(r'\bindoors?\b',text):states.append('MATCH')
    return dict(status=reduce_status(states),reason='Explicit intended-use wording only; weather-resistant alone is insufficient',raw_fields=fields)

def counts(ids,states):return {s:sum(states[p]==s for p in ids) for s in STATES}
def bounds(a,b,states):
    a,b=set(a),set(b);x,y=counts(a,states),counts(b,states);l,g=counts(a-b,states),counts(b-a,states)
    u=lambda c:c['UNKNOWN']+c['AMBIGUOUS']
    loose=[y['MATCH']-x['MATCH']-u(x),y['MATCH']+u(y)-x['MATCH']]
    tight=[g['MATCH']-l['MATCH']-u(l),g['MATCH']+u(g)-l['MATCH']]
    assert loose[0]<=tight[0]<=tight[1]<=loose[1]
    return dict(before=x,after=y,lost=l,gained=g,confirmed_delta=y['MATCH']-x['MATCH'],
                loose_bounds=loose,tight_bounds=tight,proven_decline=tight[1]<0,proven_increase=tight[0]>0,
                disappearance=x['MATCH']>0 and y['MATCH']==y['UNKNOWN']==y['AMBIGUOUS']==0)

def joint_status(attribute_states,type_state,use_states,scope_narrowed=False,unassessed=False):
    s=list(attribute_states)+[type_state]+list(use_states)
    if 'CONTRADICTION' in s:return 'CONTRADICTION'
    if 'AMBIGUOUS' in s or scope_narrowed:return 'AMBIGUOUS'
    if 'UNKNOWN' in s or unassessed:return 'UNKNOWN'
    return 'MATCH'

def metric(top,labels,highest):
    gain={'Exact':3,'Partial':1,'Irrelevant':0}
    g=[gain.get(labels.get(p),0) for p in top]
    d=1/np.log2(np.arange(2,len(top)+2));dcg=float(np.dot(g,d))
    ideal=sorted((gain[v] for v in labels.values()),reverse=True)[:len(top)]
    idcg=float(np.dot(ideal,d[:len(ideal)]))
    return dict(relevant_count=len(set(top)&highest),recall=len(set(top)&highest)/len(highest),dcg=dcg,ndcg=dcg/idcg,
        idcg=idcg,gain_sequence=g,qrel_counts={label:sum(labels.get(p,'Unjudged')==label for p in top) for label in ['Exact','Partial','Irrelevant','Unjudged']})
