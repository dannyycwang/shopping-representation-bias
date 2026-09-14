"""Lossless deterministic serialization prototypes, not LLM semantic rewrites."""
import re
from collections import Counter, defaultdict

FIELDS = ['product_name', 'product_class', 'category hierarchy', 'product_description', 'product_features']

def features(raw):
    # Preserve duplicate keys, empty keys, units and contradictory values.
    return [part.strip() for part in raw.split('|') if part.strip()]

def atoms(row):
    out = [(k, row[k]) for k in FIELDS[:-1] if row[k]]
    out += [(f'product_features[{i}]', value) for i, value in enumerate(features(row['product_features']))]
    return out

def build(row, condition):
    name, cls, cat, desc, feat = [row[k] for k in FIELDS]
    original = '\n'.join(x for x in [name, cls, cat, desc, feat] if x)
    ordered = sorted(features(feat), key=str.casefold)
    normalized = '\n'.join([f'product name: {name}', f'product class: {cls}',
                            f'category hierarchy: {cat}', 'attributes:', *ordered,
                            f'source description: {desc}'])
    if condition == 'R0': return original
    if condition == 'R2':
        return 'Explore this product. Discover the details below.\n' + original + '\nTake a closer look at the product details.'
    if condition == 'R3':
        return '\n'.join([f'The product name is {name}.', f'The product class is {cls}.',
                          f'The category hierarchy is {cat}.', f'The source description is: {desc}',
                          'The listed product attributes follow. ' + '. '.join(features(feat))])
    if condition == 'R4':
        # Exactly the same BM25 token multiset as R0: intentional negative control.
        return '\n'.join([name, cls, cat, desc] + ['- ' + x for x in features(feat)])
    if condition == 'R5': return normalized
    if condition == 'R8': return original + '\n' + normalized
    if condition == 'C_repeat': return original + '\n' + original
    if condition == 'C_order':
        return '\n'.join(x for x in [name, cls, cat, desc, '|'.join(feat.split('|')[::-1])] if x)
    raise ValueError(condition)

def audit(row, text):
    expected = atoms(row)
    missing = [source for source, value in expected if value not in text]
    residual = text
    for source, value in sorted(expected, key=lambda x:len(x[1]), reverse=True):
        residual = residual.replace(value, '')
    scaffold = set(tokens('the product name is class category hierarchy source description listed attributes follow explore this discover details below take a closer look at'))
    unsupported = set(tokens(residual)) - scaffold
    source_numbers = set(re.findall(r'\d+(?:\.\d+)?', ' '.join(v for _,v in expected)))
    novel_numbers = set(re.findall(r'\d+(?:\.\d+)?',text)) - source_numbers
    groups = defaultdict(set)
    malformed = 0
    for item in features(row['product_features']):
        key, sep, value = item.partition(':')
        if not sep or not key.strip(): malformed += 1
        else: groups[key.strip().casefold()].add(value.strip().casefold())
    return {'source_atoms': len(expected), 'missing_atoms': len(missing),
            'conflicting_keys': sum(len(v) > 1 for v in groups.values()),
            'malformed_features': malformed,
            'source_numeric_tokens': len(re.findall(r'\d+(?:\.\d+)?', ' '.join(v for _, v in expected))),
            'unsupported_residual_tokens':len(unsupported), 'novel_numeric_values':len(novel_numbers),
            'pass': not missing and not unsupported and not novel_numbers}

def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())
