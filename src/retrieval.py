import json
import hashlib
from collections import Counter
import numpy as np
from scipy import sparse
from .representations import tokens

def bm25(texts, queries, k1=1.2, b=0.75):
    vocab = {}
    rows, cols, counts, lengths = [], [], [], []
    for i, text in enumerate(texts):
        counter = Counter(tokens(text)); lengths.append(sum(counter.values()))
        for term, count in counter.items():
            j = vocab.setdefault(term, len(vocab))
            rows.append(i); cols.append(j); counts.append(count)
    tf = sparse.csr_matrix((np.asarray(counts, dtype=np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
    dl = np.asarray(lengths, dtype=np.float32)
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log(1 + (len(texts) - df + .5) / (df + .5))
    norm = k1 * (1 - b + b * dl / max(dl.mean(), 1))
    weighted = tf.copy()
    weighted.data = ((k1+1)*tf.data / (tf.data + np.repeat(norm, np.diff(tf.indptr))) * idf[tf.indices]).astype(np.float32)
    matrix = weighted.tocsc()
    scores = np.zeros((len(queries), len(texts)), dtype=np.float32)
    for i, query in enumerate(queries):
        indices = [vocab[t] for t in sorted(set(tokens(query))) if t in vocab]
        if indices: scores[i] = np.asarray(matrix[:, indices].sum(axis=1)).ravel()
    return scores

def rank(scores):
    # Catalog sorted by product_id; ties are always broken by ascending product_id.
    return np.argsort(-scores, axis=1, kind='stable').astype(np.int32)

def inverse(order):
    result = np.empty_like(order)
    np.put_along_axis(result, order, np.broadcast_to(np.arange(1, order.shape[1]+1), order.shape), axis=1)
    return result

def hybrid(bm_order, dense_order, constant):
    # Full-list RRF avoids censoring exact-pair rank movements at an arbitrary depth.
    return 1/(constant + inverse(bm_order).astype(np.float32)) + 1/(constant + inverse(dense_order).astype(np.float32))

class Encoder:
    def __init__(self, config, cache):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch; self.cfg = config; self.cache = cache
        torch.manual_seed(config['seed']); torch.set_num_threads(4)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'])
        self.model = AutoModel.from_pretrained(config['model'], revision=config['model_revision']).to(self.device).eval()
        if self.device == 'cuda': self.model.half()
        print('Encoder ready: ' + self.device, flush=True)

    def encode(self, texts, name):
        import time
        torch = self.torch
        precision = 'fp16_model_fp32_pooling' if self.device == 'cuda' else 'fp32'
        fingerprint = hashlib.sha256(('\0'.join(texts) + json.dumps(self.cfg, sort_keys=True) + precision + 'encoder_v2').encode()).hexdigest()[:16]
        path = self.cache / f'{name}_{fingerprint}.npy'
        if path.exists(): return np.load(path)
        start = time.time()
        outputs = []; sizes = []
        # Every source token is encoded. No first-256 truncation confound.
        # Each 254-token chunk gets CLS/SEP. Pool chunk means weighted by content token count.
        for offset in range(0, len(texts), 256):
            block = texts[offset:offset+256]
            ids = self.tokenizer(block, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
            chunks, owners, weights = [], [], []
            for i, item in enumerate(ids):
                sizes.append(len(item))
                for pos in range(0, max(len(item), 1), self.cfg['chunk_tokens']):
                    part = item[pos:pos+self.cfg['chunk_tokens']]
                    chunks.append([self.tokenizer.cls_token_id] + part + [self.tokenizer.sep_token_id])
                    owners.append(i); weights.append(max(len(part), 1))
            agg = np.zeros((len(block), self.model.config.hidden_size), dtype=np.float32)
            denom = np.zeros(len(block), dtype=np.float32)
            for pos in range(0, len(chunks), self.cfg['batch_size']):
                end = pos + self.cfg['batch_size']
                batch = self.tokenizer.pad({'input_ids': chunks[pos:end]}, padding=True, return_tensors='pt')
                batch = {k:v.to(self.device) for k,v in batch.items()}
                with torch.inference_mode():
                    hidden = self.model(**batch).last_hidden_state.float()
                    mask = batch['attention_mask'].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                values = pooled.float().cpu().numpy()
                w = np.asarray(weights[pos:end], dtype=np.float32)
                np.add.at(agg, owners[pos:end], values * w[:,None]); np.add.at(denom, owners[pos:end], w)
            agg /= np.maximum(denom[:,None], 1)
            agg /= np.maximum(np.linalg.norm(agg, axis=1, keepdims=True), 1e-12)
            outputs.append(agg)
            if offset % 4096 == 0:
                print(f'{name}: {min(offset+256,len(texts))}/{len(texts)}, {time.time()-start:.0f}s', flush=True)
        result = np.concatenate(outputs)
        np.save(path, result)
        path.with_suffix('.json').write_text(json.dumps({'fingerprint':fingerprint,'model':self.cfg['model'],
            'revision':self.cfg['model_revision'], 'device':self.device, 'mean_wordpieces':float(np.mean(sizes)),
            'fraction_above_254':float(np.mean(np.asarray(sizes)>254)), 'precision':'fp16_model_fp32_pooling' if self.device=='cuda' else 'fp32', 'seconds':time.time()-start},indent=2))
        return result
