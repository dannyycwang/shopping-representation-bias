"""Uniform query exposure with deterministic rotations of graded judged comparisons."""
import random
from common_corrected import RELATIONS


class BalancedSampler:
    def __init__(self, groups, seed, presentations=64):
        self.seed, self.presentations = seed, presentations
        self.groups, self.relations = {}, {}
        rng = random.Random(seed)
        for qid in sorted(groups):
            valid = [(a,b,w) for a,b,w in RELATIONS if groups[qid][a] and groups[qid][b]]
            if not valid:
                continue
            rng.shuffle(valid)
            self.relations[qid] = valid
            self.groups[qid] = {}
            for label in sorted(groups[qid]):
                items = list(groups[qid][label])
                rng.shuffle(items)
                self.groups[qid][label] = items

    def epoch(self, epoch):
        assert epoch>=1
        order = []
        rng = random.Random(self.seed*10000+epoch)
        # One visit to every eligible query in each random permutation cycle.
        for cycle in range(self.presentations):
            ids = sorted(self.groups)
            rng.shuffle(ids)
            for qid in ids:
                t = (epoch-1)*self.presentations+cycle
                relations = self.relations[qid]
                hi,lo,weight = relations[t%len(relations)]
                occurrence = t//len(relations)
                a,b = self.groups[qid][hi],self.groups[qid][lo]
                # Both grade pools rotate on every presentation of this relation.
                order.append(dict(query_id=qid,positive=a[occurrence%len(a)],negative=b[occurrence%len(b)],
                                  higher_label=hi,lower_label=lo,weight=weight,epoch=epoch,cycle=cycle))
        return order
