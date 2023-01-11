import os
import numpy as np


def evaluate_metrics(model, train_data, test_data, metrics):
    print(f'Evaluating metrics {metrics} ...')

    user_embs = model.user_weights
    item_embs = model.item_weights
    train_items_dic = train_data.user_items_dic
    test_items_dic = test_data.user_items_dic
    test_user_ids = list(test_data.user_items_dic.keys())
    metric_callers = []
    max_top_k = 0
    for metric in metrics:
        try:
            metric_callers.append(eval(metric))
            max_top_k = max(max_top_k, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))
    
    sim_matrix = np.dot(user_embs, item_embs.T)

    # for i, test_user_id in enumerate(test_user_ids):
    for test_user_id in test_user_ids:
        train_items = train_items_dic[test_user_id]
        # remove clicked items in train data
        sim_matrix[test_user_id, train_items] = -np.inf 

    item_indices = np.argpartition(-sim_matrix, max_top_k)[:, 0:max_top_k]
    sim_matrix = sim_matrix[np.arange(item_indices.shape[0])[:, None], item_indices]
    sorted_ids = np.argsort(-sim_matrix, axis=1)
    top_k_items = item_indices[np.arange(sorted_ids.shape[0])[:, None], sorted_ids]
    true_items = [test_items_dic[test_user_id] for test_user_id in test_user_ids]
    results = [[fn(top_k_items, true_items) for fn in metric_callers] \
                    for top_k_items, true_items in zip(top_k_items, true_items)]
    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    print('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


class Recall(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        hit_items = set(true_items) & set(top_k_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class NormalizedRecall(object):
    """Recall metric normalized to max 1."""
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        hit_items = set(true_items) & set(top_k_items)
        recall = len(hit_items) / min(self.top_k, len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        hit_items = set(true_items) & set(top_k_items)
        precision = len(hit_items) / (self.top_k + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, top_k_items, true_items):
        p = self.precision_k(top_k_items, true_items)
        r = self.recall_k(top_k_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(top_k_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        dcg_fn = DCG(k=self.top_k)
        idcg = dcg_fn(true_items[:self.top_k], true_items)
        dcg = dcg_fn(top_k_items, true_items)
        return dcg / (idcg + 1e-12)


class MRR(object):
    """MRR metric"""
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        true_items = set(true_items)
        mrr = 0
        for i, item in enumerate(top_k_items):
            if item in true_items:
                mrr += 1 / (i + 1.0)
        return mrr


class HitRate(object):
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        hit_items = set(true_items) & set(top_k_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate


class MAP(object):
    """
    Calculate mean average precision.
    """
    def __init__(self, k=1):
        self.top_k = k

    def __call__(self, top_k_items, true_items):
        top_k_items = top_k_items[:self.top_k]
        true_items = set(true_items)
        pos = 0
        precision = 0
        for i, item in enumerate(top_k_items):
            if item in true_items:
                pos += 1
                precision += pos / (i + 1.0)
        return precision / (pos + 1e-12)

