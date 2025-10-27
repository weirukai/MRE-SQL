import math

import numpy as np
from workflow.sql_meta_info import SQLMetaInfo
from collections import defaultdict, Counter
import random
from workflow.system_state import SystemState


def select_by_cluster_size(clusters, valid_classes):

    cluster_sizes = {cls: len(clusters[cls]) for cls in valid_classes}
    max_size = max(cluster_sizes.values(), default=0)
    candidates = [cls for cls, size in cluster_sizes.items() if size == max_size]
    return random.choice(candidates) if candidates else None


def arbitration_process(final_scores, clusters, method_score_list):

    if not final_scores:
        return None
    max_score = max(final_scores.values())
    candidates = [cls for cls, score in final_scores.items() if score == max_score]

    if len(candidates) == 1:
        return candidates[0]

    cluster_sizes = {cls: len(clusters[cls]) for cls in candidates if cls in clusters}
    if cluster_sizes:
        max_size = max(cluster_sizes.values())
        candidates = [cls for cls in cluster_sizes if cluster_sizes[cls] == max_size]
        if len(candidates) == 1:
            return candidates[0]
    method_support = defaultdict(int)
    for cls in candidates:
        for method_scores in method_score_list:
            if cls in method_scores and method_scores[cls] > 0:
                method_support[cls] += 1
    if method_support:
        max_support = max(method_support.values())
        candidates = [cls for cls in method_support if method_support[cls] == max_support]
    return random.choice(candidates) if candidates else None


def select_best_sql(best_class, clusters, args):
    if not best_class:
        return None
    candidates = [sql_meta_info.SQL for sql_meta_info in clusters.get(best_class, []) if
                  isinstance(sql_meta_info, SQLMetaInfo)]
    if getattr(args, 'data_mode', '') and 'spider' in args.data_mode:
        # no distinct sql first to avoid evaluation error in spider.
        non_distinct_sqls = [sql for sql in candidates if 'DISTINCT' not in sql.upper()]
        if non_distinct_sqls:
            return random.choice(non_distinct_sqls)
    return random.choice(candidates) if candidates else None








def uncertainty_based_ensemble(state: SystemState, clusters, BS_Selections, PS_Selections, GS_Selections,
                               LS_Selections,
                               UT_Selections, args):
    """
    clusters: dict {sql_result: [sql1, sql2,...]}
    BS_Selections: defaultdict {sql_result: pairwise_score}
    PS_Selections: dict {sql_result: [valid_sqls]}
    GS_Selections: dict {sql_result: [groupwise_scores]}
    LS_Selections: dict {sql_result: [ls_scores]}
    UT_Selections: defaultdict {sql_result: ut_score}
    """

    BS_Selections = BS_Selections if isinstance(BS_Selections, (defaultdict, dict)) else defaultdict(float)
    PS_Selections = PS_Selections if isinstance(PS_Selections, dict) else {}
    GS_Selections = GS_Selections if isinstance(GS_Selections, dict) else {}
    LS_Selections = LS_Selections if isinstance(LS_Selections, dict) else {}
    UT_Selections = UT_Selections if isinstance(UT_Selections, (defaultdict, dict)) else defaultdict(float)

    def is_valid_class(cls):
        return (
                cls.strip() not in ('[]', '[ ]', '')
                and len(clusters.get(cls, [])) > 0
        )

    valid_classes = [cls for cls in clusters.keys() if is_valid_class(cls)]
    if not valid_classes:
        return None

    bs_scores = {}
    for cls in valid_classes:
        score = BS_Selections.get(cls, 0.0)
        if isinstance(score, (int, float)):
            bs_scores[cls] = float(score)

    ps_scores = {}
    for cls in valid_classes:
        valid_sqls = PS_Selections.get(cls, [])
        if isinstance(valid_sqls, list):
            ps_scores[cls] = len([sql for sql in valid_sqls if sql in clusters.get(cls, [])])

    gs_scores = {}
    for cls in valid_classes:
        scores = GS_Selections.get(cls, [])
        if isinstance(scores, list):

            gs_scores[cls] = len(scores)

    ls_scores = {}
    for cls in valid_classes:
        scores = LS_Selections.get(cls, [])
        if isinstance(scores, list):
            ls_scores[cls] = len(scores)

    ut_scores = {}
    for cls in valid_classes:
        score = UT_Selections.get(cls, 0.0)
        if isinstance(score, (int, float)):
            ut_scores[cls] = float(score)

    def calculate_uncertainty(scores_dict, method_type):
        if not scores_dict:
            return 1.0
        scores = list(scores_dict.values())
        if method_type == "BS":
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) < 2:
                return 0.0 if scores[0] > 0 else 1.0
            max_gap = sorted_scores[0] - sorted_scores[1]
            denominator = sorted_scores[0] if sorted_scores[0] != 0 else 1e-6
            return 1 - (max_gap / denominator)

        elif method_type == "UT":
            total_score = sum(scores) or 1e-6
            probs = [s / total_score for s in scores]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 1
            return entropy / max_entropy if max_entropy > 0 else 0.0

        elif method_type == "PS":
            total = sum(scores)
            return 1 - (max(scores) / total) if total > 0 else 1.0

        elif method_type in ("GS", "LS"):
            total_rounds = sum(scores)
            if total_rounds == 0:
                return 1.0
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) < 2:
                return 0.0 if sorted_scores[0] > 0 else 1.0
            max_gap = sorted_scores[0] - sorted_scores[1]
            return 1 - (max_gap / total_rounds)

        return 1.0

    uncertainty_BS = calculate_uncertainty(bs_scores, "BS")
    uncertainty_PS = calculate_uncertainty(ps_scores, "PS")
    uncertainty_GS = calculate_uncertainty(gs_scores, "GS")
    uncertainty_LS = calculate_uncertainty(ls_scores, "LS")
    uncertainty_UT = calculate_uncertainty(ut_scores, "UT")

    raw_weights = {
        'BS': max(0, 1 - uncertainty_BS),
        'PS': max(0, 1 - uncertainty_PS),
        'GS': max(0, 1 - uncertainty_GS),
        'LS': max(0, 1 - uncertainty_LS),
        'UT': max(0, 1 - uncertainty_UT)
    }

    total_weight = sum(raw_weights.values())

    if total_weight < 1e-6:
        return select_by_cluster_size(clusters, valid_classes)

    norm_weights = {k: v / total_weight for k, v in raw_weights.items()}

    def minmax_normalize(scores, NUM_CANDIDATE, NUM_TRY, NUM_TEST, method):
        if not scores:
            return {}

        if method == "BS":
            maxv = NUM_CANDIDATE - 1
        elif method == "PS":
            maxv = NUM_CANDIDATE
        elif method in ["GS", "LS"]:
            maxv = NUM_TRY
        elif method == "UT":
            maxv = NUM_TEST
        else:
            maxv = max(scores.values())
        minv = 0
        normalized = {}
        for k, v in scores.items():
            normalized[k] = (v - minv) / (maxv - minv)
        return normalized

    norm_BS = minmax_normalize(bs_scores, args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST, "BS")
    norm_PS = minmax_normalize(ps_scores, args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST, "PS")
    norm_GS = minmax_normalize(gs_scores, args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST, "GS")
    norm_LS = minmax_normalize(ls_scores, args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST, "LS")
    norm_UT = minmax_normalize(ut_scores, args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST, "UT")
    final_scores = defaultdict(float)
    all_candidates = set().union(*[d.keys() for d in [norm_BS, norm_PS, norm_GS, norm_LS, norm_UT]])

    for cls in all_candidates:
        if cls not in valid_classes:
            continue
        bs = norm_BS.get(cls, 0) * norm_weights['BS']
        ps = norm_PS.get(cls, 0) * norm_weights['PS']
        gs = norm_GS.get(cls, 0) * norm_weights['GS']
        ls = norm_LS.get(cls, 0) * norm_weights['LS']
        ut = norm_UT.get(cls, 0) * norm_weights['UT']
        final_scores[cls] = bs + ps + gs + ls + ut
    state.SQL_meta_infos['ensemble_scores'] = final_scores
    best_class = arbitration_process(final_scores, clusters,
                                     [bs_scores, ps_scores, gs_scores, ls_scores, ut_scores])

    return select_best_sql(best_class, clusters, args)
