from enum import Enum
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
from collections import defaultdict
from IPython.display import display, clear_output
from numba import jit, njit
import pickle 
import pandas as pd
import json
import os
from tqdm.auto import tqdm
import copy
import ast
from itertools import permutations

def filter_histories(df, canonicalize=True):
    """Add cleaning columns to df — does not drop rows or build any
    plotting state.

    Adds:
      - 'history_canon_counts_str' (when canonicalize=True): each history's
        count multiset reduced to its lex-min representative under
        (arm-label permutation) x (global outcome-label permutation).
        Histories sharing a belief state collapse to the same key.
      - 'disagree': True when the ells disagree on `best_a` within the
        row's (t, canonical-history) group (or (t, history_counts_str)
        if canonicalize=False).

    't' is coerced to int; rows where the coercion fails (e.g. a stray
    corrupted CSV row) are dropped.
    """

    def _parse_counts(hc):
        if isinstance(hc, str):
            try:
                return ast.literal_eval(hc)
            except Exception:
                return ()
        return hc or ()

    out = df.copy()
    out = out[pd.to_numeric(out['t'], errors='coerce').notna()]
    out['t'] = out['t'].astype(int)

    p_cols = [c for c in out.columns if str(c).startswith('p_') and str(c)[2:].isdigit()]
    n_arms = len(p_cols)

    canon_cache = {}

    def _canon_counts_str(counts_tuple):
        key = tuple(sorted(counts_tuple)) if counts_tuple else ()
        if key in canon_cache:
            return canon_cache[key]
        counts = dict(counts_tuple) if counts_tuple else {}
        if not counts:
            canon_cache[key] = ('', {a: a for a in range(n_arms)})
            return canon_cache[key]
        arms_used = sorted({a for a, _ in counts})
        outcomes_used = sorted({o for _, o in counts})
        best = None
        best_arm_map = None
        for arm_perm in permutations(range(len(arms_used))):
            arm_map = dict(zip(arms_used, arm_perm))
            for out_perm in permutations(range(len(outcomes_used))):
                out_map = dict(zip(outcomes_used, out_perm))
                s = '-'.join(
                    f'a{a}o{o}:{c}'
                    for (a, o), c in sorted(
                        ((arm_map[a], out_map[o]), c)
                        for (a, o), c in counts.items()
                    )
                )
                if best is None or s < best:
                    best = s
                    best_arm_map = arm_map
                    
        used_orig = set(best_arm_map.keys())
        unused_orig = [a for a in range(n_arms) if a not in used_orig]
        used_canon = set(best_arm_map.values())
        unused_canon = [c for c in range(n_arms) if c not in used_canon]
        
        full_arm_map = best_arm_map.copy()
        for o, c in zip(unused_orig, unused_canon):
            full_arm_map[o] = c
            
        canon_to_orig = {c: o for o, c in full_arm_map.items()}
        
        canon_cache[key] = (best, canon_to_orig)
        return canon_cache[key]

    if canonicalize:
        results = out['history_counts'].apply(
            lambda hc: _canon_counts_str(_parse_counts(hc))
        )
        out['history_canon_counts_str'] = results.apply(lambda r: r[0])
        arm_maps = results.apply(lambda r: r[1]).tolist()
        
        if n_arms > 0:
            p_matrix = out[[f'p_{i}' for i in range(n_arms)]].values
            q_matrix = out[[f'Q_{i}' for i in range(n_arms)]].values
            delta_matrix = out[[f'delta_emp_{i}' for i in range(n_arms)]].values
            entropy_matrix = out[[f'entropy_{i}' for i in range(n_arms)]].values
            
            for a in range(n_arms):
                orig_indices = np.array([m.get(a, a) for m in arm_maps])
                out[f'canon_p_{a}'] = p_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_Q_{a}'] = q_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_delta_emp_{a}'] = delta_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_entropy_{a}'] = entropy_matrix[np.arange(len(out)), orig_indices]

            ## check which Q is the best out of the canon_Qs and Q_terminate
            if f'Q_terminate' in out.columns:
                termination_idx = n_arms 
                out[f'canon_best_a'] = out[[f'canon_Q_{i}' for i in range(n_arms)] + [f'Q_terminate']
                                        ].idxmax(axis=1).apply(lambda s: int(s.split('_')[2]) if s.startswith('canon_Q') else termination_idx)
            else:
                out[f'canon_best_a'] = out[[f'canon_Q_{i}' for i in range(n_arms)]].idxmax(axis=1).apply(lambda s: int(s.split('_')[2]))
            

                
        group_col = 'history_canon_counts_str'
        disagree = out.groupby(['t', group_col])['canon_best_a'].nunique() > 1
    else:
        group_col = 'history_counts_str'
        disagree = out.groupby(['t', group_col])['best_a'].nunique() > 1


    out = out.merge(
        disagree.rename('disagree').reset_index(), on=['t', group_col]
    )

    ## sanity check: for each history_canon_counts_str, canon_p_a.nunique() should be 1 for each a, and canon_Q_a.unique() should be 1 for each a
    for a in range(n_arms):
        for ell in out['ell'].unique():
            subset = out[out['ell'] == ell]

            ## round to 5 dp to avoid floating point issues 
            subset[f'canon_p_{a}'] = subset[f'canon_p_{a}'].round(5)
            subset[f'canon_Q_{a}'] = subset[f'canon_Q_{a}'].round(5)
            p_nunique = subset.groupby(['t', 'history_canon_counts_str'])[f'canon_p_{a}'].nunique()
            q_nunique = subset.groupby(['t', 'history_canon_counts_str'])[f'canon_Q_{a}'].nunique()

        assert (p_nunique <= 1).all(), f"p_{a} not unique within canon groups: {p_nunique[p_nunique > 1]}"
        assert (q_nunique <= 1).all(), f"Q_{a} not unique within canon groups: {q_nunique[q_nunique > 1]}"

    ## define choice probabilities wrt/ canonicalised actions, e.g. p0 is p(choose a0), where a0 is the first action in the history_counts
    return out