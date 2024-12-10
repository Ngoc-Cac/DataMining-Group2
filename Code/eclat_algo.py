from math import ceil
from copy import deepcopy
from collections import Counter
from itertools import chain, combinations

from pandas import DataFrame

from typing import Iterable


def powerset(iterable: Iterable):
    """powerset([1,2,3]) → (1,) (2,) (3,) (1,2) (1,3) (2,3)
    Adapted from: https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    return chain.from_iterable(combinations(iterable, r) for r in range(1, len(iterable)))

def _vertical_transform(df: DataFrame, item_col: str) -> tuple[DataFrame, str]:
    vert_df = df.groupby(item_col)\
                .aggregate(set)
    transact_col = vert_df.columns[0]

    vert_df = vert_df.sort_values(transact_col, axis=0,
                                  key=lambda series: series.apply(len))
    return vert_df, transact_col

def _find_common_prefix(itemset1: list, itemset2: list) -> list:
    if len(itemset1) != len(itemset2):
        raise ValueError("itemsets have different length!")
    
    for i, item in enumerate(itemset1):
        if item != itemset2[i]: break
    
    if i - 1 < 0: return []
    else: return deepcopy(itemset1[:i])

def eclat(df: DataFrame, min_support: float, item_col: str, *,
          max_iter: int = 100_000) -> DataFrame:
    vert_df, transact_col = _vertical_transform(df, item_col)
    total_transactions = len(df[transact_col].unique())
    minFreq = ceil(min_support * total_transactions)

    vert_df = vert_df[vert_df[transact_col].apply(len) >= minFreq]
    
    equiv_classes = [[]]
    freq_itemset = []
    for item, transacts in vert_df.iterrows():
        equiv_classes[0].append(([item], transacts.iloc[0]))
        freq_itemset.append(([item], len(transacts.iloc[0]),
                             len(transacts.iloc[0]) / total_transactions))

    while equiv_classes and (max_iter := max_iter - 1):
        cur_class = equiv_classes.pop()

        for i, (itemset, transacts) in enumerate(cur_class[:-1]):
            next_class = []
            for itemset2, transacts2 in cur_class[(i + 1):]:
                new_transacts = transacts & transacts2
                if len(new_transacts) < minFreq: continue
                
                new_itemset = _find_common_prefix(itemset, itemset2)
                new_itemset.append(itemset[-1])
                new_itemset.append(itemset2[-1])
                new_itemset = sorted(new_itemset, key=lambda item: len(vert_df.loc[item][transact_col]))

                next_class.append((new_itemset, new_transacts))
                freq_itemset.append((new_itemset, len(new_transacts),
                                     len(new_transacts) / total_transactions))
            
            if next_class: equiv_classes.append(next_class)
    return DataFrame(freq_itemset, columns=['itemsets', 'frequencies', 'support'])

def find_conf(itemsets: Iterable[list[str]],
              frequent_df: DataFrame) -> DataFrame:
    rules_df = []
    rules = []
    for itemset in itemsets:
        for antecedent in powerset(itemset):
            rules.append((antecedent, tuple(item for item in itemset if not item in antecedent)))
    
    for rule in rules:
        rule_mask = frequent_df['itemsets'].apply(lambda lis: Counter(lis) == Counter(rule[0] + rule[1]))
        try:
            rule_freq = frequent_df[rule_mask]['frequencies'].values[0]
            rule_sup = frequent_df[rule_mask]['support'].values[0]
        except IndexError: raise ValueError(f'Frequent itemset {rule[0] + rule[1]} not found')

        antecedent_mask = frequent_df['itemsets'].apply(lambda lis: Counter(lis) == Counter(rule[0]))
        try: antecedent_freq = frequent_df[antecedent_mask]['frequencies'].values[0]
        except IndexError: raise ValueError(f'Frequent itemset {rule[0]} not found')

        consequent_mask = frequent_df['itemsets'].apply(lambda lis: Counter(lis) == Counter(rule[1]))
        try: consequent_sup = frequent_df[consequent_mask]['support'].values[0]
        except IndexError: raise ValueError(f'Frequent itemset {rule[1]} not found')

        conf = rule_freq / antecedent_freq
        rules_df.append((rule, rule_freq, rule_sup,
                         conf, conf / consequent_sup))
    return DataFrame(rules_df, columns=['rule', 'frequencies', 'support', 'confidence', 'lift'])

if __name__ == "__main__":
    # freq itemsets:
    # (['butter'], 2)
    # (['eggs'], 2)
    # (['bread'], 3)
    # (['milk'], 3)
    # (['cheese'], 4)
    # (['butter', 'cheese'], 2)
    # (['eggs', 'cheese'], 2)
    # (['bread', 'milk'], 2)
    # (['bread', 'cheese'], 2)
    # (['milk', 'cheese'], 2)
    test = [(1, 'bread'), (2, 'bread'), (5, 'bread'),
        (1, 'milk'), (4, 'milk'), (5, 'milk'),
        (2, 'cheese'), (3, 'cheese'), (4, 'cheese'), (5, 'cheese'),
        (2, 'butter'), (4, 'butter'),
        (3, 'eggs'), (4, 'eggs')]
    test_df = DataFrame(test, columns=['index', 'item'])
    eclat(test_df, .4, 'item')