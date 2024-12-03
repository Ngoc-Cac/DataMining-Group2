from math import ceil
from copy import deepcopy

from pandas import DataFrame


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
    minFreq = ceil(min_support * len(df[transact_col].unique()))

    vert_df = vert_df[vert_df[transact_col].apply(len) >= minFreq]
    
    equiv_classes = [[]]
    freq_itemset = []
    for item, transacts in vert_df.iterrows():
        equiv_classes[0].append(([item], transacts.iloc[0]))
        freq_itemset.append(([item], len(transacts.iloc[0])))

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
                freq_itemset.append((new_itemset, len(new_transacts)))
            
            if next_class: equiv_classes.append(next_class)
    return DataFrame(freq_itemset, columns=['itemsets', 'frequencies'])

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