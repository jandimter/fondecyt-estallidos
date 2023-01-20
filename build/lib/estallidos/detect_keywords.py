from typing import List, Set, Tuple
import numpy as np
from collections import defaultdict
from itertools import product
from operator import itemgetter
"""
Extracted from @mquezada's repository. https://github.com/mquezada/twitter-event-detection
"""


def intersection_n(ha: Set[str], hb: Set[str], threshold: int = 2) -> bool:
    """
    Aux function of detect keywords, returns a Boolean in the case that the intersection
    between two Sets is greater than or equal to the threshold

    ### Parameters
    ha: Set,
        first set to intersect
    hb: Set,
        second set to intersect
    threshold: Int,
        default 2, minimum length of the set resulting from the intersection

    ### Returns
    return: bool,
        returns True in the case that the Set resulting from the intersection is greater than the threshold
    """
    if ha == hb:
        return False
    return len(ha & hb) >= threshold


def detect_keywords(tokens_sets: List[Set[str]], threshold: int = 4) -> List[Tuple[Set[str], int, Set[str], int]]:
    """
    Detects common keywords among the headlines, based on the supplementary material PLOS One,
    returns a list of tuples with the set of keywords and their frequency

    ### Parameters

    tokens_sets: List of Sets,
        list with a set for each text that you want to detect keywords

    threshold: int,
        default 4, minimum length of the set of keywords per text

    ### Returns

    keywords_union: List of tuples,
        returns a list of tuples with the set of keywords and their frequency

    ### Examples
    Detecting keywords from token set
    ```
    >>> tokens_sets = [{'bad', 'thief'}, {'good', 'hello', 'joy', 'quarry'}]
    >>> keywords_union = detect_keywords(
    ...     tokens_sets
    ... )
    >>> keywords_union
    [({'good', hello, 'joy', 'quarry'}, 1)]
    ```
    Detecting a custom number of keywords per topic
    ```
    >>> tokens_sets = [{'bad', 'thief'}, {'good', 'hello', 'joy', 'quarry'}]
    >>> keywords_union = detect_keywords(
    ...     tokens_sets,
    ...     2
    ... )
    >>> keywords_union
    [({'bad', 'thief'}, 1),
    ({'good', 'hello'}, 1)]
    ```
    """
    print("Detecting keywords...")
    h = tokens_sets
    candidates = list()
    scores = list()

    headlines_pairs = filter(lambda x: intersection_n(x[0], x[1], threshold), product(h, h))

    for ha, hb in headlines_pairs:
        g = ha & hb

        if not candidates:
            candidates.append(g)
            scores.append(defaultdict(int))
            for w in candidates[0]:
                scores[0][w] = 1

        j = np.argmax([len(candidate & g) for candidate in candidates])

        if len(candidates[j] & g) >= threshold:
            candidates[j] = candidates[j] & g
            for w in candidates[j]:
                scores[j][w] += 1
        else:
            candidates.append(g)
            scores.append(defaultdict(int))
            for w in candidates[-1]:
                scores[-1][w] = 1

    print("Merging similar topics...")
    total_scores = [sum(score.values()) for score in scores]
    keywords = sorted(zip(candidates, total_scores), key=itemgetter(1), reverse=True)
    # Merging the sets with similar keywords
    keywords_union = keywords.copy()
    range_keywords = range(len(keywords))
    ind_to_delete = []
    for i in range_keywords:
        for j in range_keywords:
            if j <= i or j in ind_to_delete or i in ind_to_delete:
                continue
            else:
                tmp_keywords = keywords.copy()
                inter = tmp_keywords[i][0].intersection(tmp_keywords[j][0])
                if len(inter) >= int(threshold/2):
                    keywords_union[i] = (
                        keywords_union[i][0].union(tmp_keywords[j][0]),
                        keywords_union[i][1] + keywords_union[j][1],
                        tmp_keywords[i][0].union(tmp_keywords[j][0])
                        )
                    if not j == i:
                        ind_to_delete.append(j)
    ind_to_delete.sort(reverse=True)
    # Eliminating the indexes that were merged
    for i in ind_to_delete:
        keywords_union.pop(i)
    keywords_union = sorted(keywords_union, key=lambda tup: tup[1], reverse=True)
    # Setting all tuples to default 3 len
    for k in range(len(keywords_union)):
        if len(keywords_union[k]) < 3:
            keywords_union[k] = (
                keywords_union[k][0],
                keywords_union[k][1],
                keywords_union[k][0]
            )
    # Adding their position on the top
    for i in range(len(keywords_union)):
        keywords_union[i] = (keywords_union[i][0], keywords_union[i][1], keywords_union[i][2], i)
    print("Keywords detected successfully")
    return keywords_union
