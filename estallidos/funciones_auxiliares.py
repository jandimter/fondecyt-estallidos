#@title Funciones auxiliares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from typing import List, Set, Tuple
from itertools import product
from operator import itemgetter
nltk.download('punkt')

def filter_text(df: DataFrame, column: str) -> DataFrame:
    """
    Function that removes special characters, translate some characters and
    convert the text to lowercase, returns a DataFrame with a new column with the filtered data
    ### Parameters
    df: DataFrame,
        DataFrame to which its column is to be filtered
    column: str,
        name of the DataFrame column to filter
    ### Returns
    df_filtered: DataFrame,
        DataFrame copy with a new column with filtered data
    ### Examples
    Filter characters from a DataFrame
    ```
    >>> df = pd.DataFrame({
    ...     'col' : ['HELLO', '12 Days', 'He said: "bonjour"']
    ... })
    >>> df_filtered = filter_text(
    ...          df,
    ...          'col'
    ... )
    >>> df_filtered
        col                 col_filtered
    0   HELLO hello         hello
    1   12 Days              days
    2   He said: "bonjour"  he said bonjour
    ```
    """
    # Creating a copy of the dataframe and creating the new column
    df_filtered = df.copy()
    df_filtered.loc[:, column + "_filtered"] = df.loc[:, column]

    # Removing special characters
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(
        lambda x: re.sub("[()`'_�@|#.,!¡¿?:%-]", '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[\']', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[/]', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[\\\\]', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[+]', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[$]', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '["]', '', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[][]', '', str(x)))

    # Translating some characters
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[贸]', 'ó', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[茅]', 'é', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[谩]', 'á', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[帽]', 'ñ', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[铆]', 'í', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[聽]', ' ', str(x)))
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: re.sub(
        '[煤]', 'ú', str(x)))

    # Lowercasing the text
    df_filtered.loc[:, column + "_filtered"] = df_filtered.loc[:, column + "_filtered"].map(lambda x: x.lower())
    return df_filtered


def tokenize_data(text_data: Series, language: str = 'spanish', stopwords_file: str = None) -> List[Set[str]]:
    """
    Function that tokenize the given text, removes stopwords and words with low frequency, returns a list of sets
    ### Parameters
    text_data: Series,
        Series to tokenize
    language: str,
        default stopwords download language
    stopwords_file: str, default None,
        PATH to file with custom stopwords
    ### Returns
    data_tokenized: List of Sets,
        list of sets with tokenized and filtered text
    ### Examples
    Tokenizing and removing default stopwords from a Series
    ```
    >>> df = pd.DataFrame({
    ...     'col_filtered' : ['hello good', 'days bad bad', 'he said bonjour']
    ... })
    >>> data_tokenized = tokenize_data(
    ...          df['col_filtered'],
    ...          'english'
    ... )
    >>> data_tokenized
    [{'good', 'hello'},
    {'bad', 'days'},
    {'bonjour', 'said'}]
    ```
    Tokenizing and removing custom stopwords from a Series
    ```
    >>> PATH/TO/FILE/custom_stopwords.csv
    hello
    days
    said
    joy
    >>> df = pd.DataFrame({
    ...     'col_filtered' : ['hello good', 'days bad bad', 'he said bonjour']
    ... })
    >>> data_tokenized = tokenize_data(
    ...          df['col_filtered'],
    ...          'english',
    ...          'PATH/TO/FILE/custom_stopwords.csv'
    ... )
    >>> data_tokenized
    [{'good'},
    {'bad'},
    {'bonjour'}]
    ```
    """
    nltk.download('stopwords')
    # Tokenizing the text
    data_tokenized = [word_tokenize(text) for text in text_data]

    # Loading spanish stopwords
    custom_stopwords = stopwords.words('spanish')
    if stopwords_file is not None:
        local_stopwords = pd.read_csv('spanish_stopwords.txt', header=None)
        custom_stopwords.extend(list(local_stopwords.iloc[:, 0]))

    # Removing stopwords
    data_tokenized = [[word for word in text if word not in custom_stopwords] for text in data_tokenized]

    # Removing low frequency words
    frequency = defaultdict(int)
    for text in data_tokenized:
        for token in text:
            frequency[token] += 1

    data_tokenized = [set([token for token in text if frequency[token] > 2]) for text in data_tokenized]

    return data_tokenized

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

def get_constituyente_news(
    keywords: List[Tuple[Set[str], int]], constituyentes_csv: str
) -> List[Tuple[Set[str], int, Set[str], int, str]]:
    """
    Function that extracts the set of keywords from constituyentes news, and also extracts the mentioned constituyente
    and its position in the top of sets of keywords, returns a list of sets of keywords with the position in the top
    and the mentioned constituyente
    ### **Parameters**
    keywords: List of tuples,
        List of news headlines keyword sets
    constituyentes_csv: str,
        PATH to the file with the names of the constituyentes
    ### **Returns**
    const_news: List of tuples,
        returns a list of tuples with the keywords that define the topic, the frequency,
        all topic keywords, position in the top and name of the mentioned constituyente
    ### **Examples**
    ```
    >>> PATH/TO/FILE/constituyentes.csv
    Jorge Perez
    Lucia Gonzales
    >>> keywords = [({'jorge', 'perez'}, 1), ({'good', 'hello'}, 1)]
    >>> const_news = get_constituyente_news(
    ...     keywords,
    ...     'PATH/TO/FILE/constituyentes.csv'
    ... )
    >>> const_news
    [({'jorge', 'perez'}, 1)]
    ```
    """
    constituyentes = pd.read_csv(constituyentes_csv)
    # Lowercasing the names and creating a set for each name
    constituyentes.loc[:, 'set_name'] = [set(n.lower().split()) for n in constituyentes.name]
    const_news = []
    for i in range(len(constituyentes)):
        const_set = constituyentes.loc[i, 'set_name']
        const_name = constituyentes.loc[i, 'name']
        for news in keywords:
            inter = news[2].intersection(const_set)
            if len(inter) >= len(const_set):
                news = (news[0], news[1], news[2], news[3], const_name, True)
                const_news.append(news)
    const_news = sorted(const_news, key=lambda tup: tup[1], reverse=True)
    return const_news

def constituyente_inter(
    constituyentes_keywords: List[Tuple[Set[str], int, Set[str], int, str]],
    news_title: Set[str], news_keywords: List[Tuple[Set[str], int, Set[str]]],
    top: int = 10
) -> Tuple[List[Tuple[Set[str], int]], str, bool, int]:
    """
    Aux function that gets the news that mention a set of keywords, differentiating between constituyente
    news and indicating if it is in the top N of news, returns a list with a tuple with the keywords and
    thefrequency, the name of the constituyente and if it is in the top N or not
    ### Parameters
    constituyentes_keywords: List of tuples,
        list of tuples with the keywords that define the topic,
        the frequency, all topic keywords, position in the top and name of the mentioned constituyente
    news_title: Set of strings,
        news headline set
    news_keywords: List of tuples,
        list of tuples with the keywords that define the topic,
        the frequency and all topic keywords
    top: int, default 10,
        amount of news from the top to show
    ### Returns
    return: Tuple,
        tuple with the keywords of the topic, repetitions, name of the constituyente and position in the top
    """
    for i in constituyentes_keywords:
        if len(i[0] & news_title) >= 4:
            if i[3] <= (top - 1):
                return (i[2], i[4], True, int(i[3]))
            else:
                return (i[2], i[4], False, int(i[3]))
    for j in range(len(news_keywords)):
        if len(news_keywords[j][0] & news_title) >= 4:
            if j <= (top - 1):
                return (news_keywords[j][2], np.nan, True, int(news_keywords[j][3]))
            else:
                return (news_keywords[j][2], np.nan, False, int(news_keywords[j][3]))
    return (np.nan, np.nan, np.nan, np.nan)

def extract_constituyente_news(
    constituyentes_keywords: List[Tuple[Set[str], int, Set[str], int, str]], news_dataset: DataFrame,
    news_keywords: List[Tuple[Set[str], int, Set[str]]], top: int = 10
) -> DataFrame:
    """
    Function that identifies the news through the set of keywords, returns a DataFrame with the
    information of the news, the keywords that identify it, the name of the constituyente if it is
    is a constituyente news and its position in the top N
    ### **Parameters**
    constituyentes_keywords: List of tuples,
        list of tuples with the keywords that define the topic,
        frequency, all topic keywords, position in the top and name of the mentioned constituyente
    news_dataset: DataFrame,
        DataFrame with the news info
    news_keywords: List of tuples,
        list of tuples with the keywords that define the topic,
        frequency and all topic keywords
    top: int, default 10,
        amount of news from the top to show
    ### **Returns**
    :unique_news: DataFrame,
        DataFrame with the information the news top N and constituyente news,
        their set of keywords, their place in the ranking and the name of the constituyente in case
        one or more appears in the headline
    ### **Examples**
    ```
    >>> constituyentes_keywords = [({'jorge', 'perez'}, 1)]
    >>> news_dataset = get_news(
    ...          '2021-07-14',
    ...          '2021-07-20',
    ...          'PATH_TO_CREDENTIALS'
    ... )
    >>> news_keywords = [
    ...     ({'tornado', 'usa'},500),
    ...     ({'chile', 'win'}, 20),
    ...     ({'jorge', 'perez'}, 1)
    ... ]
    >>> unique_news = extract_constituyente_news(
    ...          constituyentes_keywords,
    ...          news_dataset,
    ...          news_keywords
    ... )
    >>> unique_news
    # Returns the info related to the tornado and jorge perez, leaving chile out
    ```
    ```
    >>> constituyentes_keywords = [({'jorge', 'perez'}, 1)]
    >>> news_dataset = get_news(
    ...          '2021-07-14',
    ...          '2021-07-20',
    ...          'PATH_TO_CREDENTIALS'
    ... )
    >>> news_keywords = [
    ...     ({'tornado', 'usa'},500),
    ...     ({'chile', 'win'}, 20),
    ...     ({'jorge', 'perez'}, 1)
    ... ]
    >>> unique_news = extract_constituyente_news(
    ...          constituyentes_keywords,
    ...          news_dataset,
    ...          news_keywords,
    ...          1
    ... )
    >>> unique_news
    # Returns the info related to the tornado chile and jorge perez
    ```
    """
    # Creating a set with the words of the news headlines
    news_dataset.loc[:, 'news_title_set'] = [set(n.lower().split()) for n in news_dataset.title]
    news_dataset.loc[:, ['keywords', 'constituyente', 'top', 'topic_pos']] = [
        constituyente_inter(
            constituyentes_keywords,
            news_title, news_keywords, top) for news_title in news_dataset.loc[:, 'news_title_set']]

    # Filtering only by the news that are in the top N or are constituent
    df_const = news_dataset[(pd.notna(news_dataset.constituyente)) | (news_dataset.top)]
    unique_news = df_const.drop_duplicates(subset=['title', 'date'])

    # Filtering by columns of interest
    unique_news = unique_news.loc[
        :, ['uri', 'title', 'body', 'date', 'dateTimePub', 'url', 'constituyente', 'keywords', 'top', 'topic_pos']]
    unique_news.reset_index(drop=True, inplace=True)
    return unique_news
