import json
import os
import re

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from kiwipiepy import Kiwi
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ['JAVA_HOME'] = '/usr/bin/java'


def data_loading():
    raw_data1 = pd.read_json('data_baking/baking_v2.json', orient='table').dropna()
    raw_data2 = pd.read_json('data_baking/baking_v2_cont.json', orient='table').dropna()
    raw_data1.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...
    raw_data2.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    raw_data3 = pd.read_json('data_equip/crawled_texts_baking_equip.json', orient='table').dropna()
    raw_data4 = pd.read_json('data_equip/equip_v2_cont.json', orient='table').dropna()
    raw_data4.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    baking_data = pd.concat([raw_data1, raw_data2], ignore_index=True)
    equip_data = pd.concat([raw_data3, raw_data4], ignore_index=True)

    return baking_data, equip_data


def parsed_data_loading(nouns=True, tokenizer='kiwi'):
    if nouns:
        baking_data = pd.read_json('parsed_data/parsed_baking_%s.json' % tokenizer, orient='table')
        equip_data = pd.read_json('parsed_data/parsed_equip_%s.json' % tokenizer, orient='table')
    else:
        baking_data = pd.read_json('parsed_data/parsed_baking_not_nouns.json', orient='table')
        equip_data = pd.read_json('parsed_data/parsed_equip_not_nouns.json', orient='table')
    return baking_data, equip_data


def clean_text(text):
    # 필요없는 부분 제거
    text = re.sub('http[s]?://\S+', '', text)  # http url 제거
    text = re.sub('\S*@\S*\s?', '', text)  # 기자 emails 제거
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호안에 텍스트 제거 : 뉴스이름 + 기자이름
    text = re.sub(r"\w*\d\w*", '', text)  # 숫자 포함하는 텍스트 제거
    text = re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '', text)  # 특수문자 이모티콘 제거
    text = re.sub("\n", '', text)  # 개행문자 제거
    text = re.sub("\xa0", '', text)  # 개행문자 제거
    text = re.sub(r'Copyright .* rights reserved', '', text)  # "Copyright all rights reserved" 제거
    return text


def preprocessing(text):
    # 특수문자나 이모티콘 등 아래의 특수기호들을 제거합니다(%등은 남김).
    text = re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '', text)
    # 위에서 특수문자를 제거한 text에서 한글, 영문만 남기고 모두 제거하도록 합니다.
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]", ' ', text)
    return text


def cleansing(df):
    df = df.drop_duplicates(subset='본문', keep='first')
    del_content1 = '=============================\n일상의 작은 행복\n베이킹은 오븐엔조이와 함께~\n\n서로의 작은 배려로 멋진 카페를 만들어보아요 ^_^b\n\n◇오븐엔조이 기본 에티켓◇\nhttps://cafe.naver.com/delonghi/book5100010\n============================='
    del_content2 = '타카페 또는 블로그의 이벤트, 공동구매, 행사등의 공유는 금지되어있습니다.\n게시물에 주요내용이 없는 외부링크 공유 게시물은 삭제될 수 있습니다.\n원활한 카페 운영을 위해 이해와 협조 부탁드립니다.'
    df['clean title'] = df['제목'].apply(clean_text)
    df['clean text'] = df['본문'].apply(lambda x: clean_text(x.replace(del_content1, '').replace(del_content2, '')))
    return df


# 형태소 분석
def tokenization(df, nouns=True, tokenizer='kiwi'):
    if tokenizer == 'okt':
        okt = Okt()

        if nouns:
            target_title = [okt.nouns(doc) for doc in df['clean title']]
            target_text = [okt.nouns(doc) for doc in df['clean text']]

        else:
            target_title = [okt.morphs(doc) for doc in df['clean title']]
            target_text = [okt.morphs(doc) for doc in df['clean text']]
    else:
        kiwi = Kiwi(num_workers=16)
        kiwi.prepare()

        temp_title = [[each_word[0] if ('NNG' in each_word[1]) or ('NNP' in each_word[1])
                       else each_word[0] + '다' if ('VV' in each_word[1]) or ('VA' in each_word[1])
                       else None for each_word in each_doc[0][0]]
                      for each_doc in kiwi.analyze(df['clean title'], top_n=1)]

        target_title = [[each_word for each_word in each_doc if each_word] for each_doc in temp_title]

        temp_text = [[each_word[0] if ('NNG' in each_word[1]) or ('NNP' in each_word[1])
                      else each_word[0] + '다' if ('VV' in each_word[1]) or ('VA' in each_word[1])
                      else None for each_word in each_doc[0][0]]
                     for each_doc in kiwi.analyze(df['clean title'], top_n=1)]

        target_text = [[each_word for each_word in each_doc if each_word] for each_doc in temp_text]

    # 불용어 제거
    with open('nlp_data/korean_stopwords.json', encoding='utf-8') as f:
        stopwords = json.load(f)
    stopwords.extend(['에서', '고', '이다', '는', '이', '가', '한', '씨', '"', '에', '요', '걸', '더', '케', '거', '분',
                      'ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅜㅜ', 'ㅠㅜ', 'ㅠㅠ', 'ㅠㅠㅠ', 'ㅠㅠㅠㅠ'])

    df['title_tokenized'] = target_title
    df['title_tokenized'] = df['title_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])

    df['text_tokenized'] = target_text
    df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])
    return df


# TF-IDF
def tf_idf_gensim(df, drop_one_letter=False):
    if drop_one_letter:
        df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [word for word in x if len(word) > 1])
    df['joined tokens'] = df['text_tokenized'].apply(
        lambda x: str.join(' ', x).replace(
            '오븐 엔조이', '오븐엔조이').replace(
            '휘 낭시', '휘낭시에'). replace(
            '베이 킹', '베이킹').replace(
            '스 메그', '스메그'))
    df['text_tokenized'] = df['joined tokens'].apply(lambda x: x.split(' '))

    dictionary = gensim.corpora.Dictionary(df['text_tokenized'])
    bow_corpus = [dictionary.doc2bow(doc) for doc in df['text_tokenized']]
    tfidf = gensim.models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return tfidf_corpus, dictionary, df


def tf_idf_sklearn(df, drop_one_letter=False):
    if drop_one_letter:
        df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [word for word in x if len(word) > 1])
    df['joined tokens'] = df['text_tokenized'].apply(
        lambda x: str.join(' ', x).replace(
            '오븐 엔조이', '오븐엔조이').replace(
            '휘 낭시', '휘낭시에').replace(
            '베이 킹', '베이킹').replace(
            '스 메그', '스메그'))
    df['text_tokenized'] = df['joined tokens'].apply(lambda x: x.split(' '))

    vectorizer = TfidfVectorizer(min_df=0.01)
    sparse_matrix = vectorizer.fit_transform(df['joined tokens'])
    return vectorizer, sparse_matrix, df


def drop_certain_words(corpus, sparse_matrix, drop_words):
    drop_words_index = [np.where(corpus == word)[0][0] for word in drop_words]
    to_keep = sorted(set(range(sparse_matrix.shape[1])) - set(drop_words_index))
    corpus = corpus[to_keep]
    sparse_matrix = sparse_matrix[:, to_keep]
    return corpus, sparse_matrix


def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        important_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

        print("Topic %d:" % topic_idx)
        print(" ".join(important_words))
        topics.append(important_words)
    return topics


# LDA
# Pass optimization
def pass_opt(corpus, dictionary, data, min_pass, max_pass):
    perplexity_value = []
    coherence_value = []

    for p in range(min_pass, max_pass):
        print(p)
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=20, id2word=dictionary, passes=p, workers=16)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=data['text_tokenized'],
                                             dictionary=dictionary, topn=10, coherence='c_v')
        perplexity_value.append(ldamodel.log_perplexity(corpus))
        coherence_value.append(coherence_model_lda.get_coherence())

    # graphing_opt_res(coherence_value, perplexity_value, min_pass, max_pass)

    perplexity = pd.DataFrame(perplexity_value, index=range(min_pass, max_pass), columns=['perplexity'])
    coherence = pd.DataFrame(coherence_value, index=range(min_pass, max_pass), columns=['coherence'])
    parameter_tuning = pd.concat([perplexity, coherence], axis=1)

    return parameter_tuning


# Num_topic optimization
def lda_param_opt(corpus, dictionary, data, pass_value=20, min_topic=2, max_topic=40):
    perplexity_value = []
    coherence_value = []
    for i in range(min_topic, max_topic + 1):
        print(i)
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=i, id2word=dictionary,
                                              passes=pass_value, workers=16)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=data['text_tokenized'],
                                             dictionary=dictionary, topn=10, coherence='c_v')
        perplexity_value.append(ldamodel.log_perplexity(corpus))
        coherence_value.append(coherence_model_lda.get_coherence())

    # graphing_opt_res(coherence_value, perplexity_value, min_topic, max_topic)

    perplexity = pd.DataFrame(perplexity_value, index=range(min_topic, max_topic+1), columns=['perplexity'])
    coherence = pd.DataFrame(coherence_value, index=range(min_topic, max_topic+1), columns=['coherence'])
    parameter_tuning = pd.concat([perplexity, coherence], axis=1)

    return parameter_tuning


# Graphing
def graphing_opt_res(coherence_value, perplexity_value, min_value, max_value):
    x = range(min_value, max_value)
    plt.figure()
    plt.plot(x, perplexity_value)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.show()

    plt.figure()
    plt.plot(x, coherence_value)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.show()


# Document labeling
def doc_labeling(df, corpus, model):
    lda_topics = sorted(model.print_topics(num_words=11), key=lambda x: x[0])
    topic_result = pd.DataFrame(
        [[word.split('*')[1].replace('\"', '') for word in topic[1].split(' + ')] for topic in lda_topics])

    df['topic'] = [model[each_doc] for each_doc in corpus]
    df['label w/ prob'] = df['topic'].apply(lambda x: max(x, key=lambda y: y[1]))
    df['topic label'] = df['label w/ prob'].apply(lambda x: x[0])
    df['topic prob'] = df['label w/ prob'].apply(lambda x: x[1])
    return df, topic_result


def time_series_analysis(df):
    df['날짜'] = pd.to_datetime(df['날짜'])
    period_df = df.copy()
    period_df.index = period_df['날짜'].values
    period_df['날짜'] = [date.date() for date in period_df.index]
    period_df['count'] = np.ones(len(period_df))

    whole_period = period_df[['topic label', 'count']].groupby('topic label').sum().sort_values('count',
                                                                                                ascending=False)
    whole_period_percentage = whole_period / whole_period.sum()

    time_series = period_df[['날짜', 'topic label', 'count']].groupby(['날짜', 'topic label']).sum().reset_index()
    time_series_result = time_series.pivot_table(columns=['topic label'], index=['날짜'], values='count').fillna(0)
    time_series_result.index = pd.to_datetime(time_series_result.index)
    monthly_result = time_series_result.resample('3M').sum()
    monthly_percentage = (monthly_result.T / monthly_result.sum(1)).T

    plt.plot(monthly_percentage)
    plt.legend(monthly_percentage.columns)
    plt.show()

    return whole_period_percentage, monthly_percentage


def second_lda(df, cluster_num):
    cluster_df = df[df['topic label'] == cluster_num].copy()
    cluster_df['text_tokenized'] = cluster_df['title_tokenized'] + cluster_df['text_tokenized']
    cluster_corpus, cluster_dictionary, cluster_df = tf_idf_gensim(cluster_df, drop_one_letter=False)
    cluster_model = gensim.models.LdaMulticore(cluster_corpus, num_topics=5, id2word=cluster_dictionary,
                                               passes=7, workers=16, random_state=1)

    second_lda_topics = sorted(cluster_model.print_topics(num_words=10), key=lambda x: x[0])
    second_topics = pd.DataFrame(
        [[word.split('*')[1].replace('\"', '') for word in topic[1].split(' + ')] for topic in second_lda_topics])

    cluster_df['second topic'] = [cluster_model[each_doc] for each_doc in cluster_corpus]
    cluster_df['second label w/ prob'] = cluster_df['second topic'].apply(lambda x: max(x, key=lambda x: x[1]))
    cluster_df['second topic label'] = cluster_df['second label w/ prob'].apply(lambda x: x[0])
    cluster_df['second topic prob'] = cluster_df['second label w/ prob'].apply(lambda x: x[1])
    return cluster_df, second_topics
