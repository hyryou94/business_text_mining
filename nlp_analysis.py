import os

import nltk
import gensim

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from nltk.stem.porter import *

from wordcloud import WordCloud
from collections import Counter, defaultdict

from gensim import corpora
from gensim.models import CoherenceModel

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, flatten

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

# Parameters
parsed = False

if not parsed:
    # Data Loading
    baking_data, equip_data = data_loading()

    baking_data = tokenization(cleansing(baking_data))
    equip_data = tokenization(cleansing(equip_data))

    # 텍스트 클렌징
    baking_data.to_json('parsed_data/parsed_baking.json', orient='table')
    equip_data.to_json('parsed_data/parsed_equip.json', orient='table')

else:
    baking_data, equip_data = parsed_data_loading()

# 워드 클라우드
data1 = text_data['text tokenized'].tolist()


# TF-IDF
data3 = []
for i in data1:
    if len(i) == 0:
        continue
    string = i[0]
    for w in i[1:]:
        string += " "
        string += w
    data3.append(string)

vectorizer = TfidfVectorizer()
sp_matrix = vectorizer.fit_transform(data3)

word2id = defaultdict(lambda: 0)  # error발생시  0으로 출력

for idx, feature in enumerate(vectorizer.get_feature_names()):
    word2id[feature] = idx

tfidf = []
for i, sent in enumerate(data3):
#    print(f'====={i}번째 뉴스======')
#    print([(token, sp_matrix[i, word2id[token]]) for token in sent.split()])
    tfidf.append([(token, sp_matrix[i, word2id[token]]) for token in sent.split()])

# LDA
dictionary = corpora.Dictionary(data1)
corpus = [dictionary.doc2bow(text) for text in data1]

###################
lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=8,
                                      id2word=dictionary)

# 대표단어 5개씩 뽑기
lda.print_topics(num_words=5)
lda.get_document_topics(corpus)[0]


# topic 간 유사도
values = []
coherence_value = []
for i in range(2, 10):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                               num_topics=i,
                                               id2word=dictionary)
    values.append(ldamodel.log_perplexity(corpus))

    coherence_model_lda = CoherenceModel(model=ldamodel, texts=data1, dictionary=dictionary, topn=10)
    coherence_value.append(coherence_model_lda.get_coherence())

x = range(2, 10)
plt.figure()
plt.plot(x, values)
plt.xlabel('Number of Topics')
plt.ylabel('Score')
plt.show()

plt.figure()
plt.plot(x, coherence_value)
plt.xlabel('Number of Topics')
plt.ylabel('Score')
plt.show()
'''
# 클러스터링 - Ward
sp_matrix1 = vectorizer.fit_transform(data3)
df1 = pd.DataFrame(sp_matrix1.toarray(), columns=vectorizer.get_feature_names())

cluster = AgglomerativeClustering(n_clusters=3,
                                  linkage='ward')
result = cluster.fit_predict(df1)

df_ward = text_data.copy()
df_ward['클러스터'] = list(result)
df_ward.loc[df_ward['클러스터'] == 2, 'text']

# 계층적 군집분석의 시각화
plt.figure(figsize=(10, 7))
dend = shc.dendrogram(shc.linkage(df1, method='ward'))
plt.show()

# K-means Clustering
kmeans = KMeans(n_clusters=3).fit(df1)

df_means = text_data.copy()
df_means['클러스터'] = list(kmeans.labels_)
df_means.head(5)

pca = PCA(n_components=2)
principal = pca.fit_transform(df1)
principal_df = pd.DataFrame(data=principal, columns=['PCA1', 'PCA2'])

plt.scatter(principal_df.iloc[kmeans.labels_ == 0, 0],
            principal_df.iloc[kmeans.labels_ == 0, 1], c='red', label='cluster1')
plt.scatter(principal_df.iloc[kmeans.labels_ == 1, 0],
            principal_df.iloc[kmeans.labels_ == 1, 1], c='blue', label='cluster2')
plt.scatter(principal_df.iloc[kmeans.labels_ == 2, 0],
            principal_df.iloc[kmeans.labels_ == 2, 1], c='green', label='cluster3')
plt.legend()
plt.show()
'''
