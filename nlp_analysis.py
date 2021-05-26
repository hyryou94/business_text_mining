import os

import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

# Parameters
parsed = True

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

# TF-IDF
baking_corpus, baking_dictionary, baking_data = tf_idf(baking_data)
equip_corpus, equip_dictionary, equip_data = tf_idf(equip_data)

# topic 간 유사도
values = []
coherence_value = []
for i in range(2, 40):
    print(i)
    ldamodel = gensim.models.LdaMulticore(baking_corpus, num_topics=i, id2word=dictionary,
                                          passes=20, workers=16)
    values.append(ldamodel.log_perplexity(baking_corpus))

    coherence_model_lda = CoherenceModel(model=ldamodel, texts=baking_data['text_tokenized'],
                                         dictionary=baking_dictionary, topn=10)
    coherence_value.append(coherence_model_lda.get_coherence())

x = range(2, 40)
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

perplexity = pd.DataFrame(values, index=range(2, 40), columns=['perplexity'])
coherence = pd.DataFrame(coherence_value, index=range(2, 40), columns=['coherence'])
parameter_tuning = pd.concat([perplexity, coherence], axis=1)



vectorizer1 = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()

sp_matrix1 = vectorizer1.fit_transform(baking_data['joined tokens'])
sp_matrix2 = vectorizer2.fit_transform(equip_data['joined tokens'])

baking_tf_idf_sum = pd.DataFrame(sp_matrix1.sum(0), index=['sum'], columns=vectorizer1.get_feature_names()).T
baking_01, baking_99 = baking_tf_idf_sum.quantile(0.01), baking_tf_idf_sum.quantile(0.99)
upper_outlier = baking_tf_idf_sum[baking_tf_idf_sum['sum'] >= baking_99.values[0]]
lower_outlier = baking_tf_idf_sum[baking_tf_idf_sum['sum'] <= baking_01.values[0]]
filtered = baking_tf_idf_sum[(baking_tf_idf_sum['sum'] > baking_01.values[0]) & (baking_tf_idf_sum['sum'] < baking_99.values[0])]
filtered_sparse = sp_matrix1.T[(baking_tf_idf_sum['sum'] > baking_01.values[0]) & (baking_tf_idf_sum['sum'] < baking_99.values[0])].T

# LDA
#dictionary = corpora.Dictionary(filtered.index.values)
#corpus = [dictionary.doc2bow(text) for text in data1]

###################
from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=10, learning_method='online')
lda_model.fit_transform(sp_matrix1)


def get_topics(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])

get_topics(lda_model.components_,vectorizer1.get_feature_names())


lda_model = gensim.models.ldamodel.LdaModel(num_topics=8, learning_method='online')
lda_model.fit(sp_matrix1)

# 대표단어 5개씩 뽑기
lda.print_topics(num_words=5)
lda.get_document_topics(corpus)[0]



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
