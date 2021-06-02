from helper_nlp import parsed_data_loading, tf_idf_gensim, pass_opt, lda_param_opt

baking_data, equip_data = parsed_data_loading()
baking_corpus, baking_dictionary, baking_data = tf_idf_gensim(baking_data, drop_one_letter=True)

pass_result = pass_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                       min_pass=1, max_pass=40)
pass_result.to_csv('parameter_pass.csv', encoding='ms949')
topic_num_result = lda_param_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data, pass_value=7,
                                 min_topic=1, max_topic=10)
topic_num_result.to_csv('parameter_topic.csv', encoding='ms949')
