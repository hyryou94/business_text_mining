import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.models.ldamodel import LdaModel

from sklearn.feature_extraction.text import TfidfVectorizer

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf_gensim, pass_opt, lda_param_opt

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

baking_data, equip_data = data_loading()

nouns = True
tokenizer = 'kiwi'

# 텍스트 클렌징
if nouns:
    baking_data = tokenization(cleansing(baking_data), tokenizer)
    baking_data.to_json('parsed_data/parsed_baking_%s.json' % tokenizer, orient='table')

    equip_data = tokenization(cleansing(equip_data), tokenizer)
    equip_data.to_json('parsed_data/parsed_equip_%s.json' % tokenizer, orient='table')

else:
    baking_data = tokenization(cleansing(baking_data), nouns=nouns)
    baking_data.to_json('parsed_data/parsed_baking_not_nouns.json', orient='table')

    equip_data = tokenization(cleansing(equip_data), nouns=nouns)
    equip_data.to_json('parsed_data/parsed_equip_not_nouns.json', orient='table')

