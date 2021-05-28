import os

import gensim
from gensim.models.ldamodel import LdaModel

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf, pass_opt, lda_param_opt

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

# Data Loading
baking_data, equip_data = data_loading()

# 텍스트 클렌징
baking_data = tokenization(cleansing(baking_data))
baking_data.to_json('parsed_data/parsed_baking.json', orient='table')

equip_data = tokenization(cleansing(equip_data))
equip_data.to_json('parsed_data/parsed_equip.json', orient='table')