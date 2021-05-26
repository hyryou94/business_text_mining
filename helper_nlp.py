import os
import re
import json

import pandas as pd

from konlpy.tag import Okt

os.environ['JAVA_HOME'] = '/usr/bin/java'


def data_loading():
    raw_data1 = pd.read_json('data_baking/baking_v2.json', orient='table').dropna()
    raw_data2 = pd.read_json('data_baking/baking_v2_cont.json', orient='table').dropna()
    raw_data1.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...
    raw_data2.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    raw_data3 = pd.read_json('data_baking_equip/crawled_texts_baking_equip.json', orient='table').dropna()
    raw_data4 = pd.read_json('data_baking_equip/equip_v2_cont.json', orient='table').dropna()
    raw_data4.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    baking_data = pd.concat([raw_data1, raw_data2], ignore_index=True)
    equip_data = pd.concat([raw_data3, raw_data4], ignore_index=True)

    return baking_data, equip_data


def parsed_data_loading():
    baking_data = pd.read_json('parsed_data/parsed_baking.json', orient='table')
    equip_data = pd.read_json('parsed_data/parsed_equip.json', orient='table')
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


def flatten(l):
    flatlist = []
    for elem in l:
        if type(elem) == list:
            for e in elem:
                flatlist.append(e)
        else:
            flatlist.append(elem)
    return flatlist


def cleansing(df):
    df = df.drop_duplicates(subset='본문', keep='first')
    df['clean text'] = df['본문'].apply(clean_text)
    return df


# 형태소 분석
def tokenization(df):
    okt = Okt()
    # texts_ko = [okt.morphs(doc) for doc in df['clean text']]
    text_noun = [okt.nouns(doc) for doc in df['clean text']]

    # 불용어 제거
    with open('korean_stopwords.json', encoding='utf-8') as f:
        stopwords = json.load(f)
    stopwords.extend(['에서', '고', '이다', '는', '이', '가', '한', '씨', '"'])

    df['text_tokenized'] = text_noun
    df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])
    return df
