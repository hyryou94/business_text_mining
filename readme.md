# 이 git은 crawling과 analysis로 두 파트로 나누어져 있습니다.

## Directories

### Crawling

1. chrome_driver : 크롬 드라이버가 있습니다. (windows and linux) 

2. crawling_codes : 크롤링 코드가 있습니다. 베이킹과 베이킹도구 관련 크롤링코드가 나누어져 있으며 중간에 멈췄을 시 이어서 크롤링하는 코드도 추가되어있습니다.

3. error_logs : 중간에 멈췄을 시 이어서 크롤링할 때 어느 위치에서 시작해야하는지 URL을 기록하여 저장하는 폴더입니다.

### Analysis

1. nlp_data : 수업시간에 제공된 불용어사전입니다.

2. data_baking : 베이킹 관련 질문게시판을 크롤링한 raw_data입니다.

3. data_equip : 베이킹도구 관련 질문게시판을 크롤링한 raw_data입니다.

4. parsed_data : 2, 3번의 데이터를 토큰화 및 클렌징한 결과데이터입니다.

## Codes

1. helper_nlp.py : 분석에 필요한 과정을 함수화해둔 코드입니다.

2. nlp_analysis.py : gensim을 이용하여 분석을 진행한 코드입니다.

3. nlp_analysis.py : sklearn을 이용하여 분석을 진행한 코드입니다.

## 실행방법

1. 기본적으로 nlp_analysis2.py만 보시면 됩니다.