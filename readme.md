# Business Text Mining : Baking QnA Analysis by Ho-Young Ryou

한양대학교 경영일반대학원 석박사 5기 재무금융전공 2019144749 류호영 

* 이 git은 crawling과 analysis로 두 파트로 나누어져 있습니다.

## Directories

### Crawling

1. chrome_driver : 크롬 드라이버가 있습니다. (windows and linux) 

2. error_logs : 중간에 멈췄을 시 이어서 크롤링할 때 어느 위치에서 시작해야하는지 URL을 기록하여 저장하는 폴더입니다.

### Analysis

1. nlp_data : 수업시간에 제공된 불용어사전입니다.

2. data_baking : 베이킹 관련 질문게시판을 크롤링한 raw_data입니다.

3. others : gensim으로 분석을 진행하였을 때 parameter optimizing을 위해 짠 코드입니다. Sklearn으로 변경하며 실효성은 없어졌습니다.

## Codes

1. helper_nlp.py : 분석에 필요한 과정을 함수화해둔 코드입니다.

2. helper_crawling.py : 크롤링에 필요한 과정을 함수화해둔 코드입니다.

3. nlp_analysis.py : 분석을 진행한 코드입니다.

4. crawling_linux_baking_v2.py : 크롤링 과정을 진행하는 코드입니다.

5. crawling_linux_baking_v2_cont.py : 크롤링 진행 중 중단시 해당 부분부터 이어서 진행하는 코드입니다.

## 실행방법

1. 크롤링은 위에 써있듯이 두 가지 버전이 있습니다. 원래는 두 게시판을 크롤링하여 네 가지 버젼이 있었으나 분석에서는 baking만을 사용하였기 때문에 baking 게시판 크롤링만 남겨두었습니다. 

2. 분석은 nlp_analysis.py만 보시면 됩니다.

## Appendix

현재 남아있는 코드들은 실제 결과물 추출에 사용된 코드들만 남아있습니다. 실제 시도했던 모든 코드는 staging 서버에 있습니다.