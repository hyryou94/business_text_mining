# Settings
import os
from selenium import webdriver
import time

import pandas as pd
from bs4 import BeautifulSoup as bs

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# ID, PWD
my_id = 'hobbang1994'
my_pwd = 'r945106'
path = '/home/hyryou94/crawling'

# Input
file_path = os.path.join(path, '../../data_baking', 'crawled_texts_baking2.json')

# Headless
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

# Login
driver = webdriver.Chrome(os.path.join(path, '../../chrome_driver/chromedriver'), options=options)
login_url = 'https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com'
driver.get(login_url)

driver.execute_script("document.getElementsByName('id')[0].value = \'" + my_id + "\'")
driver.execute_script("document.getElementsByName('pw')[0].value = \'" + my_pwd + "\'")
time.sleep(1)

driver.find_element_by_id('log.login').click()
time.sleep(1)

# Crawling
target_url = 'https://cafe.naver.com/delonghi?iframe_url=%2FArticleRead.nhn%3Fclubid%3D10526290%26articleid%3D779418%26referrerAllArticles%3Dfalse%26menuid%3D436%26page%3D1%26boardtype%3DL'
driver.get(target_url)
time.sleep(1)

driver.switch_to.frame('cafe_main')

users = []
dates = []
counts = []
titles = []
body = []
comments_num = []
comments = []
scores = []

# 첫번쨰 게시물 선택
check = 0
for k in range(200):
    for t in range(250):
        try:
            driver.implicitly_wait(10)

            # 날짜, 조회수, 댓글 개수 수집
            dates.append(driver.find_element_by_css_selector('span.date').text)
            counts.append(driver.find_element_by_css_selector('span.count').text)
            comments_num.append(driver.find_element_by_css_selector('strong.num').text)
            scores.append(driver.find_element_by_css_selector('em.u_cnt._count').text)

            # 제목수집
            time.sleep(1)
            titles.append(driver.find_element_by_css_selector('h3.title_text').text)

            # 유저이름
            users.append(driver.find_element_by_css_selector('div.nick_box').text)

            # 글 본문
            try:
                body.append(driver.find_element_by_css_selector('div.se-module.se-module-text').text)

            except:
                try:
                    body.append(driver.find_element_by_css_selector('div.ContentRenderer').text)
                except:
                    body.append('')

            # 댓글 수집
            soup = bs(driver.page_source, 'lxml')
            iscomment = soup.find_all('span', class_='text_comment')

            if len(iscomment) == 0:
                box = []

            else:
                WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'text_comment')))

                soup = bs(driver.page_source, 'lxml')
                iscomment = soup.find_all('span', class_='text_comment')
                box = []

                for i in iscomment:
                    box.append([i.get_text()])

            comments.append(box)

            # 크롤러 진행상황보기 (50개마다 출력)
            check += 1
            current_link = driver.find_element_by_xpath('//*[@id="app"]/div/div/div[1]/div[2]/a[2]').get_attribute(
                'href')
            if check % 100 == 0:
                print('현재 %d개의 크롤링을 완료하였습니다.' % check)

            elif check == 1:
                print('첫 번째 크롤링이 성공적이었습니다.')

            else:
                pass

            # 다음글클릭
            try:
                driver.find_element_by_css_selector(
                    '#app > div > div > div.ArticleTopBtns > div.right_area > a.BaseButton.btn_next.BaseButton--skinGray.size_default > span').click()
                driver.implicitly_wait(20)

            except:
                current_url = driver.find_element_by_xpath('//*[@id="app"]/div/div/div[2]/div[1]/div[3]/a[2]').get_attribute('href')
                driver.close()

                driver = webdriver.Chrome(os.path.join(path, '../../chrome_driver/chromedriver'), options=options)
                driver.get(login_url)
                driver.execute_script("document.getElementsByName('id')[0].value = \'" + my_id + "\'")
                driver.execute_script("document.getElementsByName('pw')[0].value = \'" + my_pwd + "\'")
                time.sleep(1)

                driver.find_element_by_id('log.login').click()
                time.sleep(1)

                driver.get(current_url)
                time.sleep(5)

        except:
            print(current_link)
            with open(os.path.join(path, 'baking_error.txt'), 'w') as file:
                file.write(current_link)

            
    # 중간저장
    crawled_texts = pd.DataFrame([users, dates, counts, titles, body, comments_num, comments, scores],
                                 index=['닉네임', '날짜', '조회수', '제목', '본문', '댓글개수', '댓글', '좋아요']).T
    crawled_texts.to_json(file_path, orient='table')

    # 크롬이 다시 열릴때 가지고 올 다음 링크를 가져오기
    link = driver.find_element_by_xpath('//*[@id="app"]/div/div/div[1]/div[2]/a[2]').get_attribute('href')
    driver.close()

    driver = webdriver.Chrome(os.path.join(path, '../../chrome_driver/chromedriver'), options=options)
    driver.get(login_url)

    driver.execute_script("document.getElementsByName('id')[0].value = \'" + my_id + "\'")
    driver.execute_script("document.getElementsByName('pw')[0].value = \'" + my_pwd + "\'")
    time.sleep(1)

    driver.find_element_by_id('log.login').click()
    time.sleep(1)

    driver.get(link)
    driver.switch_to.frame('cafe_main')
    time.sleep(5)

driver.quit()

crawled_texts = pd.DataFrame([users, dates, counts, titles, body, comments_num, comments, scores],
                             index=['닉네임', '날짜', '조회수', '제목', '본문', '댓글개수', '댓글', '좋아요']).T
crawled_texts.to_json(file_path, orient='table')
