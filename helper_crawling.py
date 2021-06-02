# Settings
import os
import platform
import time

import pandas as pd
from bs4 import BeautifulSoup as bs

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By



# Login
def login(my_id, my_pwd, driver):
    login_url = 'https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com'
    driver.get(login_url)

    driver.execute_script("document.getElementsByName('id')[0].value = \'" + my_id + "\'")
    driver.execute_script("document.getElementsByName('pw')[0].value = \'" + my_pwd + "\'")
    time.sleep(1)

    driver.find_element_by_id('log.login').click()
    time.sleep(1)


# Crawling
def to_target(target_url, driver, menu=None, initial=True):
    driver.get(target_url)
    time.sleep(1)

    if initial:
        driver.find_element_by_xpath(menu).click()
        time.sleep(1)

        driver.switch_to.frame('cafe_main')
        driver.find_element_by_xpath('//*[@id="main-area"]/div[4]/table/tbody/tr[1]/td[1]/div[2]/div/a[1]').click()


def body_crawling(driver):
    try:
        body = driver.find_element_by_css_selector('div.se-module.se-module-text').text

    except:
        try:
            body = driver.find_element_by_css_selector('div.ContentRenderer').text
        except:
            body = ''
    return body


def comment_crawling(driver):
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
    return box


def crawling(driver):
    driver.implicitly_wait(10)
    date = driver.find_element_by_css_selector('span.date').text  # 날짜
    count = driver.find_element_by_css_selector('span.count').text  # 조회수
    comment_count = driver.find_element_by_css_selector('strong.num').text  # 댓글수
    score = driver.find_element_by_css_selector('em.u_cnt._count').text  # 좋아요

    time.sleep(1)
    title = driver.find_element_by_css_selector('h3.title_text').text  # 제목
    user = driver.find_element_by_css_selector('div.nick_box').text  # 유저이름
    body = body_crawling(driver)  # 본문
    comment = comment_crawling(driver)  # 댓글
    return date, count, comment_count, score, title, user, body, comment


def run(my_id, my_pwd, file_path, driver_path, options, target_url, menu=None, initial=True, iteration=200, batch_size=250):
    driver = webdriver.Chrome(driver_path, options=options)
    login(my_id, my_pwd, driver)
    to_target(target_url, driver, menu, initial=initial)

    if not initial:
        driver.switch_to.frame('cafe_main')

    crawled = []
    check = 0
    for k in range(iteration):
        for t in range(batch_size):
            try:
                crawled.append(crawling(driver))

                next_button = driver.find_element_by_xpath('//*[@id="app"]/div/div/div[1]/div[2]/a[1]')
                if next_button.text != '다음글':
                    next_button = driver.find_element_by_xpath('//*[@id="app"]/div/div/div[1]/div[2]/a[2]')

                next_link = next_button.get_attribute('href')

                # 크롤러 진행상황보기 (50개마다 출력)
                check += 1
                if check % 100 == 0:
                    print('현재 %d개의 크롤링을 완료하였습니다.' % check)

                elif check == 1:
                    print('첫 번째 크롤링이 성공적이었습니다.')

                else:
                    pass

                # 다음글클릭
                driver.find_element_by_css_selector(
                    '#app > div > div > div.ArticleTopBtns > div.right_area > a.BaseButton.btn_next.BaseButton--skinGray.size_default > span').click()
                driver.implicitly_wait(20)

            except:
                driver.get(next_link)
                driver.switch_to.frame('cafe_main')
                time.sleep(5)

                with open(os.path.join('error_logs', os.path.split(file_path)[-1].split('.')[0] + 'error_log.txt'), 'w') as file:
                    file.write(next_link)

        # 중간저장
        crawled_texts = pd.DataFrame(crawled,
                                     columns=['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글'])
        crawled_texts.to_json(file_path, orient='table')

        # 크롬이 다시 열릴때 가지고 올 다음 링크를 가져오기
        driver.close()

        driver = webdriver.Chrome(driver_path, options=options)
        login(my_id, my_pwd, driver)

        driver.get(next_link)
        driver.switch_to.frame('cafe_main')
        time.sleep(5)

    driver.quit()

    crawled_texts = pd.DataFrame(crawled,
                                 columns=['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글'])
    crawled_texts.to_json(file_path, orient='table')


class crawling_settings:
    def __init__(self):
        if platform.system() == 'Linux':
            # Path
            self.file_path = os.path.join('data_baking', 'baking_v2_cont.json')
            self.driver_path = os.path.join('chrome_driver', 'chromedriver')

            # Headless
            self.options = webdriver.ChromeOptions()
            self.options.add_argument('headless')
            self.options.add_argument('window-size=1920x1080')
            self.options.add_argument("disable-gpu")

        else:
            # Path
            self.file_path = os.path.join('data_baking', 'baking_v2_cont.json')
            self.driver_path = os.path.join('chrome_driver', 'chromedriver.exe')

            # Headless
            self.options = webdriver.ChromeOptions()
            self.options.add_argument('window-size=1920x1080')
            self.options.add_argument("disable-gpu")
