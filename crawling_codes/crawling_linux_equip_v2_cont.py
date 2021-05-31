import os
import platform

from selenium import webdriver
from crawling_codes.helper_crawling import run

if __name__ == "__main__":
    # ID, PWD
    my_id = 'hobbang1994'
    my_pwd = 'r945106'

    if platform.system() == 'Linux':
        # Path
        path = '/home/hyryou94/crawling'
        file_path = os.path.join(path, 'data_baking_equip', 'equip_v2_cont.json')
        driver_path = os.path.join(path, 'chrome_driver', 'chromedriver')  # 윈도우는 .exe 붙여줘야함

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    else:
        # Path
        path = '//'
        file_path = os.path.join(path, 'data_baking_equip', 'equip_v2_cont.json')
        driver_path = os.path.join(path, 'chrome_driver', 'chromedriver.exe')

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    # Input
    target_url = 'https://cafe.naver.com/delonghi?iframe_url=%2FArticleRead.nhn%3Fclubid%3D10526290%26articleid%3D632950%26referrerAllArticles%3Dfalse%26menuid%3D433%26page%3D1%26boardtype%3DL'

    # Other parameters
    iteration = 200
    batch_size = 250

    # Run
    run(my_id, my_pwd,
        file_path=file_path, driver_path=driver_path, options=options,
        target_url=target_url, initial=False,
        iteration=iteration, batch_size=batch_size)
