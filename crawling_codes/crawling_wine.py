import os
import platform

from selenium import webdriver
from crawling_codes.helper_crawling import run

if __name__ == "__main__":
    # ID, PWD
    my_id = 'hobbang1994'
    my_pwd = 'r945106'

    # Input
    if platform.system() == 'Linux':
        # Path
        path = os.getcwd()
        file_path = os.path.join(path, 'data_baking', 'wine_text.json')
        driver_path = os.path.join(path, 'chrome_driver/chromedriver')  # 윈도우는 .exe 붙여줘야함

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    else:
        # Path
        path = '//'
        file_path = os.path.join(path, 'data_baking', 'wine_text.json')
        driver_path = os.path.join(path, 'chrome_driver/chromedriver.exe')

        # Options
        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    # Input
    target_url = 'https://cafe.naver.com/winerack24'
    menu = '//*[@id="menuLink60"]'

    # Other parameters
    iteration = 200
    batch_size = 250

    # Run
    run(my_id, my_pwd,
        file_path=file_path, driver_path=driver_path, options=options,
        target_url=target_url, menu=menu, initial=True,
        iteration=iteration, batch_size=batch_size)
