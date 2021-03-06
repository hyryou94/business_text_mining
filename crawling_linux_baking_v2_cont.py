import os
import platform

from selenium import webdriver
from helper_crawling import run

if __name__ == "__main__":
    # ID, PWD
    my_id = ''
    my_pwd = ''

    if platform.system() == 'Linux':
        # Path
        file_path = os.path.join('data_baking', 'baking_v2_cont2.json')
        driver_path = os.path.join('chrome_driver', 'chromedriver')

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    else:
        # Path
        file_path = os.path.join('data_baking', 'baking_v2_cont2.json')
        driver_path = os.path.join('chrome_driver', 'chromedriver.exe')

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    # Input
    target_url = 'https://cafe.naver.com/delonghi?iframe_url=%2FArticleRead.nhn%3Fclubid%3D10526290%26articleid%3D795867%26referrerAllArticles%3Dfalse%26menuid%3D436%26page%3D1%26boardtype%3DL'

    # Other parameters   
    iteration = 200
    batch_size = 250
    
    # Run
    run(my_id, my_pwd,
        file_path=file_path, driver_path=driver_path, options=options,
        target_url=target_url, initial=False,
        iteration=iteration, batch_size=batch_size)
