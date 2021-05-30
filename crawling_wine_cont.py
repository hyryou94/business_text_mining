import os
import platform

from selenium import webdriver
from helper_crawling import run

if __name__ == "__main__":
    # ID, PWD
    my_id = 'hyryou94'
    my_pwd = 'r945106'

    if platform.system() == 'Linux':
        # Path
        path = os.getcwd()
        file_path = os.path.join('/home/hyryou94/gdrive/SharedDrives/HandaProjects/wine_crawling', 'wine_text.json')
        driver_path = os.path.join(path, 'chrome_driver/chromedriver')  # 윈도우는 .exe 붙여줘야함

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    else:
        # Path
        path = '//'
        file_path = os.path.join(path, 'data_wine', 'baking_v2_cont2.json')
        driver_path = os.path.join(path, 'chrome_driver/chromedriver.exe')

        # Headless
        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

    # Input
    target_url = 'https://cafe.naver.com/ArticleRead.nhn?clubid=20564405&page=1&menuid=60&boardtype=L&articleid=110208&referrerAllArticles=false'

    # Other parameters   
    iteration = 200
    batch_size = 250
    
    # Run
    run(my_id, my_pwd,
        file_path=file_path, driver_path=driver_path, options=options,
        target_url=target_url, initial=False,
        iteration=iteration, batch_size=batch_size)
