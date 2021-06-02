import os

from selenium import webdriver
from helper_crawling import run, crawling_settings

if __name__ == "__main__":
    # ID, PWD
    my_id = 'hobbang1994'
    my_pwd = 'r945106'

    # Input
    target_url = 'https://cafe.naver.com/delonghi'
    menu = '//*[@id="menuLink436"]'

    # Other parameters
    iteration = 200
    batch_size = 250
    path_options = crawling_settings()

    # Run
    run(my_id, my_pwd,
        file_path=path_options.file_path, driver_path=path_options.driver_path, options=path_options.options,
        target_url=target_url, menu=menu, initial=True,
        iteration=iteration, batch_size=batch_size)
