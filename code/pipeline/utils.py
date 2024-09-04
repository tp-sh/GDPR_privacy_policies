
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

def get_selenium_driver():
    option = webdriver.ChromeOptions()
    option.add_argument('headless')  # 设置option
    option.add_argument('disable-infobars')
    option.add_argument('lang=zh_CN.UTF-8')
    service = Service()
    driver = webdriver.Chrome(service=service, options=option)
    return driver