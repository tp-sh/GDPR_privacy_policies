import requests
import os
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import bs4

from loguru import logger
from .bodyExtraction import bodyExtraction
from .tagFilter import tag_filter
from .xmlConvertion import xml_convertion


from .utils import get_selenium_driver

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

def dfs(driver, frame_ids):
    """递归搜索iframe
    """
    iframe_elements = driver.find_elements(By.XPATH, "//iframe")
    text = driver.find_element(By.XPATH, "//*").get_attribute("innerText")
    frame_id = 0 # 选中的隐私政策id
    frame_ids.append(frame_id)
    max_cnt = text.count("隐私") + text.count("rivacy") # 最多关键词数量
    final_html = driver.execute_script("return document.documentElement.outerHTML;")

    ret_path, ret_html, ret_cnt = frame_ids.copy(), final_html, max_cnt
    for fid, iframe in enumerate(iframe_elements):
        driver.switch_to.frame(iframe)
        frame_ids[-1] = fid+1
        sub_path, sub_html, sub_cnt = dfs(driver, frame_ids)
        if (sub_cnt > max_cnt):
            ret_path, ret_html, ret_cnt = sub_path, sub_html, sub_cnt
        driver.switch_to.parent_frame()
    return ret_path, ret_html, ret_cnt

def get_pp_html(url):
    logger.info(f"GET {url}")
    driver = get_selenium_driver()
    driver.get(url)
    driver.implicitly_wait(5)
    driver.maximize_window()
    if driver.title:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        frame_ids, final_html, _ = dfs(driver, [])
        logger.success(f"success with selenium, get title {driver.title}")
        return final_html, frame_ids
    else:
        logger.warning(f"Failed with selenium")
        return  "", []

def preprocess(html_text, url, title, frame_ids = [0]):
    """传入 html_text和url  (url可以是上传的?)"""
    if (html_text == ""): return
    soup2 = BeautifulSoup(html_text, "lxml")
    # step1 主体提取
    body = bodyExtraction(soup2)
    # step2 标题过滤
    titles = tag_filter(url, body, frame_ids)
    with open(f"{title}.txt", 'w', encoding="utf8") as f:
        f.writelines(titles)
    # step3 xml重建
    xml = xml_convertion(titles, body) # ET xml elmentTree

    # 写xml
    xml.write(title+".xml", encoding='utf-8', xml_declaration=True)

    return xml

# url = "https://dict.eudic.net/home/mobilePrivacy?useragent=ting_en_huawei"
# html_text = get_pp_html(url, False)
# xml = preprocess(html_text, url, "debug")