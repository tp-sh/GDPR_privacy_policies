# -*- coding: utf-8 -*-


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


path="/usr/bin/chromedriver" # chromdriver.exe 放的路径
chrome_options = Options()
chrome_options.add_argument("--headless")  
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(path,chrome_options = chrome_options)
webs = open("./webs.txt")                       #需要爬取pp的网站列表文件
def get_html(webname):
    driver.get(webname)
    time.sleep(2)
    Optionsss = ['Accept', 'ACCEPT ALL', 'Agree', 'Yes','AGREE']             #应对页面的cookie的询问
    for i in Optionsss:
            cookies = driver.find_elements_by_xpath("//*[contains(text(), '{}')]".format(i))
            for cookie in cookies:
                if cookie != None:
                    try:
                        class1 = cookie.get_attribute('class')
                        if class1 != None or class1 != "":
                            accept_cookie =(By.XPATH, "//*[@class='{}']".format(class1))
                            WebDriverWait(driver, 3).until(EC.element_to_be_clickable(accept_cookie)).click()
                    except:
                        pass
                    
                    try:
                        id1 = cookie.get_attribute('id')
                        if id1 != None or id1 != "": 
                            accept_cookie =(By.XPATH, "//*[@id='{}']".format(id1))
                            WebDriverWait(driver, 3).until(EC.element_to_be_clickable(accept_cookie)).click()
                    except:
                        pass
        #################################
    time.sleep(5)

    privacy_filters = ["Privacy","rivacy Policy","rivacy policy","erm privacy"]             #定位隐私政策链接的关键词
    signup_filters = ["reate account", "egister", "ign up","ign-up", "SIGN UP"]
    init_url = driver.current_url
    for i in privacy_filters:                                                   #根据关键词定位各链接并自动点击跳转
        try:
            print("detecting")
            check = driver.find_elements_by_xpath("//*[contains(text(), '{}')]".format(i))
            if check != None:
                for element in check:
                    try:
                        element.click()
                        print("privacy detected")
                    except:
                        pass
        except:
            print("failed to locate")
    if init_url == driver.current_url:
        print("unable to locate privacy url")
        for i in signup_filters:
            try:
                check2 = driver.find_elements_by_xpath("//*[contains(text(), '{}')]".format(i))
                if check2 != None:
                    for element in check2:
                        try:
                            element.click()
                            time.sleep(3)
                            if init_url != driver.current_url:
                                for i in privacy_filters:
                                    try:
                                        check3 = driver.find_elements_by_xpath("//*[contains(text(), '{}')]".format(i))
                                        if check3 != None:
                                            for element in check3:
                                                try:
                                                    element.click()
                                                except:
                                                    pass
                                    except:
                                        pass
                        except:
                            pass
            except:
                pass
        if init_url == driver.current_url:
            print("unable to auto locate, need manual intervene")
        else:
            htmlfile = 'test.html'
            f = open(htmlfile,'wb')
            f.write(driver.page_source.encode('gbk','ignore'))
            print('write succeed')
            f.close()    
    else:
        htmlfile = 'test.html'
        f = open(htmlfile,'wb')
        f.write(driver.page_source.encode('gbk','ignore'))
        print('write succeed')
        f.close() 
        

                                
            
            
    
    
    


