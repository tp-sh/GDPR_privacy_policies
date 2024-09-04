# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import bs4
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

import csv
from lxml import etree
import lxml.html
from shutil import copyfile
import joblib

import os
from .utils import get_selenium_driver
from loguru import logger

def tdepth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d


def tagname_convertion(t):
    if t == 'h1':
        t = 1
    elif t == 'h2':
        t = 2
    elif t == 'h3':
        t = 3
    elif t == 'h4':
        t = 4
    elif t == 'h5':
        t = 5
    elif t == 'h6':
        t = 6
    elif t == 'strong' or t=='b' or t=='span' or t=='em' or t=='u' or t=='i':
        t = 7
    else:
        t = 8
    return t

def leadinglabel(text):
    littleletter = 'abcdefghijklmnopqrstuvwxyz'
    upperletter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    romanletter = ['i','ii','iii','iv','v','vi','vii','viii']
    labelmatrix = [0]*12
    try:
        labellt = text[0:12]
    except:
        labellt = text
    list2 = re.split(r'[,.: ();<>]', labellt)
    indexloc = -1
    listlen =(len(list2) if len(list2)<=4 else 4)
    for i in range(0,listlen):
        indexloc = indexloc+len(list2[i])+1
        if re.match(r'\d',list2[i]) != None:
            t = 3*i
            try:
                labelmatrix[t+1] = int(list2[i])
            except:
                continue
            labelmatrix[t] = 1
            
            if indexloc < len(labellt):
                sep = labellt[indexloc]
                if sep == '.':
                    labelmatrix[t+2] = 1
                elif sep == ':':
                    labelmatrix[t+2] = 2
                elif sep == '(' or sep == ')':
                    labelmatrix[t+2] = 3
                else:
                    labelmatrix[t+2] = 4
        if re.match(r'[a-z]',list2[i]) != None:
            if len(list2[i]) == 1:
                t = 3*i
                labelmatrix[t] = 2
                labelmatrix[t+1] = littleletter.index(list2[i])+1
                
                if indexloc < len(labellt):
                    sep = labellt[indexloc]
                    if sep == '.':
                        labelmatrix[t+2] = 1
                    elif sep == ':':
                        labelmatrix[t+2] = 2
                    elif sep == '(' or sep == ')':
                        labelmatrix[t+2] = 3
                    else:
                        labelmatrix[t+2] = 4
        if re.match(r'[A-Z]',list2[i]) != None:
            if len(list2[i]) == 1:
                t = 3*i
                labelmatrix[t] = 3
                labelmatrix[t+1] = upperletter.index(list2[i])+1
                
                if indexloc < len(labellt):
                    sep = labellt[indexloc]
                    if sep == '.':
                        labelmatrix[t+2] = 1
                    elif sep == ':':
                        labelmatrix[t+2] = 2
                    elif sep == '(' or sep == ')':
                        labelmatrix[t+2] = 3
                    else:
                        labelmatrix[t+2] = 4
        if re.match(r'[iv]',list2[i]) != None:
            t = 3*i
            try:
                labelmatrix[t+1] = romanletter.index(list2[i])+1
            except:
                continue
            labelmatrix[t] = 4
            
            
            if indexloc < len(labellt):
                sep = labellt[indexloc]
                if sep == '.':
                    labelmatrix[t+2] = 1
                elif sep == ':':
                    labelmatrix[t+2] = 2
                elif sep == '(' or sep == ')':
                    labelmatrix[t+2] = 3
                else:
                    labelmatrix[t+2] = 4
    return labelmatrix

def textlen(t):
    try:
        text_list = t.split()
    except:
        text_list = []
    tlen = len(text_list)
    return tlen
  
# 从简化过的body的descendants中选取候选结点1
def get_candidate(body):
    candidate = []
    tags = list(body.find_all(True))

    for i in range(len(tags)):
        if len(list(tags[i].strings)) == 1:
            candidate.append(tags[i])  # 可能包含text相同的结点
    return candidate        
  
#去掉可能为列表的元素
def label_check(c):
    candidate = []
    set1 = {'td', 'th', 'a'}
    set2 = {'li', 'td', 'th', 'a'}
    for item in c:
        flag = True
        for p in item.parents:
            if p and (p.name in set1):
                flag = False
                break
        if flag and item.name not in set2:
            candidate.append(item)
    return candidate

# 提取每个标签的名称、文本内容 [text,tagname,depth]
def convert_tag_to_list(c, d):
    tags_list = []
    for item in c:
        tag_list = []
        fix_string = item.string.strip().strip('"').strip("'")  # 为了适配Xpath的语法
        tag_list.append(fix_string)
        tag_list.append(item.name)
        depth = len(list(item.parents)) - d
        tag_list.append(depth)

        if tag_list not in tags_list:
            tags_list.append(tag_list)
    return tags_list

# 利用xpath定位元素，再利用selenium获取对应font-size,weight
# tags_list : [text, tagname, depth, font-size, font-weight, color ]

def feature_extract(tags_list, url, frame_ids):
    driver = get_selenium_driver()
    service = Service()
    driver = webdriver.Chrome(service=service, options=option)
    # driver.implicitly_wait(6)
    driver.get(url)
    # 切换至对应 frame
    for frame_id in frame_ids:
        if (frame_id == 0): break
        frames = driver.find_elements(By.XPATH, "//iframe")
        driver.switch_to.frame(frames[frame_id-1])

    html = driver.find_element(By.TAG_NAME, 'html')
    html_fs = html.value_of_css_property('font-size')
    html_fw = html.value_of_css_property('font-weight')

    # print(body_fc, html_fw, html_fs)
    fontsize_list=[]
    fontweight_list=[]
    depth_list=[]
    
    feature_list = []
        
    for item in tags_list:
        t_text = item[0]
        tagname = item[1]
        depth = item[2]
        tlen = textlen(t_text)
        logger.info(f"内容: {t_text[:10]} length: {len(t_text)} tag: {tagname} depth:{depth}")
        if tlen != 0:
            labelmatrix = leadinglabel(t_text)
        else:
            labelmatrix = [0]*12
        taglevel = tagname_convertion(tagname)
        # 注意Xpath的语法格式, 单双引号的异常处理
        try:
            part_text = [t[:5] for t in re.split('[【】，。：, .?？！=^&*)(、（）a-zA-Z0-9:/\[\]\n]', t_text) if len(t)]
            if len(part_text):
                e = driver.find_element(By.XPATH, f"//*[name()='{tagname}' and contains(string(),'{part_text[0]}')]")
            else: 
                logger.warning(f"{t_text} subfind failed")
                continue
        except NoSuchElementException:
            logger.warning("no such Element")
            continue
        except BaseException:
            logger.warning("Base exception")
            continue   
        fs = e.value_of_css_property('font-size')
        fw = e.value_of_css_property('font-weight')
        if e.value_of_css_property('text-decoration') == 'underline':
            t_underline = 1
        else:
            t_underline =0
        if e.value_of_css_property('font-style') == 'italic':
            t_italic = 1
        else:
            t_italic = 0
        
        rel_fs = round(float(fs[:-2]) / float(html_fs[:-2]), 2)
        rel_fw = float(fw) / float(html_fw)
        
        fontsize_list.append(float(rel_fs))
        fontweight_list.append(float(rel_fw))
        depth_list.append(int(depth))

        item.append(rel_fs)
        item.append(rel_fw)
        node_feature = labelmatrix+[int(tlen),float(rel_fs),float(rel_fw),int(depth),float(rel_fs),float(rel_fw),int(depth),taglevel,t_italic,t_underline,t_text]
        feature_list.append(node_feature)
    #driver.quit()

    unique_feature_list = []
    for i in range(len(feature_list)):
        if i == 0:
            unique_feature_list.append(feature_list[i])

        # tags_list[i][0] 文本内容
        elif feature_list[i][-1] == feature_list[i-1][-1]:
            # >=保证保留 后者 ， 可以使得显性标签（eg. h2）在最外层
            if feature_list[i][13] * feature_list[i][14] >= feature_list[i-1][13] * feature_list[i-1][14]:
                unique_feature_list.pop()
                unique_feature_list.append(feature_list[i])
            else:
                continue
        else:
            unique_feature_list.append(feature_list[i])

        
    fontsize_list2 = list(set(fontsize_list))
    fontweight_list2 = list(set(fontweight_list))
    depth_list2 = list(set(depth_list))
    
    fontsize_list2.sort(reverse=True)
    fontweight_list2.sort(reverse=True)
    depth_list2.sort(reverse=True)
    for item in feature_list:
        item[16] = fontsize_list2.index(item[16])+1
        item[17] = fontweight_list2.index(item[17])+1
        item[18] = depth_list2.index(item[18])+1

    return unique_feature_list

def style_feature(c, f, d, frame_ids):

    l1 = convert_tag_to_list(c, d)
    l2 = feature_extract(l1, f, frame_ids)
    return l2

def level_classifier(k):
    clf = joblib.load('ETtrees')
    feature_y = []
    for item in k:
        feature_x = item[0:-1]
        y = clf.predict([feature_x])
        item2 = [y[0],item[-1]]
        feature_y.append(item2)
        
    return feature_y

def tag_filter(url, body, frame_ids):

    # 获取主体部分的深度，用于计算每个节点的相对深度
    body_depth = len(list(body.parents))

    candidate1 = get_candidate(body)   # 从所有Tag元素中筛选只包含一个文本节点的元素，含文本重复的结点
    candidate2 = label_check(candidate1)
    candidate3 = style_feature(candidate2, url, body_depth, frame_ids)
    candidate4 = level_classifier(candidate3)
    lines = ['level:'+str(item[-1])+repr(item) + '\n' for item in candidate4 ]



    return lines
# 将candidate6写入中间文件，人工进行核对     
            
            