
from bs4 import BeautifulSoup
import bs4
import xml.etree.ElementTree as ET
from loguru import logger

def xml_construct(candidate_dic, strings):
    root = ET.Element('policy')
    url = ET.SubElement(root, 'url')
    

    temp = ''
    i = 0
    current = root
    current_parent = None
    while i < len(strings):
        if i not in candidate_dic.keys():
            temp += (strings[i] + ' ')

        elif candidate_dic[i][-2] == 1:
            if temp != '':
                p = ET.SubElement(current, 'p')
                p.text = temp
                temp = ''

            sec1 = ET.SubElement(root, 'section')
            current = sec1
            title = ET.SubElement(sec1, 'title')
            title.text = candidate_dic[i][3]
            title.set('tagname', candidate_dic[i][4])
            title.set('depth', candidate_dic[i][0])
            title.set('font-size', candidate_dic[i][1])
            title.set('font-weight', candidate_dic[i][2])

        elif candidate_dic[i][-2] == 2:
            if temp != '':
                p = ET.SubElement(current, 'p')
                p.text = temp
                temp = ''

            sec2 = ET.SubElement(sec1, 'section')
            current = sec2
            title = ET.SubElement(sec2, 'title')
            title.text = candidate_dic[i][3]
            title.set('tagname', candidate_dic[i][4])
            title.set('depth', candidate_dic[i][0])
            title.set('font-size', candidate_dic[i][1])
            title.set('font-weight', candidate_dic[i][2])

        elif candidate_dic[i][-2] == 3:
            if temp != '':
                p = ET.SubElement(current, 'p')
                p.text = temp
                temp = ''

            if candidate_dic[i][-1] == 1:
                current_parent = sec1
            elif candidate_dic[i][-1] == 2:
                current_parent = sec2
            elif candidate_dic[i][-1] == 4:  # 注意 此处需要异常处理
                try:
                    current_parent = sec2
                except BaseException:
                    current_parent = sec1

            sec3 = ET.SubElement(current_parent, 'section')
            current = sec3
            title = ET.SubElement(sec3, 'title')
            title.text = candidate_dic[i][3]
            title.set('tagname', candidate_dic[i][4])
            title.set('depth', candidate_dic[i][0])
            title.set('font-size', candidate_dic[i][1])
            title.set('font-weight', candidate_dic[i][2])

        elif candidate_dic[i][-2] == 4:
            if temp != '':
                p = ET.SubElement(current, 'p')
                p.text = temp
                temp = ''

            if candidate_dic[i][-1] == 1:
                current_parent = sec1
            elif candidate_dic[i][-1] == 2:
                current_parent = sec2
            elif candidate_dic[i][-1] == 3:
                current_parent = sec3

            sec4 = ET.SubElement(current_parent, 'section')
            current = sec4
            title = ET.SubElement(sec4, 'title')
            title.text = candidate_dic[i][3]
            title.set('tagname', candidate_dic[i][4])
            title.set('depth', candidate_dic[i][0])
            title.set('font-size', candidate_dic[i][1])
            title.set('font-weight', candidate_dic[i][2])

        i += 1

    # 最后一段
    p = ET.SubElement(current, 'p')
    p.text = temp

    return root


def prettyXml(element, indent, newline, level=0):
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list

    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)
    return


def xml_convertion(titles, body):
    candidate = read_candidate(titles)
    logger.info("重建xml")
    strings = list(body.stripped_strings)
    # strings = get_strings(body)

    # 标记各级标题在strings序列中的位置
    # key:value —— index of sentence(in the strings):
    candidate_dic = {}
    pre = 0  # 前一项的level
    for i in range(len(strings)):
        text = strings[i]
        for item in candidate:
            if text == item[5]:
                candidate_dic[i] = item[2:] + [pre]
                pre = item[-1]
                # [depth, size, weight, text, tagname, level, pre_level]
                break
    #print(candidate_dic)

    root = xml_construct(candidate_dic, strings)  # 生成原始的xml文件,返回根元素结点
    prettyXml(root, '\t', '\n')  # 执行美化方法
    tree = ET.ElementTree(root)

    return tree


# 读取人工修正后的candidate列表
def read_candidate(lines):
    res = []
    for i in range(len(lines)):
        lines[i].strip()
        if lines[i]: # 排除空行
            res.append(eval(lines[i][lines[i].find('['):]))

    return res


# 用于保留 a 标签，有待调整
def get_strings(body):
    strings = []
    des = list(body.descendants)

    for i in range(len(des)):
        if type(des[i]) == bs4.element.NavigableString and des[i-1].name != 'a':
            strings.append(des[i].string.strip())

        elif des[i].name == 'a':
            strings.append(repr(des[i]))

    return strings
