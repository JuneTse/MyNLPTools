#coding:utf-8
import re

def get_substring_all_index(text,pattern):
    '''
    返回子串的所有索引位置
    '''
    finder=re.finditer(pattern,text)
    spans=[]
    for f in finder:
        span=f.span()
        spans.append(span)
    return spans

if __name__=="__main__":
    spans=get_substring_all_index(text="hello word hello computer", pattern="hello")
    print(spans)