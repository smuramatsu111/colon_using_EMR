from os import pipe2
import re
import json

unit_dict = json.load(open("./mylib/unit.json", 'r'))

def format_data(text):
    text = text.replace(',', '､')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = re.sub('<br>|\&lt;br\&gt;', '\n', text)
    text = re.sub('<(.*)>', ' ', text)

    text = re.sub('&ensp;|&nbsp;|&emsp;|&thinsp;|\t', ' ', text)

    text = text.replace('&amp;', '')
    text = text.replace('&quot;', '\"')
    text = text.replace('&apos;', '\'')

    text = re.sub('\-\-\-*', '', text)
    text = re.sub('ー*ー', '', text)
    text = re.sub('\.\.*\.\.\.', '', text)
    text = re.sub('\_\_*\_', '', text)
    text = re.sub('==*=', '', text)
    text = re.sub('\*\*\**', '', text)

    time = r'[1-2]?[0-9]:[0-5][0-9]'
    replace = re.compile(r'{time}([\~]{time})?|[1-2]?[0-9]時([0-5][0-9]分)?|[0-5][0-9]分'.format(time=time))
    text = re.sub(replace, f'<TIME>', text)

    date = r'(\([月火水木金土日]\)|[月火水木金土日](曜日)?)?'
    year = r'(20|19)?[0-9]{2}'
    month = r'(1[0-2]|0?[1-9])'
    day = r'([12][0-9]|3[01]|0?[1-9])'

    replace = re.compile(r'(明治|大正|昭和|平成|令和)[1-9]?([2-9]|元)年({month}月)?({day}日)?{date}'.format(month=month,day=day,date=date))
    text = re.sub(replace, f'<DATE>', text)

    replace = re.compile(r'({year}年)?{month}月{day}日{date}|({year}[\-/])?{month}[\-/]{day}|{year}\.{month}\.{day}'.format(year=year,month=month,day=day,date=date))
    text = re.sub(replace, f'<DATE>', text)

    replace= re.compile(r'({year}年)?{month}月|{year}[\-/]{month}|{year}年|{day}日'.format(year=year,month=month,day=day))
    text = re.sub(replace, f'<DATE>', text)

    text = re.sub('(20|19)?[0-9]{2}(1[0-2]|0?[1-9])([12][0-9]|3[01]|0?[1-9])(\([月火水木金土日]\)|[月火水木金土日](曜日)?)?', f'<DATE>', text)
    text = re.sub(r'[月火水木金土日]曜日', '<DATE>', text)

    for unit in unit_dict.keys():
        replace = re.compile(r'(([0-9]+|[0-9]+\.[0-9]+)[\~/\-]?([0-9]+|[0-9]+\.[0-9]+)?)%s' % unit)
        text = re.sub(replace, f'<NUM>{unit}', text)
    for unit in unit_dict.values():
        replace = re.compile(r'(([0-9]+|[0-9]+\.[0-9]+)[\~/\-]?([0-9]+|[0-9]+\.[0-9]+)?)%s' % unit)
        text = re.sub(replace, f'<NUM>{unit}', text)
    text = re.sub('(\r)?\n', '<NL>', text)
    

    return text

