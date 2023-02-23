from log_parser_lib import *
import os
import string
import re

if __name__ == '__main__':
    translator = str.maketrans(string.punctuation.replace('/',''), ' '*(len(string.punctuation)-1))
    log_path='../log_dpnm_tb'
    date_list=os.listdir(log_path)
    log_data=[]
    for date in date_list:
        if os.path.isfile(log_path+'/'+date):
            continue
        log_ = read_file(log_path+'/'+date+'/all.log')
        log_data.extend(log_)
    print(f'##Read total {len(log_data)} num of logs##')
    log_dict, synant_dict=make_dict(log_data)
    log_patterns=make_log_pattern_dict(log_data, log_dict)
    event_list=classify_pattern_to_events(log_patterns,synant_dict)
    print(f'Save logs events')
    with open(log_path+'/'+'evnet_list.txt', 'w') as f:
        for i, single_event in enumerate(event_list):
            f.write('-------------------------------------------------------------'+'\n')
            f.write(str(i+1)+'\n')
            for single_pattern in single_event:
                f.write(' '.join(single_pattern)+'\n')
    new_log_path='../log_dpnm_tb/after_preprocessing'
    for date in date_list:
        log_list=[]
        if os.path.isfile(log_path+'/'+date):
            continue
        if date=='after_preprocessing':
            continue
        with open(log_path+'/'+date+'/all.log') as f:
            text = f.read()
        for each in text.split('\n'):
            if not each:
                continue
            log_list.append(tuple([each[:each.find(':',15)],each[each.find(':',15):] ]))
        for log_ in log_list[:]:
            if log_[0].endswith('lcmd'):
                log_list.remove(log_)
        log_list=[' '.join([x[0], re.sub(r'[0-9]+','',x[1]).lower().translate(translator)]) for x in log_list]
        log_str='\n'.join(log_list)
        os.makedirs(os.path.dirname(new_log_path+'/'+date+'/'), exist_ok=True)
        with open(new_log_path+'/'+date+'/all.log', 'w') as f:
            f.write(log_str)