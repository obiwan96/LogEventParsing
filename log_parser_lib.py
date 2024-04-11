import os 
import numpy as np
from datetime import datetime, timedelta
import string
import re
import pandas as pds
import random
from nltk.corpus import wordnet
date_word_list=['jan', 'feb', 'mar', 'apr','may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 
                'mon', 'tue', 'wed', 'thu','fri', 'sat', 'sun']
except_word_list=['fail', 'expected','disabled', 'old', 'new','group','address', 'warnings', 'output']
loc_word_list=['jincheon', 'jangseongbaegam', 'hyehwa','woosan','jangseong','yeongcheon', 'yangsa', 'jangseongbaeg' ]
translator = str.maketrans(string.punctuation.replace('/',''), ' '*(len(string.punctuation)-1))

########################
# 1. Reading Log files #
########################
def pre_process(sentence):
    single_log={}
    #print(sentence)
    try:
        single_log['date']=datetime.strptime(sentence[:15], "%b %d %H:%M:%S")
    except:
        print(sentence[:15])
    # Do not know what's meaning of number between date and application name.
    # ex) [3], [4] 
    single_log['application']=sentence[sentence.find('%')+1:sentence.find(':',15)]
    single_log['log']=sentence[sentence.find(':',15)+2:]
    return single_log
def read_file(file):
    try:
        with open(file) as f:
            text=f.read()
    except:
        try:
            with open(file, encoding="UTF-8") as f:
                text=f.read()
        except:
            try:
                with open(file, encoding="ISO-8859-1") as f:
                    text=f.read()
            except:
                #print('read ' + file + ' failed')
                return []
    log_data=[]
    for sentence in text.split('\n'):
        if len(sentence)==0 or sentence in ['\n','\r\n']:
            continue
        #print(sentence)
        single_log={}
        try:
            single_log['date']=datetime.strptime(sentence[:15], "%b %d %H:%M:%S")
        except:
            # Previous log is continueing
            log_data[-1]['log']+=sentence
            continue
        # The number between date and application name is log level. 
        # We remove it in now.
        # ex) [3], [4] 
        single_log['application']=sentence[sentence.find('%')+1:sentence.find(':',15)]
        single_log['log']=sentence[sentence.find(':',15)+2:]
        if single_log['log'].startswith('adt') or  single_log['log'].startswith('FAN'):
            continue
        log_data.append(single_log)
    return log_data

def read_log_files(file_path='.'):
    file_list = os.listdir(file_path)
    excel_list=[]
    for file in file_list[:]:
        if file.endswith('log'):
            continue
        if file.startswith('SNIC'):
            file_list.remove(file)
        if not file.endswith('txt') and not file.startswith('messages'):
            file_list.remove(file)
            if file.endswith('xlsx'):
                excel_list.append(file)
        if file.startswith('ubi'):
            file_list.remove(file)
    log_data = []
    for file in file_list:
        tmp_log_data=read_file(file_path+'/'+file)
        log_data.extend(tmp_log_data)
    for file in excel_list:
        #print(file)
        df = pds.read_excel(file_path+'/'+file,header=None, engine='openpyxl')
        for column in df.columns:
            li=df[column].dropna().values.tolist()
            for sentence in li:
                log_data.append(pre_process(sentence))
    print(f'read total {len(log_data)} lines of logs')
    return log_data

###########################
# Making vocab dictionary #
###########################
def make_dict(log_data):
    log_dict={}
    for single_log_data in log_data:
        sentence_words=[words for words in  re.sub(r'[0-9]+','',single_log_data['log']).lower().translate(translator).split()]
        for word in sentence_words:
            if word in log_dict:
                log_dict[word]+=1
            else:
                log_dict[word]=1
    for word in list(log_dict.keys())[:]:
        if log_dict[word]<3 and not word in except_word_list:
            del log_dict[word]
        elif word.startswith('/'):
            del log_dict[word]

    for word in date_word_list:
        if word in log_dict:
            log_dict.pop(word)
    for word in loc_word_list:
        if word in log_dict:
            log_dict.pop(word)

    sorted_log_dict=sorted(log_dict.items(), key=lambda item: item[1], reverse= True)
    print('\n########################################')
    print('Here are most frequent vocabularies.')
    print(sorted_log_dict[:10])
    synant_dict={x[0] : [set([]),set([])] for x in sorted_log_dict} #first list is for synonym, second list is for antonym
    for i in range(len(sorted_log_dict)):
        for syn in wordnet.synsets(sorted_log_dict[i][0]):
            for l in syn.lemmas():
                if l.name() in synant_dict and l.name()!=sorted_log_dict[i][0]:
                    synant_dict[sorted_log_dict[i][0]][0].add(l.name())
                if l.antonyms():
                    if l.antonyms()[0].name() in synant_dict:
                        synant_dict[sorted_log_dict[i][0]][1].add(l.antonyms()[0].name())
    return log_dict, synant_dict

##########################
# 2. Log pattern parsing #
##########################
def log_parser(single_log_data, log_dict):
    unk='[UNK]'
    date='[date]'
    file_path='[path]'
    loc='[loc]'
    interface='[interface]'
    sentence_words=re.sub(r'[0-9]+','',single_log_data['log']).lower().translate(translator).split()
    single_pattern=[]
    if len(sentence_words)==0:
        return []
    if sentence_words[0]=='interface':
        single_pattern=['interface', interface]
        sentence_words=re.sub(r'[0-9]+','',single_log_data['log'][single_log_data['log'].find(' ',10):]
                             ).lower().translate(translator).split()
    for word in sentence_words:
        if word=='x' or word.startswith('ff'):
            single_pattern.append('[number]')
        elif not len(single_pattern)==0 and single_pattern[-1]=='interface':
            single_pattern.append(interface)
        elif word in log_dict:
            single_pattern.append(word)
        elif word in date_word_list:
            single_pattern.append(date)
        elif word in loc_word_list:
            single_pattern.append(loc)
        elif word.startswith('/'):
            single_pattern.append(file_path)
        else:
            if len(single_pattern)==0:
                single_pattern.append(unk)
            elif not single_pattern[-1] == unk:
                single_pattern.append(unk)
    single_pattern=tuple(single_pattern)
    return single_pattern

def print_pattern_with_freq(single_pattern_element):
    print(' '.join(single_pattern_element[0])+ f' : {single_pattern_element[1]}')
def print_pattern(single_pattern):
    print(' '.join(single_pattern))

def make_log_pattern_dict(log_data, log_dict):
    log_patterns={}

    #translator = str.maketrans(string.punctuation , ' '*(len(string.punctuation)))
    for single_log_data in log_data:
        #sentence_words=[words for words in  re.sub(r'[0-9]+','',single_log_data['log']).lower().translate(translator).split()]
        single_pattern=log_parser(single_log_data, log_dict)
        if single_pattern ==[]:
            continue
        #if sentence_words[0]=='area':
        #    print(sentence_words)
        if not single_pattern in log_patterns:
            log_patterns[single_pattern]=1
        else:
            log_patterns[single_pattern]+=1
    log_patterns_sorted=sorted(log_patterns.items(), key=lambda x: x[1], reverse=True)
    print('\n########################################')
    print('Here are most frequent log patterns')
    for log_pattern in log_patterns_sorted[:10]:
        print_pattern_with_freq(log_pattern)
    print(f'Find {len(log_patterns)} number of log patterns')

    ## The result is list!!!! not dict##
    return list(log_patterns.items())

def find_pattern_num(single_pattern,log_patterns):
    for event_num, log_pattern_element in enumerate(log_patterns):
        if log_pattern_element[0]==single_pattern:
            return event_num+1

#####################
# 3. Event classify #
#####################

# Minimum Edit Distance Algorithm
# refer to https://joyjangs.tistory.com/38
def edit_dist(pattern1, pattern2):
    dp=[[0]*(len(pattern2)+1) for _ in range(len(pattern1)+1)]
    for i in range(1, len(pattern1)+1):
        dp[i][0]=i
    for i in range(1, len(pattern2)+1):
        dp[0][i]=i
    for i in range(1,len(pattern1)+1):
        for j in range(1,len(pattern2)+1):
            if pattern1[i-1]==pattern2[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
    return dp[-1][-1]

#Edit distance with synonym and antonym
def edit_dist_synatn(pattern1, pattern2,synant_dict):
    dp=[[0]*(len(pattern2)+1) for _ in range(len(pattern1)+1)]
    for i in range(1, len(pattern1)+1):
        dp[i][0]=i
    for i in range(1, len(pattern2)+1):
        dp[0][i]=i
    for i in range(1,len(pattern1)+1):
        for j in range(1,len(pattern2)+1):
            if pattern1[i-1]==pattern2[j-1] or pattern1[i-1] in synant_dict \
                and pattern2[j-1] in synant_dict[pattern1[i-1]][0]:
                dp[i][j]=dp[i-1][j-1]
            elif pattern1[i-1] in synant_dict and pattern2[j-1] in synant_dict[pattern1[i-1]][1]:
                dp[i][j]=dp[i-1][j-1]+6
                
            else:
                dp[i][j]=min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
    return dp[-1][-1]

def classify_pattern_to_events(log_patterns, use_synant=None,print_resuts=False):
    event_list=[]
    for single_pattern in log_patterns[:]:
        find_event, event_list = put_new_pattern_to_event_list(single_pattern[0], event_list, use_synant)
        if not find_event:
            event_list.append([single_pattern[0]])
    for solo_event in event_list[:]:
        if len(solo_event)==1:
            find_event, event_list = put_new_pattern_to_event_list(solo_event[0], event_list, use_synant)
            if find_event:
                event_list.remove(solo_event)
    print('\n########################################')
    print(f'Total num of event is {len(event_list)}')
    print(f'Total num of log pattern is {sum(len(single_event) for single_event in event_list)}')
    if print_resuts:
        for single_event in event_list:
            print('-------------------------------------------------------------')
            for single_pattern in single_event:
                print_pattern(single_pattern)
    return event_list

def put_new_pattern_to_event_list(single_pattern, event_list, use_synant=None):
    edit_algo=lambda x,y:edit_dist_synatn(x,y,use_synant) if use_synant else edit_dist(x,y)
    find_event=False
    for i, single_event in enumerate(event_list):
        ed_sum=0
        ed_min=100
        if single_pattern in single_event:
            continue
        for tmp_pattern in single_event:
            ed_sum+=edit_algo(tmp_pattern,single_pattern)
            ed_min=min(edit_algo(tmp_pattern,single_pattern), ed_min)
            #ed_sum+=edit_dist_synatn(tmp_pattern,single_pattern,use_synant)
            #ed_min=min(edit_dist_synatn(tmp_pattern,single_pattern,use_synant), ed_min)
        if ed_sum/len(single_event)/len(single_pattern)<=0.5:
        #if ed_min/len(single_pattern[0])<0.5:
            event_list[i].append(single_pattern)
            find_event=True
            break
    return find_event, event_list

#######################
# 4. Log event parser #
#######################
def find_event_num(single_pattern,event_list):
    for event_num, single_event in enumerate(event_list):
        for pattern_elem in single_event:
            if pattern_elem==single_pattern:
                return event_num+1

if __name__ == "__main__":
    log_data=read_log_files()
    log_dict=make_dict(log_data)
    log_patterns=make_log_pattern_dict(log_data, log_dict)
    event_list=classify_pattern_to_events(log_patterns)
    interface_up_down_patterns=[_[0] for _ in log_patterns[:2]]
    print('\n#####################################################')
    print("Let's select logs randomly, and putting in log parser")
    print('#####################################################\n')
    for _ in range(5):
        log=random.choice(log_data)
        pattern=log_parser(log,log_dict)
        #if pattern in interface_up_down_patterns:
        #    continue
        print(f'log : {log["log"]}')
        print('pattern : ', end='')
        print_pattern(pattern)
        print(f'event : event {find_event_num(pattern,event_list)}')
        print('-----------------------------------------------------------')