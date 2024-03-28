from log_parser_lib import *
from automathon import DFA
import os
import pickle as pkl

if __name__ == '__main__':
    log_path='../log_dpnm_tb'
    date_list=os.listdir(log_path)
    log_data=[]
    for date in date_list:
        if os.path.isfile(log_path+'/'+date):
            continue
        log_ = read_file(log_path+'/'+date+'/all.log')
        log_data.extend(log_)
    print(f'##Read total {len(log_data)} num of logs##')
    '''log_path_list=['/mnt/e/obiwan/SNIC Log/1st','/mnt/e/obiwan/SNIC Log/2nd','../1st_example_log']
    for log_path_ in log_path_list:
        log_data.extend(read_log_files(log_path_))'''
    log_dict, synant_dict=make_dict(log_data)
    log_patterns=make_log_pattern_dict(log_data, log_dict)
    event_list=classify_pattern_to_events(log_patterns,synant_dict)

    print(f'total {len(event_list)} number of events are classified')
    event_sequence=[]
    Q = ['q0']
    sigma = [str(i+1) for i in range(len(event_list))]
    delta = {'q0':{}}
    initialState = 'q0'
    F = ['q0']
    for date in date_list:
        prev_event_num=0
        if os.path.isfile(log_path+'/'+date):
            continue
        log = read_file(log_path+'/'+date+'/all.log')
        for single_log in log:
            single_pattern=log_parser(single_log, log_dict)
            event_num=find_event_num(single_pattern,event_list)
            if not 'q'+str(event_num) in delta:
                delta['q'+str(event_num)]={}
            if not str(event_num) in delta['q'+str(prev_event_num)]:
                delta['q'+str(prev_event_num)][str(event_num)]='q'+str(event_num)
            prev_event_num=event_num
        #print(delta)
    '''for log_path_ in log_path_list:
        prev_event_num=0
        log_data_=read_log_files(log_path_)
        for single_log in log_data_:
            single_pattern=log_parser(single_log, log_dict)
            event_num=find_event_num(single_pattern,event_list)
            if event_num==None:
                continue
            if not 'q'+str(event_num) in delta:
                delta['q'+str(event_num)]={}
            if not str(event_num) in delta['q'+str(prev_event_num)]:
                delta['q'+str(prev_event_num)][str(event_num)]='q'+str(event_num)
            prev_event_num=event_num'''
    automata1 = DFA(Q, sigma, delta, initialState, F)
    #automata1.view("DFA_for_all")
    data=((log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F))
    print(f'Save logs events to event_list.txt')
    with open('evnet_list.txt', 'w') as f:
        for i, single_event in enumerate(event_list):
            f.write('-------------------------------------------------------------'+'\n')
            f.write(str(i+1)+'\n')
            for single_pattern in single_event:
                f.write(' '.join(single_pattern)+'\n')
    print(f'Save all data to data.pkl')
    with open('data.pkl','wb') as f:
        pkl.dump(data,f)