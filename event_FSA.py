from log_parser_lib import *
from automathon import DFA
import os

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
            if event_num==None:
                print(date)
                print(single_log)
                print(single_pattern)
                quit()
            if not 'q'+str(event_num) in delta:
                delta['q'+str(event_num)]={}
            if not str(event_num) in delta['q'+str(prev_event_num)]:
                delta['q'+str(prev_event_num)][str(event_num)]='q'+str(event_num)
            prev_event_num=event_num
        #print(delta)
    automata1 = DFA(Q, sigma, delta, initialState, F)
    automata1.view("DFA_for_all")