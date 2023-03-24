from log_parser_lib import *
from automathon import DFA
import os
import pickle as pkl


if __name__ == '__main__':
    with open('data.pkl','rb') as f:
        data=pkl.load(f)
    (log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F) = data
    automata1 = DFA(Q, sigma, delta, initialState, F)
    
    log_path='../anomaly_log/tmp/'
    log_data_=read_log_files(log_path)
    prev_event_num=0
    newly_added_events_list=[]
    for single_log in log_data_:
        single_pattern=log_parser(single_log, log_dict)
        if find_pattern_num(single_pattern,log_patterns) == None:
            find_event, event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
            if not find_event:
                print('\n******************')
                print('Log does not exist in DB : '+single_log['log'] + ' ;\t so we will put in DB')
                event_list.append([single_pattern])
                newly_added_events_list.append(len(event_list))
            log_patterns.append((single_pattern, 1))
            #event_list.append([single_pattern])
        #if find_pattern_num(single_pattern,log_patterns) == None:
        #    print(single_pattern)
        #print(single_pattern)
        event_num=find_event_num(single_pattern,event_list)
        find_path=False
        if str(event_num) in delta['q'+str(prev_event_num)]:
            find_path=True
        else:
            for next_sate in delta['q'+str(prev_event_num)]:
                if str(event_num) in delta['q'+str(next_sate)]:
                    find_path=True
                    #print('find path within 2 transition!')
                    break
        if not find_path:
            if not event_num in newly_added_events_list and not prev_event_num in newly_added_events_list:
                print('--------------------')
                print(f'transition from q{prev_event_num} to q{event_num} does not exist in FSA!')
                if prev_event_num!=0:
                    print('Prev: '+ ' '.join(event_list[prev_event_num-1][0]) + '    Now: ' 
                    + ' '.join(single_pattern))
                    print('Prev: '+ prev_log_message + '    Now: ' + single_log['log'])
            if not 'q'+str(event_num) in delta:
                delta['q'+str(event_num)]={}
            if not str(event_num) in delta['q'+str(prev_event_num)]:
                delta['q'+str(prev_event_num)][str(event_num)]='q'+str(event_num)

        prev_event_num = event_num
        prev_log_message=single_log['log']