from log_parser_lib import *
from automathon import DFA
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F

class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(lstm_model, self).__init__()
        #input_dim will be sequence length
        self.lstm=nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc=nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x = [batch_size, input_dim,1]
        out, h=self.lstm(x.unsqueeze(2))
        x = self.fc(h[0].squeeze())
        x=F.log_softmax(x.unsqueeze(0),dim=1)
        return x

input_dim=20

if __name__ == '__main__':
    with open('data.pkl','rb') as f:
        data=pkl.load(f)
    (log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
    automata1 = DFA(Q, sigma, delta, initialState, F_)
    model=torch.load('/home/obiwan/tmp/model/input20_acc95.pt')
    model.load_state_dict(torch.load('/home/obiwan/tmp/model/input20_acc95_state_dict.pt'))
    
    log_path='../anomaly_log/'
    log_file_list=os.listdir(log_path)
    newly_added_events_list=[]
    for log_file in log_file_list:
        print(f'\n>>> reading {log_file} file... <<<')
        log=read_file(log_path+'/'+log_file)
        abnormal_continue=0
        model_input=[]
        prev_event_num=0
        for single_log in log:
            single_pattern=log_parser(single_log, log_dict)
            if find_pattern_num(single_pattern,log_patterns) == None:
                find_event, event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
                if not find_event:
                    print('*****************************')
                    print('Find Log does not exist in DB : ')
                    print(single_log['date'].strftime("%b %d %H:%M:%S")+single_log['log'] )
                    #print('*****************************\n')
                    event_list.append([single_pattern])
                    newly_added_events_list.append(len(event_list))
                log_patterns.append((single_pattern, 1))
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
                    print('-----------------------------------------')
                    print(f'transition from q{prev_event_num} to q{event_num} does not exist in FSA!')
                    if prev_event_num!=0:
                        print('Prev: '+ prev_log_message)
                        print('Now: ' + single_log['date'].strftime("%b %d %H:%M:%S")+single_log['log'])
                if not 'q'+str(event_num) in delta:
                    delta['q'+str(event_num)]={}
                if not str(event_num) in delta['q'+str(prev_event_num)]:
                    delta['q'+str(prev_event_num)][str(event_num)]='q'+str(event_num)
            if len(model_input)==input_dim and not event_num in newly_added_events_list:
                input_data=torch.tensor([model_input[:]],dtype=torch.float32)
                #print(input_data.shape)
                with torch.no_grad():
                    prediction=model(input_data)
                prediction=torch.topk(prediction,10,dim=1)[1].tolist()[0]
                if not event_num-1 in prediction:
                    abnormal_continue+=1
                else:
                    abnormal_continue=0
                if abnormal_continue==8:
                    print('----------------------------------')
                    print(f'LSTM model detects abnormal log continuosly : \n'+single_log['date'].strftime("%b %d %H:%M:%S")+single_log['log'])

                model_input.pop(0)

            prev_event_num = event_num
            prev_log_message=single_log['date'].strftime("%b %d %H:%M:%S")+single_log['log']
            if not event_num in newly_added_events_list:
                model_input.append(event_num)