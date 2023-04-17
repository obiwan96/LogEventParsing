from log_parser_lib import *
from automathon import DFA
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
    print(f'event num is {len(event_list)}')

    normal_log_path='../log_dpnm_tb'
    abnormal_log_path='../anomaly_log/'
    overloaded_log_path='/mnt/e/obiwan/SNIC Log/overloaded_log'
    log_data=[]
    norm_num=abnor_num=0
    date_list=os.listdir(normal_log_path)
    for date in date_list:
        if os.path.isfile(normal_log_path+'/'+date):
            continue
        log_=read_file(normal_log_path+'/'+date+'/all.log')
        if log_:
            log_data.append([log_,0])
            norm_num+=1
    file_list = os.listdir(overloaded_log_path)
    for file in file_list:
        log_data.append([read_file(overloaded_log_path+'/'+file+'/all.log'),1])
        abnor_num+=1
    file_list=os.listdir(abnormal_log_path)
    for file in file_list:
        log_data.append([read_file(abnormal_log_path+'/'+file),1])
        abnor_num+=1
    print(f'test data has {norm_num} of normal data and {abnor_num} of abnormal data')
    assert(norm_num+abnor_num==len(log_data))
    model=torch.load('/home/obiwan/tmp/model/input20_acc95.pt')
    model.load_state_dict(torch.load('/home/obiwan/tmp/model/input20_acc95_state_dict.pt'))
    model.eval()
    tp=fp=fn=tn=0
    lstm_right=lstm_wrong=0
    for single_file_data in log_data:
        model_input=[]
        prev_event_num=find_event_num(log_parser(single_file_data[0][0],log_dict),event_list)
        model_input.append(prev_event_num)
        abnormal=False
        abnormal_continue=0
        for single_log in single_file_data[0][1:]:
            single_pattern=log_parser(single_log, log_dict)
            if single_pattern==[]:continue
            if find_pattern_num(single_pattern,log_patterns) == None:
                if not single_file_data[1]:
                    print('pattern no find ')
                    print(single_pattern)
                abnormal=True
                break
            event_num=find_event_num(single_pattern,event_list)
            find_path=False
            if str(event_num) in delta['q'+str(prev_event_num)]:
                find_path=True
            else:
                for next_state in delta['q'+str(prev_event_num)]:
                    if str(event_num) in delta['q'+str(next_state)]:
                        find_path=True
                        break
            if not find_path:
                abnormal=True
                if not single_file_data[1]:
                    print('fsa state no find ')
                    print(single_file_data[0][0])
                #break
            if len(model_input)==input_dim:
                input_data=torch.tensor([model_input[:]],dtype=torch.float32)
                with torch.no_grad():
                    prediction=model(input_data)
                prediction=torch.topk(prediction,10,dim=1)[1].tolist()[0]
                if not event_num-1 in prediction:
                    #print('lstm find abnormal')
                    abnormal_continue+=1
                    if not single_file_data[1]:lstm_wrong+=1
                    #break
                else:
                    abnormal_continue=0
                    if not single_file_data[1]: lstm_right+=1
                if abnormal_continue==7:
                    abnormal=True
                    break
                model_input.pop(0)
            prev_event_num = event_num
            prev_log_message=single_log['log']
            model_input.append(event_num)
        if single_file_data[1]:
            if abnormal:
                tp+=1
            else:
                fn+=1
        elif abnormal:
            fp+=1
        else:
            tn +=1
    
    acc = (tp+tn)/(tp+fp+fn+tn)*100
    prec=tp/(tp+fp)*100
    rec=tp/(tp+fn)*100
    print(f'total score \n accuracy : {acc:.2f}, f1 : {2*(prec*rec)/(prec+rec):.2f}, prec : {prec:.2f}, rec : {rec:.2f}')
    #print(lstm_right, lstm_wrong)
    #print(lstm_right/(lstm_right+lstm_wrong))
