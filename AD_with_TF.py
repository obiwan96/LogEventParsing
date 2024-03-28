from log_parser_lib import *
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from numpy import log as ln
from datetime import timedelta

# LSTM model for event number prediction
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

input_dim=10

if __name__ == '__main__':
    if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
    with open('data.pkl','rb') as f:
        data=pkl.load(f)
    (log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
    print(f'event num is {len(event_list)}')

    normal_log_path='../log_dpnm_tb'
    log_data=[]
    norm_num=0
    date_list=os.listdir(normal_log_path)
    for date in date_list:
        if os.path.isfile(normal_log_path+'/'+date):
            continue
        log_=read_file(normal_log_path+'/'+date+'/all.log')
        if log_:
            log_data.append(log_)
            norm_num+=1
    print(f'{norm_num} of normal data readed')
    assert(norm_num==len(log_data))
                
    # add all value in log_patterns dictionary
    num_all_log=sum(single_log[1] for single_log in log_patterns)
    num_all_doc=len(log_data)

    # let's calcurate tf-idf of all patterns
    tf_idf=[0 for _ in range(len(log_patterns))]
    for single_file_data in log_data:
        tmp_df=[False for _ in range(len(log_patterns))]
        for single_log in single_file_data[0]:
            single_pattern=log_parser(single_log, log_dict)
            if single_pattern==[]:continue
            tmp_df[find_pattern_num(single_pattern,log_patterns)-1]=True
        for i in range(len(tmp_df)):
            if tmp_df[i]:
                tf_idf[i]+=1
    tf=[]
    idf=[]
    for i in range(len(tf_idf)):
        tf.append(num_all_log/(1+log_patterns[i][1]))
        idf.append(num_all_doc/(1+tf_idf[i]))
        tf_idf[i]=ln(ln(num_all_log/(1+log_patterns[i][1]))*num_all_doc/(1+tf_idf[i])+1)
    print(f'tf-idf of all patterns are calculated')
    print(f'averagae of tf-idf is {sum(tf_idf)/len(tf_idf)} and std is {sum((tf_idf[i]-sum(tf_idf)/len(tf_idf))**2 for i in range(len(tf_idf)))/len(tf_idf)}')
    print(f'average of tf is {sum(tf)/len(tf)} and std is {sum((tf[i]-sum(tf)/len(tf))**2 for i in range(len(tf)))/len(tf)} and max is {max(tf)} and min is {min(tf)}')
    print(f'average of idf is {sum(idf)/len(idf)} and std is {sum((idf[i]-sum(idf)/len(idf))**2 for i in range(len(idf)))/len(idf)} and max is {max(idf)} and min is {min(idf)}')

    # Loading Models
    event_pred_model=torch.load('../model/eventnum50_input10_acc93.pt')
    event_pred_model.load_state_dict(torch.load('../model/eventnum50_input10_acc93_state_dict.pt'))
    event_pred_model.eval()

    # Calculate occurence_probability and repeat_rate
    occurrence_porb_list=[]
    repeat_rate_list=[]
    abnormal_score_list=[]
    for file_num, single_file_data in enumerate(log_data):
        print(f'processing {file_num+1}th file of {len(log_data)} num of files')
        model_input=[]
        prev_event_num=find_event_num(log_parser(single_file_data[0],log_dict),event_list)
        model_input.append(prev_event_num)
        date_now=single_file_data[0]['date']
        recent_event_nums=[[date_now,prev_event_num]]
        for single_log in single_file_data[1:]:
            single_pattern=log_parser(single_log, log_dict)
            date_now=single_log['date']
            if single_pattern==[]:continue            
            event_num=find_event_num(single_pattern,event_list)
            # Event Prediction (LSTM) calculate
            if len(model_input)==input_dim:
                input_data=torch.tensor([model_input[:]],dtype=torch.float32)
                with torch.no_grad():
                    prediction=event_pred_model(input_data)
                occurence_probability=np.exp(prediction[0][prev_event_num-1].item())
                model_input.pop(0)
            else:
                occurence_probability=1
            # Repeat Rate calculate
            recent_event_nums=[x for x in recent_event_nums if x[0]>date_now-timedelta(minutes=5)]
            reapeat_rate=len([x for x in recent_event_nums if x[1]==event_num])
            abnormal_score=tf_idf[find_pattern_num(single_pattern,log_patterns)-1]*occurence_probability*reapeat_rate

            # Update recent_event_nums and etc
            recent_event_nums.append([date_now, event_num])
            prev_event_num=event_num
            model_input.append(event_num)
    # Calcurate average, std, max, min of occurence_probability with np
    print(f'average of occurence_probability is {np.average(occurrence_porb_list)} and std is {np.std(occurrence_porb_list)} and max is {np.max(occurrence_porb_list)} and min is {np.min(occurrence_porb_list)}')
    print(f'average of repeat_rate is {np.average(repeat_rate_list)} and std is {np.std(repeat_rate_list)} and max is {np.max(repeat_rate_list)} and min is {np.min(repeat_rate_list)}')
    print(f'average of abnormal_score is {np.average(abnormal_score_list)} and std is {np.std(abnormal_score_list)} and max is {np.max(abnormal_score_list)} and min is {np.min(abnormal_score_list)}')
    


    #print(lstm_right, lstm_wrong)
    #print(lstm_right/(lstm_right+lstm_wrong))
