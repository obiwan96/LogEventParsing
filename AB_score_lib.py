from log_parser_lib import *
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from numpy import log as ln
from math import log, sqrt
import matplotlib.pyplot as plt
import pandas as pd

'''
with open('data.pkl','rb') as f:
    data=pkl.load(f)
(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
with open('tf_data.pkl', 'rb')as f:
    tf_idf=pkl.load(f)
'''

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=5):
        super(TransformerModel, self).__init__()
        self.output_dim=output_dim
        self.transformer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward =hidden_dim,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        #self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x = [batch_size, input_dim]
        #print(self.output_dim)
        #print(x)
        x=F.one_hot(x.to(torch.int64), num_classes=self.output_dim)
        # x = [batch_size, input_dim, output_dim]
        out = self.transformer_encoder(x.to(torch.float32))
        # out = [batch_size, input_dim, output_dim]
        x = F.log_softmax(out[:,-1:,:].squeeze(1).to(torch.float32), dim=1)  # 확률 분포를 위해 softmax 적용
        return x

def calculate_tf_idf(log_data, log_patterns):
    # add all value in log_patterns dictionary
    num_all_log=sum(single_log[1] for single_log in log_patterns)
    num_all_doc=len(log_data)+2

    # let's calcurate tf-idf of all patterns
    tf_idf=[0 for _ in range(len(log_patterns))]
    for single_file_data in log_data:
        tmp_df=[False for _ in range(len(log_patterns))]
        for single_log in single_file_data:
            single_pattern=log_parser(single_log, log_dict)
            if single_pattern==[]:continue
            tmp_df[find_pattern_num(single_pattern,log_patterns)-1]=True
        for i in range(len(tmp_df)):
            if tmp_df[i]:
                tf_idf[i]+=1
    df=tf_idf[:]
    tf=[]
    idf=[]
    for i in range(len(tf_idf)):
        tf.append(ln(num_all_log/(1+log_patterns[i][1])))
        idf.append(num_all_doc/(1+tf_idf[i]))
        tf_idf[i]=sqrt(tf[i]*idf[i])
    print(f'tf-idf of all patterns are calculated')
    print(f'averagae of tf-idf is {sum(tf_idf)/len(tf_idf)} and std is {sum((tf_idf[i]-sum(tf_idf)/len(tf_idf))**2 for i in range(len(tf_idf)))/len(tf_idf)}, max is {max(tf_idf)}, min is {min(tf_idf)}')
    print(f'average of tf is {sum(tf)/len(tf)} and std is {sum((tf[i]-sum(tf)/len(tf))**2 for i in range(len(tf)))/len(tf)} and max is {max(tf)} and min is {min(tf)}')
    print(f'average of idf is {sum(idf)/len(idf)} and std is {sum((idf[i]-sum(idf)/len(idf))**2 for i in range(len(idf)))/len(idf)} and max is {max(idf)} and min is {min(idf)}')
    return tf_idf, num_all_doc

def calculate_abnormal_score_for_files(log_data,log_patterns, event_list, tf_idf, num_all_doc, synant_dict=None):
    event_pred_model=torch.load('../model/eventnum50_input10_acc99_transform.pt')
    event_pred_model.load_state_dict(torch.load('../model/eventnum50_input10_acc99_transform_state_dict.pt'))
    event_pred_model.eval()
    num_all_log=sum(single_log[1] for single_log in log_patterns)
    ori_all_event_num=len(event_list)
    print(ori_all_event_num)
    #num_all_doc=len(log_data)+2
    occurrence_porb_list=[]
    repeat_rate_list=[]
    abnormal_score_list=[]
    input_dim=10
    for file_num, single_file_data in enumerate(log_data):
        print(f'processing {file_num+1}th file of {len(log_data)} num of files')
        model_input=[]
        #event_num=find_event_num(log_parser(single_file_data[0],log_dict),event_list)
        #model_input.append(event_num)
        #date_now=single_file_data[0]['date']
        #recent_event_nums=[[date_now,event_num]]
        recent_event_nums=[]
        for single_log in single_file_data:
            single_pattern=log_parser(single_log, log_dict)
            date_now=single_log['date']
            if single_pattern==[]:continue
            event_num=find_event_num(single_pattern,event_list)
            if event_num == None:
                # Unseen Log Pattern!
                log_patterns.append([single_pattern, 1])
                tf_idf.append(ln(ln(num_all_log)*(num_all_doc+2)))
                assert(len(log_patterns)==len(tf_idf))
                find_event,event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
                if find_event:
                    event_num=find_event_num(single_pattern,event_list)
                else:
                    # Unseen Event! Make new event
                    event_list.append([single_pattern])
                    event_num=len(event_list)
            # Event Prediction (LSTM) calculate
            if event_num>ori_all_event_num:
                occurence_probability=1.2e-6
            else:
                if len(model_input)>=input_dim:
                    input_data=torch.tensor([model_input[len(model_input)-input_dim:]],dtype=torch.float32)
                    with torch.no_grad():
                        prediction=event_pred_model(input_data)
                    occurence_probability=np.exp(prediction[0][event_num-1].item())
                    model_input=model_input[len(model_input)-input_dim+1:]
                else:
                    occurence_probability=1
            # Repeat Rate calculate
            recent_event_nums=[x for x in recent_event_nums if x[0]>date_now-timedelta(minutes=10)]
            reapeat_rate=ln((len([x for x in recent_event_nums if x[1]==event_num])+2))
            
            abnormal_score=tf_idf[find_pattern_num(single_pattern,log_patterns)-1]*(1-occurence_probability)*reapeat_rate
            occurrence_porb_list.append(occurence_probability)
            repeat_rate_list.append(reapeat_rate)
            abnormal_score_list.append(abnormal_score)
            # Update recent_event_nums and etc
            recent_event_nums.append([date_now, event_num])
            if event_num<=ori_all_event_num:
                model_input.append(event_num-1)
    print(f'average of occurence_probability is {np.average(occurrence_porb_list)} and std is {np.std(occurrence_porb_list)} and max is {np.max(occurrence_porb_list)} and min is {np.min(occurrence_porb_list)}')
    print(f'average of repeat_rate is {np.average(repeat_rate_list)} and std is {np.std(repeat_rate_list)} and max is {np.max(repeat_rate_list)} and min is {np.min(repeat_rate_list)}')
    print(f'average of abnormal_score is {np.average(abnormal_score_list)} and std is {np.std(abnormal_score_list)} and max is {np.max(abnormal_score_list)} and min is {np.min(abnormal_score_list)}')
    return occurrence_porb_list, repeat_rate_list, abnormal_score_list

def calculate_abnormal_score_for_df(log_data, log_dict,log_patterns, event_list, tf_idf, num_all_doc, synant_dict=None):
    # Now, log_data is DataFrame that has all log data
    # and it's row is datetime.
    event_pred_model=torch.load('../model/eventnum50_input10_acc99_transform.pt')
    event_pred_model.load_state_dict(torch.load('../model/eventnum50_input10_acc99_transform_state_dict.pt'))
    event_pred_model.eval()
    num_all_log=len(log_data)
    ori_all_event_num=len(event_list)
    print(ori_all_event_num)
    occurrence_porb_list=[]
    repeat_rate_list=[]
    abnormal_score_list=[]
    input_dim=10
    model_input=[]
    recent_event_nums=[]
    ab_df=pd.DataFrame(columns=['tf_idf', 'occurence_probability','repeat_rate'])
    for index, row in log_data.iterrows():
        single_pattern=log_parser(row, log_dict)
        date_now=index
        if single_pattern==[]:continue
        event_num=find_event_num(single_pattern,event_list)
        if event_num == None:
            # Unseen Log Pattern!
            log_patterns.append([single_pattern, 1])
            tf_idf.append(ln(ln(num_all_log)*(num_all_doc+2)))
            assert(len(log_patterns)==len(tf_idf))
            find_event,event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
            if find_event:
                event_num=find_event_num(single_pattern,event_list)
            else:
                # Unseen Event! Make new event
                event_list.append([single_pattern])
                event_num=len(event_list)
        # Event Prediction (LSTM) calculate
        if event_num>ori_all_event_num:
            occurence_probability=1.2e-6
        else:
            if len(model_input)>=input_dim:
                input_data=torch.tensor([model_input[len(model_input)-input_dim:]],dtype=torch.float32)
                with torch.no_grad():
                    prediction=event_pred_model(input_data)
                occurence_probability=np.exp(prediction[0][event_num-1].item())
                model_input=model_input[len(model_input)-input_dim+1:]
            else:
                occurence_probability=1
        # Repeat Rate calculate
        recent_event_nums=[x for x in recent_event_nums if x[0]>date_now-timedelta(minutes=10)]
        reapeat_rate=ln((len([x for x in recent_event_nums if x[1]==event_num])+2))
        
        new_row={}
        new_row['tf_idf']=tf_idf[find_pattern_num(single_pattern,log_patterns)-1]
        new_row['occurence_probability']=occurence_probability
        new_row['repeat_rate']=reapeat_rate
        if not any(isinstance(value, str) for value in new_row.values()):
            ab_df.loc[index] = new_row
        ab_df=ab_df[ab_df['tf_idf'].apply(lambda x: not isinstance(x,str))]
    return ab_df

def anomaly_detection_for_file(single_log_data,log_patterns, event_list, tf_idf, synant_dict=None, threshold=[15,2], way='repeat'):
    assert(len(log_patterns)==len(tf_idf) and len(tf_idf) ==sum([len(x) for x in event_list]))
    num_all_log=sum(single_log[1] for single_log in log_patterns)
    ori_all_event_num=len(event_list)
    #print(ori_all_event_num)
    num_all_doc=len(log_data)+2
    recent_abnormal_score=[]
    input_dim=10
    model_input=[]
    recent_event_nums=[]
    abnormal_num=0
    last_abnormal=None
    for single_log in single_log_data:
        single_pattern=log_parser(single_log, log_dict)
        date_now=single_log['date']
        if single_pattern==[]:continue            
        event_num=find_event_num(single_pattern,event_list)
        if event_num == None:
            # Unseen Log Pattern!
            log_patterns.append([single_pattern, 1])
            tf_idf.append(ln(ln(num_all_log)*(num_all_doc+2)))
            #print(len(log_patterns), len(tf_idf), sum([len(x) for x in event_list]))
            assert(len(log_patterns)==len(tf_idf) )
            #print(single_pattern)
            find_event,event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
            if find_event:
                event_num=find_event_num(single_pattern,event_list)
            else:
                # Unseen Event! Make new event
                event_list.append([single_pattern])
                event_num=len(event_list)
            assert(len(tf_idf) ==sum([len(x) for x in event_list]))
        # Event Prediction (LSTM) calculate
        if event_num>ori_all_event_num:
            occurence_probability=1.2e-6
        else:
            if len(model_input)>=input_dim:
                input_data=torch.tensor([model_input[len(model_input)-input_dim:]],dtype=torch.float32)
                with torch.no_grad():
                    prediction=event_pred_model(input_data)
                occurence_probability=np.exp(prediction[0][event_num-1].item())
                model_input=model_input[len(model_input)-input_dim+1:]
            else:
                occurence_probability=1
        # Repeat Rate calculate
        recent_event_nums=[x for x in recent_event_nums if x[0]>date_now-timedelta(minutes=10)]
        reapeat_rate=ln((len([x for x in recent_event_nums if x[1]==event_num])+2))
        #print(len(tf_idf), find_pattern_num(single_pattern,log_patterns)-1)
        abnormal_score=tf_idf[find_pattern_num(single_pattern,log_patterns)-1]*(1-occurence_probability)*reapeat_rate
        if abnormal_score > threshold[0]:
            if way=='repeat':
                abnormal_num+=1
                if abnormal_num == threshold[1]:
                    #print('abnormal score consistently over!')
                    return True
            if way=='time':
                if last_abnormal:
                    if date_now-last_abnormal < timedelta(minutes=threshold[1]):
                        return True
                last_abnormal=date_now
        else:
            abnormal_num=0
        #recent_abnormal_score.append(abnormal_score)
        # Update recent_event_nums and etc
        recent_event_nums.append([date_now, event_num])
        if event_num<=ori_all_event_num:
            model_input.append(event_num-1)
    return False