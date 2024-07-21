from log_parser_lib import *
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from numpy import log as ln
from math import log, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

'''
with open('data.pkl','rb') as f:
    data=pkl.load(f)
(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
with open('tf_data.pkl', 'rb')as f:
    tf_idf=pkl.load(f)
'''

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
        x=F.log_softmax(x,dim=1)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TransformerModel, self).__init__()
        if output_dim%5==0:
            num_heads=5
        elif output_dim%3==0:
            num_heads=3
        elif output_dim%2==0:
            num_heads=2
        else:
            num_heads=1
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

def event_prediction_model_training(log_data, log_dict, event_list, model_type='transformer', 
        input_dim=10, hidden_dim_rate=2, learning_rate=0.01, n_epochs=500, topk_num=5, reducing_rate=1,
        write_summary='/home/dpnm/tmp/tensorboard/', save_model=True):
    ###################################################
    # log_data should be the list of one day of logs  #
    # [[first_day_logs], [second_day_logs], ...]      #
    ###################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim=input_dim*hidden_dim_rate
    event_num = len(event_list)
    assert(model_type in ['transformer', 'lstm'])
    if model_type=='transformer':
        model=TransformerModel(input_dim, hidden_dim, event_num)
    else:
        model=lstm_model(input_dim, hidden_dim, event_num)
    model=model.to(device)
    # Make input and output data
    input_data=[]
    output_data=[]
    for single_data in log_data:
        event_flow=[]
        for single_log in single_data:
            single_pattern=log_parser(single_log, log_dict)
            log_event_num=find_event_num(single_pattern,event_list)
            if log_event_num is None:
                print(f'event num not found! {single_log}')
                continue
            event_flow.append(log_event_num-1) # already -1!!
        if len(event_flow)>input_dim:
            for i in range(len(event_flow)-input_dim):
                input_data.append(event_flow[i:i+input_dim])
                output_data.append(event_flow[i+input_dim])
    
    input_data=torch.tensor(input_data,dtype=torch.float32)
    #input_data=input_data.type(torch.LongTensor)
    output_data=torch.tensor(output_data)
    #output_data=F.one_hot(output_data,num_classes=event_num)
    print(input_data.shape, output_data.shape)
    data_size=input_data.size(dim=0)
    indices=torch.randperm(data_size)
    input_data=input_data[indices]
    output_data=output_data[indices]
    if reducing_rate:
        data_size=int(data_size/reducing_rate)
        input_data=input_data[:data_size]
        output_data=output_data[:data_size]
    x_train=input_data[:data_size-int(data_size/5)]
    y_train=output_data[:data_size-int(data_size/5)]
    x_test=input_data[data_size-int(data_size/5):]
    y_test=output_data[data_size-int(data_size/5):]
    print(f'train data has {len(x_train)} data and test data has {len(x_test)} data')
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=20000, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
    # Let's train
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if write_summary:
        writer = SummaryWriter(write_summary)
    start=time.time()
    for epoch in range(1,n_epochs+1):
        model.train()
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            x_train=x_train.to(device)
            y_train=y_train.to(device)
            #print(x_train.shape, y_train.shape)
            # result: [batch_size, input_dim], [batch_size]
            prediction=model(x_train)
            #print(prediction.shape)
            #print(y_train.shape)
            #prediction=torch.argmax(prediction, dim=1)
            #print(prediction.shape)
            loss=loss_function(prediction, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%50==0:
            print('Epoch {:4d}/{} Batch {}/{} loss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx+1, len(dataloader),loss.item() ))
            if write_summary:
                writer.add_scalar("Training/Loss", loss.item(),epoch+1)
            print('running test..')
            model.eval()
            all_preds=[]
            all_labels=[]
            for batch in test_dataloader:
                x,y=batch
                x=x.to(device)
                y=y.to(device)
                with torch.no_grad():
                    prediction=model(x)
                #all_preds +=torch.argmax(prediction,dim=1)
                all_preds += torch.topk(prediction.cpu(),topk_num,dim=1)[1]
                all_labels += y.cpu()
            all_preds = torch.stack(all_preds).numpy()
            all_labels = torch.stack(all_labels).numpy()
            assert len(all_preds)==len(all_labels)
            #print(all_preds)
            #print(all_labels)
            #test_accuracy = np.sum(all_preds == all_labels) / len(all_preds)
            right_num=0
            for i, label in enumerate(all_labels):
                if label in all_preds[i]:
                    right_num+=1.0
            test_accuracy=right_num/len(all_preds)
            print("Test Accuracy: {0:.3f}".format(test_accuracy))
            if write_summary:
                writer.add_scalar("Test/Accuracy", test_accuracy,epoch+1)
    if write_summary:
        writer.close()
    print(f"Learning takes {(time.time()-start)/60:.2f} minutes")
    if save_model:
        print(f"Model saved in ../model/eventnum{event_num}_input{input_dim}_acc{test_accuracy*100:.0f}_{model_type}.pt")
        torch.save(model, f'../model/eventnum{event_num}_input{input_dim}_acc{test_accuracy*100:.0f}_{model_type}.pt')
        torch.save(model.state_dict(), f'../model/eventnum{event_num}_input{input_dim}_acc{test_accuracy*100:.0f}_{model_type}_state_dict.pt')
    return model

def calculate_tf_idf(log_data, log_dict, log_patterns):
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
        tf_idf[i]=ln(tf[i]*idf[i])
    print(f'tf-idf of all patterns are calculated')
    print(f'averagae of tf-idf is {sum(tf_idf)/len(tf_idf)} and std is {sum((tf_idf[i]-sum(tf_idf)/len(tf_idf))**2 for i in range(len(tf_idf)))/len(tf_idf)}, max is {max(tf_idf)}, min is {min(tf_idf)}')
    print(f'average of tf is {sum(tf)/len(tf)} and std is {sum((tf[i]-sum(tf)/len(tf))**2 for i in range(len(tf)))/len(tf)} and max is {max(tf)} and min is {min(tf)}')
    print(f'average of idf is {sum(idf)/len(idf)} and std is {sum((idf[i]-sum(idf)/len(idf))**2 for i in range(len(idf)))/len(idf)} and max is {max(idf)} and min is {min(idf)}')
    return tf_idf, num_all_doc, num_all_log

def calculate_abnormal_score_for_files(log_data,log_dict, log_patterns, event_list, 
                                        tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict=None, repeat_threshold=10):
    #event_pred_model=torch.load('../model/eventnum50_input10_acc99_transform.pt')
    #event_pred_model.load_state_dict(torch.load('../model/eventnum50_input10_acc99_transform_state_dict.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    event_pred_model.eval()
    #num_all_log=sum(single_log[1] for single_log in log_patterns)
    ori_all_event_num=len(event_list)
    print(ori_all_event_num)
    #num_all_doc=len(log_data)+2
    occurrence_prob_list=[]
    repeat_rate_list=[]
    abnormal_score_list=[]
    tf_idf_list=[]

    input_dim=10
    for single_file_data in tqdm(log_data):
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
                        prediction=event_pred_model(input_data.to(device)).cpu()
                    occurence_probability=np.exp(prediction[0][event_num-1].item())
                    model_input=model_input[len(model_input)-input_dim+1:]
                else:
                    occurence_probability=1
            # Repeat Rate calculate
            recent_event_nums=[x for x in recent_event_nums if x[0]>date_now-timedelta(minutes=repeat_threshold)]
            reapeat_rate=ln((len([x for x in recent_event_nums if x[1]==event_num])+2))
            
            abnormal_score=tf_idf[find_pattern_num(single_pattern,log_patterns)-1]*(1-occurence_probability)*reapeat_rate
            tf_idf_list.append(tf_idf[find_pattern_num(single_pattern,log_patterns)-1])
            occurrence_prob_list.append(occurence_probability)
            repeat_rate_list.append(reapeat_rate)
            abnormal_score_list.append(abnormal_score)
            # Update recent_event_nums and etc
            recent_event_nums.append([date_now, event_num])
            if event_num<=ori_all_event_num:
                model_input.append(event_num-1)
    print(f'average of occurence_probability is {np.average(occurrence_prob_list)} and std is {np.std(occurrence_prob_list)} and max is {np.max(occurrence_prob_list)} and min is {np.min(occurrence_prob_list)}')
    print(f'average of repeat_rate is {np.average(repeat_rate_list)} and std is {np.std(repeat_rate_list)} and max is {np.max(repeat_rate_list)} and min is {np.min(repeat_rate_list)}')
    print(f'average of abnormal_score is {np.average(abnormal_score_list)} and std is {np.std(abnormal_score_list)} and max is {np.max(abnormal_score_list)} and min is {np.min(abnormal_score_list)}')
    print(f'average of tf_idf is {np.average(tf_idf_list)} and std is {np.std(tf_idf_list)} and max is {np.max(tf_idf_list)} and min is {np.min(tf_idf_list)}')
    return occurrence_prob_list, repeat_rate_list, abnormal_score_list, tf_idf_list

def calculate_abnormal_score_for_df(log_data, log_dict,log_patterns, event_list, tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict=None):
    # Now, log_data is DataFrame that has all log data
    # and it's row is datetime.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #event_pred_model=torch.load('model/eventnum50_input10_acc99_transform.pt')
    #event_pred_model.load_state_dict(torch.load('model/eventnum50_input10_acc99_transform_state_dict.pt'))
    event_pred_model.eval()
    #num_all_log=len(log_data)
    ori_all_event_num=len(event_list)
    #print(ori_all_event_num)
    occurrence_prob_list=[]
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
            print('unseen log pattern')
            log_patterns.append([single_pattern, 1])
            tf_idf.append(ln(ln(num_all_log)*(num_all_doc+2)))
            assert(len(log_patterns)==len(tf_idf))
            find_event,event_list = put_new_pattern_to_event_list(single_pattern, event_list, synant_dict)
            if find_event:
                event_num=find_event_num(single_pattern,event_list)
            else:
                print('unseen log event')
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
                    prediction=event_pred_model(input_data.to(device)).cpu()
                occurence_probability=np.exp(prediction[0][event_num-1].item())
                model_input=model_input[len(model_input)-input_dim+1:]
            else:
                occurence_probability=0.9
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
        recent_event_nums.append([date_now, event_num])
        if event_num<=ori_all_event_num:
            model_input.append(event_num-1)
    return ab_df

def anomaly_detection_for_file(single_log_data, log_dict, log_patterns, event_list, tf_idf, num_all_doc, 
                               num_all_log, event_pred_model, synant_dict=None, threshold=[15,2], way='repeat'):
    assert(len(log_patterns)==len(tf_idf) and len(tf_idf) ==sum([len(x) for x in event_list]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #num_all_log=sum(single_log[1] for single_log in log_patterns)
    ori_all_event_num=len(event_list)
    #print(ori_all_event_num)
    #num_all_doc=len(log_data)+2
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
                    prediction=event_pred_model(input_data.to(device)).cpu()
                occurence_probability=np.exp(prediction[0][event_num-1].item())
                model_input=model_input[len(model_input)-input_dim+1:]
            else:
                occurence_probability=0.9
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