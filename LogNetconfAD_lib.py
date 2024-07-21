import os 
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from log_parser_lib import *
from AB_score_lib import *
import datetime

class HeavyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HeavyRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size*2,batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, int(input_size/2))
        self.fc4 = nn.Linear(int(input_size/2), 5)
        self.fc5 = nn.Linear(5, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out[:, -1, :])  # 마지막 time step의 출력만 사용
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(input_size, int(input_size/2))
        self.fc3 = nn.Linear(int(input_size/2), 5)
        self.fc4 = nn.Linear(5, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out[:, -1, :])  # 마지막 time step의 출력만 사용
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class CustomPaddedDataset(Dataset):
    def __init__(self, dataframe_groupby, output_data, input_feature_num):
        self.dataframe_groupby = dataframe_groupby
        self.group_names=[]
        for key, group in dataframe_groupby:
            if len(group)!=0:
                self.group_names.append(key)            
        self.max_seq_len=max(len(group[1]) for group in dataframe_groupby)
        self.output_data=output_data

    def __len__(self):
        return len(self.dataframe_groupby)

    def __getitem__(self, idx):
        # 각 그룹의 데이터 가져오기
        group = self.dataframe_groupby.get_group(self.group_names[idx])
        group_tensor = torch.tensor(group.values, dtype=torch.float32)
        pad_amount=[]
        padded_group_tensor=nn.functional.pad(group_tensor, pad=(0,0,0,self.max_seq_len-len(group)))
        #features = group.values  # 입력 특성
        return padded_group_tensor, torch.tensor(self.output_data[idx])

def calculate_average_values(row_B, df_A, column_name, window_minutes):
    # NETCONF data is too frequent. Average values are calculated within a certain time window.
    print(column_name)
    window_start = row_B.name - pd.Timedelta(minutes=window_minutes)
    window_end = row_B.name + pd.Timedelta(minutes=window_minutes)
    filtered_values = df_A[column_name].loc[(df_A.index >= window_start) & (df_A.index <= window_end)]
    if not filtered_values.empty:
        return filtered_values.mean()
    else:
        return None

def AD_with_DF(data, num_epochs = 1000,epoch_print_period=50, freq='6H', lr=0.01, use_simple=True):
    # fill missing data
    data.fillna(0,inplace=True)
    # remove the ambiguos data
    condition1=data.index > datetime.datetime(2024,1,17,0,0)
    condition2=data.index < datetime.datetime(2024,1,16,12,0)
    data=data.loc[condition1|condition2]
    grouped_data=data.groupby(pd.Grouper(freq=freq))
    input_feature_num=data.shape[1]
    dataset_size=len(grouped_data)
    tmp_=0
    for key, group in grouped_data:
        if len(group)==0:
            tmp_+=1
    dataset_size-=tmp_
    hidden_size = input_feature_num*2 
    output_size = 1 
    batch_size = 20

    train_ratio = 0.7
    valid_ratio = 1 - train_ratio
    indices = list(range(dataset_size))
    split = int(train_ratio * dataset_size)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # It is divided into abnormal/normal data based on split_time.
    split_timestamp=datetime.datetime(2024,1,16,12,0)
    targets=[1 if k < split_timestamp else 0 for k in grouped_data.groups]

    if use_simple:
        model = SimpleRNN(input_feature_num, hidden_size, output_size)
    else:
        model = HeavyRNN(input_feature_num, hidden_size, output_size)
    dataset = CustomPaddedDataset(grouped_data, targets, input_feature_num)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(dataset,sampler=valid_sampler)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses=[]
    accuraccies=[]
    epoches=[]
    f1=[]
    for epoch in range(num_epochs):
        for batch_inputs, batch_outputs in train_dataloader:
            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(batch_inputs)  # 모델 예측
            #print(outputs)
            #print(outputs.squeeze().shape, batch_outputs.shape)
            loss = criterion(outputs.squeeze(), batch_outputs.float())  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
        if epoch%50 ==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
            epoches.append(epoch+1)
            right=0 
            false=0
            tp=0
            fp=0
            fn=0
            tn=0
            for batch_inputs, batch_outputs in test_dataloader:
                outputs=model(batch_inputs)
                if outputs > 0.5:
                    if batch_outputs==1:
                        tp+=1
                        right+=1
                    else:
                        fp+=1
                        false+=1
                else:
                    if batch_outputs==1:
                        fn+=1
                        false+=1
                    else:
                        tn+=1
                        right+=1
            accuraccies.append(right/(right+false))
            prec=tp/(tp+fp+1e-6)
            rec=tp/(tp+fn+1e-6)
            f1.append(2*prec*rec/(prec+rec+1e-6))
            print(f'Predciction accuracy is {right/(right+false):.2f}')
            print(f'F1 score is {2*prec*rec/(prec+rec+1e-6):.2f}')
            print(f'{right} corrects among {right+false} data')
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoches')
    ax1.set_ylabel('Losses', color=color)
    ax1.plot(epoches, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('F1score', color=color)
    ax2.plot(epoches, f1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Losses, F1 scores')
    plt.show()

def AD(normal_data, abnormal_data, log_dict, log_patterns, event_list, tf_idf, num_all_doc, num_all_log, 
       event_pred_model, synant_dict=None, num_epochs = 1000, epoch_print_period=50, lr=0.01, use_simple=True):
    # Let's calculate AB score and merge with NETCONF data
    x_data=[]
    y_data=[]
    input_feature_num=0
    for date in normal_data.keys():
        #mon=int(date[:2])
        #day=int(date[3:])
        log_df=pd.DataFrame(normal_data[date]['log'])
        print(log_df.shape)
        print(log_df.columns)

        #log_df['date']=log_df['date'].apply(lambda x: x.replace(year=2024, month=mon, day=day))
        log_df.set_index('date', inplace=True)
        tf_idf_results=calculate_abnormal_score_for_df(log_df,log_dict, log_patterns, event_list, 
                            tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict)
        print(tf_idf_results.shape)
        for column_name, cloumn in normal_data[date]['netconf'].items():
            tf_idf_results['average_'+column_name]=tf_idf_results.apply(
                lambda row: calculate_average_values(row, normal_data, column_name, 10), axis=1)
            print(tf_idf_results.shape)
            if not input_feature_num:
                input_feature_num=tf_idf_results.shape[1]
            assert(input_feature_num==tf_idf_results.shape[1])
            x_data.append(tf_idf_results)
            y_data.append(0)

    for date in abnormal_data.keys():
        #mon=int(date[:2])
        #day=int(date[3:])
        log_df=pd.DataFrame(abnormal_data[date]['log'])
        #log_df['date']=log_df['date'].apply(lambda x: x.replace(year=2024, month=mon, day=day))
        log_df.set_index('date', inplace=True)
        tf_idf_results=calculate_abnormal_score_for_df(log_df,log_dict, log_patterns, event_list, 
                            tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict)
        for column_name, cloumn in abnormal_data[date]['netconf'].items():
            tf_idf_results['average_'+column_name]=tf_idf_results.apply(
                lambda row: calculate_average_values(row, abnormal_data, column_name, 10), axis=1)
            tf_idf_results.fillna(0, inplace=True)
            print(tf_idf_results.shape)
            assert(input_feature_num==tf_idf_results.shape[1])
            x_data.append(tf_idf_results)
            y_data.append(1)
    # fill missing data
    data.fillna(0,inplace=True)
    # remove the ambiguos data
    condition1=data.index > datetime.datetime(2024,1,17,0,0)
    condition2=data.index < datetime.datetime(2024,1,16,12,0)
    data=data.loc[condition1|condition2]
    grouped_data=data.groupby(pd.Grouper(freq=freq))
    input_feature_num=data.shape[1]
    dataset_size=len(grouped_data)
    tmp_=0
    for key, group in grouped_data:
        if len(group)==0:
            tmp_+=1
    dataset_size-=tmp_
    hidden_size = input_feature_num*2 
    output_size = 1 
    batch_size = 20

    train_ratio = 0.7
    valid_ratio = 1 - train_ratio
    indices = list(range(dataset_size))
    split = int(train_ratio * dataset_size)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # It is divided into abnormal/normal data based on split_time.
    split_timestamp=datetime.datetime(2024,1,16,12,0)
    targets=[1 if k < split_timestamp else 0 for k in grouped_data.groups]

    if use_simple:
        model = SimpleRNN(input_feature_num, hidden_size, output_size)
    else:
        model = HeavyRNN(input_feature_num, hidden_size, output_size)
    dataset = CustomPaddedDataset(grouped_data, targets, input_feature_num)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(dataset,sampler=valid_sampler)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses=[]
    accuraccies=[]
    epoches=[]
    f1=[]
    for epoch in range(num_epochs):
        for batch_inputs, batch_outputs in train_dataloader:
            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(batch_inputs)  # 모델 예측
            #print(outputs)
            #print(outputs.squeeze().shape, batch_outputs.shape)
            loss = criterion(outputs.squeeze(), batch_outputs.float())  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
        if epoch%50 ==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
            epoches.append(epoch+1)
            right=0 
            false=0
            tp=0
            fp=0
            fn=0
            tn=0
            for batch_inputs, batch_outputs in test_dataloader:
                outputs=model(batch_inputs)
                if outputs > 0.5:
                    if batch_outputs==1:
                        tp+=1
                        right+=1
                    else:
                        fp+=1
                        false+=1
                else:
                    if batch_outputs==1:
                        fn+=1
                        false+=1
                    else:
                        tn+=1
                        right+=1
            accuraccies.append(right/(right+false))
            prec=tp/(tp+fp+1e-6)
            rec=tp/(tp+fn+1e-6)
            f1.append(2*prec*rec/(prec+rec+1e-6))
            print(f'Predciction accuracy is {right/(right+false):.2f}')
            print(f'F1 score is {2*prec*rec/(prec+rec+1e-6):.2f}')
            print(f'{right} corrects among {right+false} data')
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoches')
    ax1.set_ylabel('Losses', color=color)
    ax1.plot(epoches, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('F1score', color=color)
    ax2.plot(epoches, f1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Losses, F1 scores')
    plt.show()