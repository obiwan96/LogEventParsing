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
from tqdm import tqdm, trange
import pickle as pkl

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

class LSTM_netconf(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_netconf, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(input_size, int(input_size/2))
        self.fc3 = nn.Linear(int(input_size/2), 5)
        self.fc4 = nn.Linear(5, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # 마지막 time step의 출력만 사용
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        pe.required_grad=False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        #return x
        return self.dropout(x)

class Transformer_netconf(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers=2, nhead=2):
        super(Transformer_netconf, self).__init__()
        if input_size%5==0:
            nhead=5
        elif input_size%3==0:
            nhead=3
        elif input_size%2==0:
            nhead=2
        else:
            nhead=1
        #self.embedding=nn.Linear(input_size, input_size*2)
        #self.pos_encoder = PositionalEncoding(input_size*2)
        self.encoder_layers = nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward =hidden_dim ,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        #self.fc1 = nn.Linear(input_size*2, input_size)
        self.fc2 = nn.Linear(input_size, int(input_size / 2))
        self.fc3 = nn.Linear(int(input_size / 2), 5)
        self.fc4 = nn.Linear(5, output_size)
    
    def forward(self, x):
        #x = x.permute(1, 0, 2)
        #x=self.embedding(x)#.transpose(0,1))
        #x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 마지막 time step의 출력만 사용
        #x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
class CustomPaddedDataset(Dataset):
    def __init__(self, x_data, output_data, input_feature_num):
        self.x_data = x_data
        #self.group_names=list(range(len(x_data)))
        self.max_seq_len=max(len(group) for group in x_data)
        self.output_data=output_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 각 그룹의 데이터 가져오기
        group = self.x_data[idx]
        group_tensor = torch.tensor(group.values, dtype=torch.float32)
        pad_amount=[]
        padded_group_tensor=nn.functional.pad(group_tensor, pad=(0,0,0,self.max_seq_len-len(group)))
        #features = group.values  # 입력 특성
        return padded_group_tensor, torch.tensor(self.output_data[idx])

def calculate_average_values(row_B, df_A, column_name, window_minutes):
    # NETCONF data is too frequent. Average values are calculated within a certain time window.
    #print(column_name)
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
            eps=1e-10
            loss = criterion(outputs.squeeze()+eps, batch_outputs.float())  # 손실 계산
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

def Make_data_for_AD(normal_data, abnormal_data, log_dict, log_patterns, event_list, tf_idf, num_all_doc, num_all_log, 
       event_pred_model, whole_netconf_features, synant_dict=None):
    # Let's calculate AB score and merge with NETCONF data
    x_data=[]
    y_data=[]
    input_feature_num=len(whole_netconf_features)+3
    for date in tqdm(normal_data.keys()):
        #print(date)
        #print(normal_data[date]['log'])
        #mon=int(date[:2])
        #day=int(date[3:])
        #log_len=len(normal_data[date]['log'])
        log_df=pd.DataFrame(normal_data[date]['log'])
        
        #log_df['date']=log_df['date'].apply(lambda x: x.replace(year=2024, month=mon, day=day))
        log_df.set_index('date', inplace=True)
        tf_idf_results=calculate_abnormal_score_for_df(log_df,log_dict, log_patterns, event_list[:], 
                            tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict)
        #print(tf_idf_results.shape)
        zero_column_name=[]
        for column_name in whole_netconf_features:
            if column_name not in normal_data[date]['netconf'].columns:
                zero_column_name.append(column_name)
                #tf_idf_results['average-'+column_name]=0
            else:
                tmp_column=pd.Series(tf_idf_results.apply(lambda row: 
                        calculate_average_values(row, normal_data[date]['netconf'], column_name, 10), axis=1), name='average-'+column_name)
                tf_idf_results=pd.concat([tf_idf_results, tmp_column], axis=1)
        tf_idf_results=pd.concat((pd.concat(((pd.Series(np.zeros(tf_idf_results.shape[0]), 
                    name='average-'+column_name)) for column_name in zero_column_name), axis=1), tf_idf_results), axis=1)
        tf_idf_results.fillna(0, inplace=True)
        #print(tf_idf_results.shape)
        assert(input_feature_num==tf_idf_results.shape[1])
        x_data.append(tf_idf_results)
        y_data.append(0)

    for date in tqdm(abnormal_data.keys()):
        #mon=int(date[:2])
        #day=int(date[3:])
        log_df=pd.DataFrame(abnormal_data[date]['log'])
        #log_df['date']=log_df['date'].apply(lambda x: x.replace(year=2024, month=mon, day=day))
        log_df.set_index('date', inplace=True)
        tf_idf_results=calculate_abnormal_score_for_df(log_df,log_dict, log_patterns, event_list[:], 
                            tf_idf, num_all_doc, num_all_log, event_pred_model, synant_dict)
        zero_column_name=[]
        for column_name in whole_netconf_features:
            if column_name not in abnormal_data[date]['netconf'].columns:
                zero_column_name.append(column_name)
                #tf_idf_results['average-'+column_name]=0
            else:
                tmp_column=pd.Series(tf_idf_results.apply(lambda row: 
                        calculate_average_values(row, abnormal_data[date]['netconf'], column_name, 10), axis=1), name='average-'+column_name)
                tf_idf_results=pd.concat([tf_idf_results, tmp_column], axis=1)
        tf_idf_results=pd.concat((pd.concat(((pd.Series(np.zeros(tf_idf_results.shape[0]), 
                    name='average-'+column_name)) for column_name in zero_column_name), axis=1), tf_idf_results), axis=1)
        tf_idf_results.fillna(0, inplace=True)
        #print(tf_idf_results.shape)
        assert(input_feature_num==tf_idf_results.shape[1])
        x_data.append(tf_idf_results)
        y_data.append(1)
    return (x_data, y_data)

def seperate_data(data, input_feature_num=None, ratio=0.8, batch_size = 20):
    x_data, y_data = data
    dataset_size=len(y_data)
    print(f'data set has {dataset_size} data')
    indices = list(range(dataset_size))
    split = int(ratio * dataset_size)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:split], indices[split-9:]
    except_indices=[10,11,12,53,54,55]
    train_indices=[i for i in train_indices if i not in except_indices]
    valid_indices=[i for i in valid_indices if i not in except_indices]
    print (train_indices)
    print(valid_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    dataset = CustomPaddedDataset(x_data, y_data, input_feature_num)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(dataset,sampler=valid_sampler)
    return (train_dataloader, test_dataloader)

def AD(dataloader, input_feature_num, model_name='simple', num_epochs = 800, epoch_print_period=50, lr=0.01):
    assert(model_name in ['simple', 'heavy', 'lstm', 'transformer'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Training '+model_name+' model')
    train_dataloader, test_dataloader = dataloader
    hidden_size = input_feature_num*2 
    output_size = 1 

    if model_name=='simple':
        model = SimpleRNN(input_feature_num, hidden_size, output_size)
    elif model_name=='heavy':
        model = HeavyRNN(input_feature_num, hidden_size, output_size)
    elif model_name=='lstm':
        model = LSTM_netconf(input_feature_num, hidden_size, output_size)
    else:
        model = Transformer_netconf(input_feature_num, hidden_size, output_size, num_layers=1)
    model=model.to(device)
    pos_weight = torch.tensor([2]).to(device)
    eps=1e-10
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses=[]
    accuraccies=[]
    epoches=[]
    f1=[]
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_outputs in train_dataloader:
            outputs = model(batch_inputs.to(device))  
            loss = criterion(outputs.squeeze()+eps, batch_outputs.float().to(device))
            optimizer.zero_grad()  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping
            optimizer.step() 
        if epoch%epoch_print_period ==epoch_print_period-1:
            model.eval()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
            epoches.append(epoch+1)
            right=0 
            false=0
            tp=0
            fp=0
            fn=0
            tn=0
            right_index=[]
            for batch_inputs, batch_outputs in test_dataloader:
                outputs=model(batch_inputs.to(device)).cpu()
                if outputs >= 0.45:
                    if batch_outputs==1:
                        tp+=1
                        right+=1
                        right_index.append(1)
                    else:
                        fp+=1
                        false+=1
                        right_index.append(0)
                else:
                    if batch_outputs==1:
                        fn+=1
                        false+=1
                        right_index.append(0)
                    else:
                        tn+=1
                        right+=1
                        right_index.append(1)
            print(right_index)
            accuraccies.append(right/(right+false))
            prec=tp/(tp+fp+1e-6)
            rec=tp/(tp+fn+1e-6)
            f1.append(2*prec*rec/(prec+rec+1e-6))
            #print(f'Detection accuracy is {right/(right+false):.2f}')
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
    fig_file_name='loss_f1_rnn_'+str(max(f1))+'_'+model_name+'.png'
    plt.savefig('../results/'+fig_file_name)
    plt.clf()
    print('plot saved')
    return losses, f1