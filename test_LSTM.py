import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import os
from log_parser_lib import *
import time

# LSTM model hyperparameters
# event_num will be set in __main__
event_num=211
output_dim=event_num
learning_rate=0.1
n_epochs=500

# LSTM model.
# get last input_dim number of event numbers and predict the next event number.
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
    log_path_list=['/mnt/e/obiwan/SNIC Log/1st','/mnt/e/obiwan/SNIC Log/2nd','../1st_example_log']
    for log_path_ in log_path_list:
        log_data.extend(read_log_files(log_path_))
    log_dict, synant_dict=make_dict(log_data)
    log_patterns=make_log_pattern_dict(log_data, log_dict)
    event_list=classify_pattern_to_events(log_patterns,synant_dict)
    print(f'total {len(event_list)} number of events are classified')

    event_num=len(event_list)
    for input_dim in [5, 10, 20, 40]:
        print(f'input dimension is {input_dim}')
        print('Learning 3 times')
        for _ in range(3):
            hidden_dim=input_dim*2
            model=lstm_model(input_dim, hidden_dim, event_num)

            input_data=[]
            output_data=[]
            for date in date_list:
                event_flow=[]
                if os.path.isfile(log_path+'/'+date):
                    continue
                log = read_file(log_path+'/'+date+'/all.log')
                for single_log in log:
                    single_pattern=log_parser(single_log, log_dict)
                    if single_pattern[0] in ['adt', 'fan']:
                        continue
                    log_event_num=find_event_num(single_pattern,event_list)
                    if not log_event_num:
                        #print(f'event num not found! {single_log}')
                        continue
                    event_flow.append(log_event_num)
                if len(event_flow)>input_dim:
                    for i in range(len(event_flow)-input_dim):
                        input_data.append(event_flow[i:i+input_dim])
                        output_data.append(event_flow[i+input_dim]-1)
            input_data=torch.tensor(input_data,dtype=torch.float32)
            output_data=torch.tensor(output_data)
            #output_data=F.one_hot(output_data,num_classes=event_num)
            #print(input_data.shape, output_data.shape)
            data_size=input_data.size(dim=0)
            indices=torch.randperm(data_size)
            input_data=input_data[indices]
            output_data=output_data[indices]
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
            start=time.time()
            for epoch in range(n_epochs):
                for batch_idx, samples in enumerate(dataloader):
                    x_train, y_train = samples
                    #print(x_train.shape, y_train.shape)
                    prediction=model(x_train)
                    #prediction=torch.argmax(prediction, dim=1)
                    loss=loss_function(prediction, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                #print('Epoch {:4d}/{} Batch {}/{} loss: {:.6f}'.format(
                #        epoch, n_epochs, batch_idx+1, len(dataloader),loss.item() ))
            model.eval()
            all_preds=[]
            all_labels=[]
            for batch in test_dataloader:
                x,y=batch
                with torch.no_grad():
                    prediction=model(x)
                #all_preds +=torch.argmax(prediction,dim=1)
                all_preds += torch.topk(prediction,5,dim=1)[1]
                all_labels += y
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
            print(f"Learning takes {(time.time()-start)/60:.2f} minutes\n")