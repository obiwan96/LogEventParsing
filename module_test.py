from log_parser_lib import *
from automathon import DFA
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import *

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

event_pred_model=torch.load('../model/input20_acc95.pt')
event_pred_model.load_state_dict(torch.load('../model/input20_acc95_state_dict.pt'))
# LSTM model for log prediciton. Sadly, name is same as above..
class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_model, self).__init__()
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        # x = [batch_size, input_seq, input_dim]
        out, h=self.lstm(x)
        # out = [batch_size, input_seq, hidden_dim]
        # h[0] = [num_layers, batch_size, hidden_dim], but we will use only the last state
        out = h[0][-1].squeeze(0)
        # out = [batch_size, hidden_dim]
        return out # using only last state

input_dim=20
model_name = "bert-base-uncased"
input_seq=10
max_token_length=20

if __name__ == '__main__':
    if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
    with open('data.pkl','rb') as f:
        data=pkl.load(f)
    (log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
    automata1 = DFA(Q, sigma, delta, initialState, F_)
    print(f'event num is {len(event_list)}')
    tokenizer = AutoTokenizer.from_pretrained(model_name) #BERT Tokenizer
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))

    normal_log_path='../log_dpnm_tb'
    abnormal_log_path='../anomaly_log'
    overloaded_log_path='../overloaded_log'
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
    
    # Loading Models
    event_pred_model.eval()
    log_pred_model=torch.load('../model/acc100.pt')
    #print(log_pred_model.__class__.__name__)
    log_pred_model.load_state_dict(torch.load('../model/acc100_state_dict.pt'))
    log_pred_model.cuda()
    log_pred_model.eval()
    # Loading BERT
    transformer = AutoModel.from_pretrained(model_name)
    transformer.cuda()
    transformer.load_state_dict(torch.load('../model/transformer_acc100_state_dict.pt', map_location=device))
    config = AutoConfig.from_pretrained(model_name)
    BERT_hidden_size = int(config.hidden_size)
    cos=nn.CosineSimilarity(dim=1)

    # Let's go
    tp=fp=fn=tn=0
    lstm_right=lstm_wrong=0
    for file_num, single_file_data in enumerate(log_data):
        print(f'processing {file_num+1}th file of {len(log_data)} num of files')
        model_input=[]
        BERT_input=torch.empty(1, 0, BERT_hidden_size).to(device)
        prev_event_num=find_event_num(log_parser(single_file_data[0][0],log_dict),event_list)
        model_input.append(prev_event_num)
        single_log_=re.sub(r'[0-9]+','',single_file_data[0][0]['log']).lower().translate(translator)
        sentence_token = tokenizer.encode(single_log_, max_length=max_token_length,padding='max_length',
            truncation=True, add_special_tokens=True)
        with torch.no_grad():
            out=transformer(torch.tensor([sentence_token]).to(device)).last_hidden_state[:,0,:].unsqueeze(1)
        BERT_input=torch.cat((BERT_input,out),1)
        abnormal=False
        abnormal_continue=0
        for single_log in single_file_data[0][1:]:
            single_pattern=log_parser(single_log, log_dict)
            if single_pattern==[]:continue
            
            # Pattern test
            if find_pattern_num(single_pattern,log_patterns) == None:
                if not single_file_data[1]:
                    print('pattern no find ')
                    print(single_pattern)
                abnormal=True
                break
            event_num=find_event_num(single_pattern,event_list)
            present_sentence=re.sub(r'[0-9]+','',single_log['log']).lower().translate(translator)
            present_sentence_token = tokenizer.encode(present_sentence, max_length=max_token_length,padding='max_length',
                truncation=True, add_special_tokens=True)
            #with torch.no_grad():
            #    present_sentence_vector=transformer(torch.tensor([present_sentence_token]).to(device)).last_hidden_state[:,0,:].unsqueeze(1)
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
                    print('fsm state no find ')
                    print(single_file_data[0][0])
                #break
            
            # Event Prediction (LSTM) test
            if len(model_input)==input_dim:
                input_data=torch.tensor([model_input[:]],dtype=torch.float32)
                with torch.no_grad():
                    prediction=event_pred_model(input_data)
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
            
            # Log Prediction (BERT+LSTM) test
            if BERT_input.shape[1]==input_seq:
                with torch.no_grad():
                    prediction=log_pred_model(BERT_input)
                #print(prediction.shape, present_sentence_vector.shape)
                #print(cos(prediction.unsqueeze(0), present_sentence_vector.squeeze())< 1-5e4)
                if cos(prediction.unsqueeze(0), present_sentence_vector.squeeze()) < 1-5e4:
                    print('log predictioner find abnormal')
                    abnormal=True
                    break
                BERT_input=BERT_input[:,1:,:]
            prev_event_num = event_num
            prev_log_message=single_log['log']
            model_input.append(event_num)
            #BERT_input=torch.cat((BERT_input,present_sentence_vector),1)
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
