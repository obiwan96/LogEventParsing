import torch.nn.functional as F
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import gc
from log_parser_lib import *
import torch
import time

model_name = "bert-base-uncased"
epsilon = 1e-6
num_train_epochs = 100
print_each_n_step = 5
input_seq=10
max_token_length=20
learning_rate=0.01
batch_size=100

# LSTM model.
# get last input_dim number of event numbers and predict the next event number.
class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_model, self).__init__()
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        #self.fc=nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x = [batch_size, input_seq, input_dim]
        out, h=self.lstm(x)
        # out = [batch_size, input_seq, hidden_dim]
        # h[0] = [num_layers, batch_size, hidden_dim], but we will use only the last state
        out = h[0][-1].squeeze(0)
        # out = [batch_size, hidden_dim]
        return out # using only last state
transformer = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
BERT_hidden_size = int(config.hidden_size)
    
log_path='../log_dpnm_tb'
if __name__ == '__main__':
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
    date_list=os.listdir(log_path)
    input_data=[]
    output_data=[]
    for date in date_list:
        event_flow=[]
        if os.path.isfile(log_path+'/'+date):
            continue
        log_ = read_file(log_path+'/'+date+'/all.log')
        for single_log in log_:
            single_log=re.sub(r'[0-9]+','',single_log['log']).lower().translate(translator)
            sentence_token = tokenizer.encode(single_log, max_length=max_token_length,padding='max_length',
                truncation=True, add_special_tokens=True)
            #print(single_log, len(sentence_token))
            assert len(sentence_token) == max_token_length
            event_flow.append(sentence_token)
            if len(event_flow)>input_seq:
                input_data.append(event_flow[:input_seq])
                output_data.append(event_flow[input_seq])
                event_flow=[]
    
    input_data=torch.tensor(input_data,dtype=torch.int64)
    output_data=torch.tensor(output_data)
    print(input_data.shape, output_data.shape)
    data_size=input_data.size(dim=0)
    indices=torch.randperm(data_size)
    input_data=input_data[indices]
    output_data=output_data[indices]
    x_train=input_data[:int(data_size/2)]
    y_train=output_data[:int(data_size/2)]
    #x_train=input_data[:data_size-int(data_size/5)]
    #y_train=output_data[:data_size-int(data_size/5)]
    x_test=input_data[data_size-int(data_size/4):]
    y_test=output_data[data_size-int(data_size/4):]
    print(f'train data has {len(x_train)} data and test data has {len(x_test)} data')
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Let's train
    model=lstm_model(BERT_hidden_size,BERT_hidden_size)
    transformer_vars = [i for i in transformer.parameters()]
    model_vars = transformer_vars + [v for v in model.parameters()]
    loss_function = nn.CosineEmbeddingLoss()
    optimizer = optim.SGD(model_vars, lr=learning_rate)
    if torch.cuda.is_available():
        model.cuda()
        transformer.cuda()
        device = torch.device("cuda")
        loss_function.cuda()
    else:
        device = torch.device("cpu")
    writer = SummaryWriter('/home/dpnm/tmp/tensorboard/')
    start=time.time()
    for epoch in range(num_train_epochs):
        model.train()
        transformer.train()
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples[0].to(device), samples[1].to(device)
            small_batch_size=x_train.shape[0]
            #print(x_train.shape, y_train.shape)
            sent_vector=torch.empty(small_batch_size, 0, BERT_hidden_size).to(device)
            for i in range(input_seq):
                out=transformer(x_train[:,i,:].squeeze(1)).last_hidden_state[:,0,:].unsqueeze(1) #only use sentence BERT.
                sent_vector=torch.cat((sent_vector,out),1)
            #print(sent_vector.shape)
            prediction=model(sent_vector)
            #print(prediction.shape)
            y_train = transformer(y_train).last_hidden_state[:,0,:]
            #print(y_train.shape)
            #prediction=torch.argmax(prediction, dim=1)
            #print(prediction.shape)
            loss=loss_function(prediction, y_train,torch.Tensor(small_batch_size).cuda().fill_(1.0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%print_each_n_step==0:
            print('Epoch {:4d}/{} Batch {}/{} loss: {:.6f}'.format(
                    epoch, num_train_epochs, batch_idx+1, len(dataloader),loss.item() ))
            writer.add_scalar("Training/Loss", loss.item(),epoch+1)
            print('running test..')
            cos=nn.CosineSimilarity(dim=1)
            euc_dist=nn.PairwiseDistance()
            model.eval()
            transformer.eval()
            all_preds=[]
            all_labels=[]
            right_num=0
            wrong_num=0
            euc_right_num=0
            euc_wrong_num=0
            for batch in test_dataloader:
                x,y=batch[0].to(device), batch[1].to(device)
                small_batch_size=x.shape[0]
                sent_vector=torch.empty(small_batch_size, 0, BERT_hidden_size).to(device)
                with torch.no_grad():
                    for i in range(input_seq):
                        out=transformer(x[:,i,:].squeeze(1)).last_hidden_state[:,0,:].unsqueeze(1)
                        sent_vector=torch.cat((sent_vector,out),1)
                    prediction=model(sent_vector)
                    y = transformer(y).last_hidden_state[:,0,:]
                norm_pred=F.normalize(prediction,dim=1)
                norm_y=F.normalize(y,dim=1)
                small_batch_right=(cos(prediction,y)>1-5e-4).sum()
                small_batch_euc_right=(euc_dist(norm_pred,norm_y)<0.01).sum()
                right_num+=small_batch_right
                wrong_num+=small_batch_size-small_batch_right
                euc_right_num+=small_batch_euc_right
                euc_wrong_num+=small_batch_size-small_batch_euc_right
            test_accuracy=right_num/(right_num+wrong_num)
            euc_test_accuracy=euc_right_num/(euc_right_num+euc_wrong_num)
            print(f"Test Accuracy: {test_accuracy:.3f}. Predict {right_num} through {right_num+wrong_num}")
            print(f"Euc Test Accuracy: {euc_test_accuracy:.3f}. Predict {euc_right_num} through {euc_right_num+euc_wrong_num}")
            writer.add_scalar("Test/Accuracy", test_accuracy,epoch+1)
    writer.close()
    print(f"Learning takes {(time.time()-start)/60:.2f} minutes")
    torch.save(model, f'/home/dpnm/tmp/model/acc{test_accuracy*100:.0f}.pt')
    torch.save(model.state_dict(), f'/home/dpnm/tmp/model/acc{test_accuracy*100:.0f}_state_dict.pt')
    torch.save(transformer, f'/home/dpnm/tmp/model/transformer_acc{test_accuracy*100:.0f}.pt')
    torch.save(transformer.state_dict(), f'/home/dpnm/tmp/model/transformer_acc{test_accuracy*100:.0f}_state_dict.pt')
    #torch.save(model, f'/home/obiwan/tmp/model/input{input_dim}_acc{test_accuracy*100:.0f}.pt')

