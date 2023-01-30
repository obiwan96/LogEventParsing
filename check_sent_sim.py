from SNIC_log_reading_and_parsing import read_log_files, make_dict, make_log_pattern_dict
from transformers import *
import torch

if __name__ == '__main__':
    log_data = read_log_files()
    log_data.extend(read_log_files('SNIC Log/1st'))
    log_data.extend(read_log_files('SNIC Log/2nd'))
    print(f'##Read total {len(log_data)} num of logs##')
    log_dict=make_dict(log_data)
    log_patterns=make_log_pattern_dict(log_data, log_dict)

    # BERT
    model_name = "bert-base-uncased"
    max_seq_length = 50
    max_sent_length=350
    bert_path='/home/dpnm/tmp/runs/best_model_bert'
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    transformer.cuda()
    transformer.load_state_dict(torch.load(bert_path, map_location=device))
    transformer.eval()

    hidden_states=[]
    transformer=transformer.to(device)
    log_patterns_cpy=log_patterns
    while len(log_patterns) > 0:
        if len(log_patterns) > max_sent_length:
            tmp_log_patterns=log_patterns[:max_sent_length]
            log_patterns=log_patterns[max_sent_length:]
        else:
            tmp_log_patterns=log_patterns
            log_patterns=[]
        input_log_token=[]
        for single_pattern in tmp_log_patterns:
            input_log_token.append(tokenizer.encode(single_pattern[0], is_split_into_words=True,add_special_tokens=True,
                max_length=max_seq_length, padding="max_length", truncation=True))
        att_mask=[]
        for sentence_token in input_log_token:
            att_mask.append([int(token_id > 0) for token_id in sentence_token])
        input_log_token=torch.tensor(input_log_token).to(device)
        att_mask=torch.tensor(att_mask).to(device)
        #print(input_log_token.size(), att_mask.size())
        hidden_states.extend(transformer(input_log_token, attention_mask=att_mask)[1])
    #print(len(hidden_states))
    log_patterns=log_patterns_cpy
    # Let's see the similarity
    test_num=1
    print("\nLet's calculate the similarity based on Sent-BERT")
    print('object sentence : '+' '.join(log_patterns[test_num][0]))
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    eud=torch.nn.PairwiseDistance()
    cos_sim_dict={}
    eud_sim_dict={}
    for i in range(len(log_patterns)):
        cos_sim_dict[log_patterns[i][0]]=cos(hidden_states[test_num], hidden_states[i])
        eud_sim_dict[log_patterns[i][0]]=eud(hidden_states[test_num].reshape(1,hidden_size), hidden_states[i].reshape(1,hidden_size))

    print_untill=20
    print('\n## Based on cosine similarity')
    cos_sim_dict=sorted(cos_sim_dict.items(), key=lambda item: item[1], reverse=True)
    for i in range(print_untill):
        print(' '.join(list(cos_sim_dict[i][0]))+ f' : {cos_sim_dict[i][1].item()}')

    print('\n## Based on euclidian similarity')
    eud_sim_dict=sorted(eud_sim_dict.items(), key=lambda item: item[1], reverse=False)
    for i in range(print_untill):
        print(' '.join(list(eud_sim_dict[i][0]))+ f' : {eud_sim_dict[i][1].item()}')
    





