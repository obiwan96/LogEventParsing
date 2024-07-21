from log_parser_lib import *
from AB_score_lib import *
from LogNetconfAD_lib import *

import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from numpy import log as ln
from datetime import timedelta
import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def json_date_parsing(file_name):
    date_=datetime.datetime.strptime(file_name, '%Y-%m-%dT%H-%M-%S.json')
    return date_

def json_read_content(file_name, switch_type):
    assert switch_type in ['cisco', 'juniper']
    #print(file_name)
    with open(file_name) as f:
        json_data=json.load(f)
    if switch_type=='cisco':
        key_list=list(json_data.keys())
        #print(key_list)
        for key in key_list:
            if 'status' in key:
                for key2 in json_data[key].keys():
                    if json_data[key][key2] in ['up', 'on']:
                        json_data[key+'-'+key2]=1
                    elif json_data[key][key2] in ['down', 'off']:
                        json_data[key+'-'+key2]=0
                    elif type(json_data[key][key2])==int and json_data[key][key2]!=0:
                        json_data[key+'-'+key2]=json_data[key][key2]
                del json_data[key]
        json_data['temperature']=float(json_data['temperature'][:-1])

    else:
        del json_data['time'], json_data['down-interfaces'], json_data['up-interfaces'], json_data['interface-information']
        stat=json_data['statistics']
        del json_data['statistics']
        for protocol in stat.keys():
            for key, value in stat[protocol].items():
                if type(value)==int and value!=0:
                    json_data[protocol+'-'+key]=value
        json_data['chassi-temperature']=float(json_data['chassi-temperature'][:-1])
    json_data['cpu-util']=float(json_data['cpu-util'][:-1])
    json_data['mem-util']=float(json_data['mem-util'][:-1])
    #change all key name to add switch type infront of key name
    keys_list=list(json_data.keys())
    for key in keys_list:
        json_data[switch_type+'-'+key]=json_data[key]
        del json_data[key]
    return json_data

def read_netconf_data(file_path, date, switch_type):
    data=pd.DataFrame()
    find_file=False
    date_month, date_day = date
    file_list=os.listdir(file_path+'/'+switch_type)
    for file in file_list:
        if not file.endswith('.json'):
            continue
        file_date=json_date_parsing(file)
        if file_date.month==date_month and file_date.day==date_day:
            find_file=True
            data=pd.concat([data, pd.Series(json_read_content(file_path+'/'+switch_type+'/'+file, switch_type), name=file_date)])
    return data, find_file

def read_data(file_path):
    log_file_path=file_path+'/log_netconf/log'
    netconf_file_path=file_path+'/log_netconf/ncclient'
    netconf_data_path='../2406_3rd/ncclient_log'
    log_file_list=os.listdir(log_file_path)
    date_list=[]
    for file in log_file_list:
        tmp_date_list=os.listdir(log_file_path+'/'+file)
        date_list.extend(tmp_date_list)
    date_list=list(set(date_list))
    print(date_list)
    data={}
    for date in date_list[:1]:
        # read log data
        tmp_data={'log':[], 'netconf':[]}
        all_log=[]
        for file in log_file_list:
            if os.path.exists(log_file_path+'/'+file+'/'+date):
                file_list_=os.listdir(log_file_path+'/'+file+'/'+date)
                for file_name_ in file_list_:
                    log_=read_file(log_file_path+'/'+file+'/'+date+'/'+file_name_)
                    all_log.extend(log_)
            all_log=sorted(all_log, key=lambda x:x['date'])
            tmp_data['log']=all_log
        
        # read netconf data
        netconf_data=pd.DataFrame()
        date_month=int(date[:2])
        date_day=int(date[3:])

        # Cisco
        tmp_netconf_data, find_file_1=read_netconf_data(netconf_file_path, (date_month, date_day), 'cisco')
        netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=0)
        # Juniper
        tmp_netconf_data, find_file_2=read_netconf_data(netconf_file_path, (date_month, date_day), 'juniper')
        netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=0)
        if not find_file_1 and not find_file_2:
            tmp_netconf_data, _ =read_netconf_data(netconf_data_path, (date_month, date_day), 'cisco')
            netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=0)
            tmp_netconf_data, _ =read_netconf_data(netconf_data_path, (date_month, date_day), 'juniper')
            netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=0)
        tmp_data['netconf']=netconf_data
        data[date]=tmp_data
    print(f'{len(date_list)} num of date readed')
    return data

if __name__ == '__main__':
    loading_dict_data=True
    loading_model=True
    test_ab_score=False
    model_path='../model/eventnum302_input10_acc94_transformer'
    normal_log_path='../normal_data'
    abnormal_log_path='../overloaded_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('normal data read')
    normal_data=read_data(normal_log_path)
    print('abnormal data read')
    abnormal_data=read_data(abnormal_log_path)

    # Make Log Dictionary

    if loading_dict_data:
        with open('data.pkl','rb') as f:
            data=pkl.load(f)
        (log_dict, synant_dict, log_patterns,event_list),_ = data
        with open('tf_data.pkl', 'rb')as f:
            tf_idf, num_all_docs, num_all_log =pkl.load(f)
    else:
        normal_log_data=[]
        for date in normal_data.keys():
            normal_log_data.extend(normal_data[date]['log'])
        log_dict, synant_dict=make_dict(normal_log_data)
        log_patterns=make_log_pattern_dict(normal_log_data, log_dict)
        event_list=classify_pattern_to_events(log_patterns,synant_dict)
        event_num=len(event_list)
        data = ((log_dict, synant_dict, log_patterns,event_list), 0)
        print(f'total {event_num} number of events are classified')
        print(f'Save classified data to data.pkl')
        with open('data.pkl','wb') as f:
            pkl.dump(data,f)

        normal_log_data=[normal_data[date]['log'] for date in normal_data.keys()]
        tf_idf, num_all_docs, num_all_log = calculate_tf_idf(normal_log_data, log_dict, log_patterns)
        data = (tf_idf, num_all_docs, num_all_log)
        with open('tf_data.pkl','wb') as f:
            pkl.dump(data,f)
        del normal_log_data

    # Load Event Prediction Model or Train the model
    if loading_model:
        # Load model
        event_pred_model=torch.load(model_path+'.pt')
        event_pred_model.load_state_dict(torch.load(model_path+'_state_dict.pt'))
        event_pred_model.eval()
    else:
        # Train the Event Prediction Model
        print('Training the Event Prediction Model')
        normal_log_data=[normal_data[date]['log'] for date in normal_data.keys()]
        event_pred_model = event_prediction_model_training(normal_log_data, log_dict, event_list,n_epochs=500,reducing_rate=20,learning_rate=0.1)
        del normal_log_data
    
    normal_log_data=[normal_data[date]['log'] for date in normal_data.keys()]
    abnormal_log_data=[abnormal_data[date]['log'] for date in abnormal_data.keys()]
    '''
    occurrence_porb_list, repeat_rate_list, abnormal_score_list, tf_idf_list = calculate_abnormal_score_for_files(normal_log_data, 
        log_dict, log_patterns, event_list, tf_idf, num_all_docs, num_all_log, event_pred_model)
    plt.scatter([x for x in range(len(abnormal_score_list))], abnormal_score_list, s=0.1)
    plt.savefig('../results/normal_data_abnormal_score.png')
    plt.clf()
    plt.scatter([x for x in range(len(occurrence_porb_list))], occurrence_porb_list, s=0.1)
    plt.savefig('../results/normal_data_occurrence_probability.png')
    plt.clf()
    plt.scatter([x for x in range(len(tf_idf_list))], tf_idf_list, s=0.1)
    plt.savefig('../results/normal_data_tf_idf.png')
    print('plot saved')
    plt.clf()

    occurrence_porb_list, repeat_rate_list, abnormal_score_list, tf_idf_list = calculate_abnormal_score_for_files(abnormal_log_data, 
        log_dict, log_patterns, event_list, tf_idf, num_all_docs, num_all_log, event_pred_model)
    plt.scatter([x for x in range(len(abnormal_score_list))], abnormal_score_list, s=0.1)
    plt.savefig('../results/abnormal_data_abnormal_score.png')
    plt.clf()
    plt.scatter([x for x in range(len(occurrence_porb_list))], occurrence_porb_list, s=0.1)
    plt.savefig('../results/abnormal_data_occurrence_probability.png')
    plt.clf()
    plt.scatter([x for x in range(len(tf_idf_list))], tf_idf_list, s=0.1)
    plt.savefig('../results/abnormal_data_tf_idf.png')
    plt.clf()
    print('plot saved')'''

    #Test f1 score with AB score only
    if test_ab_score:
        # Test Repeat based
        ori_event_list=[x[:] for x in event_list]
        '''f1_data={}
        repeat_index=[1, 2, 3, 5,7,10]
        for i in [12,11, 10, 9,8,7,6,5]:
            print(f'single threshold is {i}')
            tmp_f1=[]
            for j in repeat_index:
                threshold=[i,j]
                tp, fp, fn, tn = 0, 0, 0, 0
                for single_file_data in normal_log_data:
                    #(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
                    event_list=[x[:] for x in ori_event_list]
                    if anomaly_detection_for_file(single_file_data, log_dict, log_patterns[:],event_list[:],
                                                tf_idf[:], num_all_docs, num_all_log, event_pred_model, synant_dict, threshold=threshold):
                        fp+=1
                    else:
                        tn+=1
                for single_file_data in abnormal_log_data:
                    #(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
                    event_list=[x[:] for x in ori_event_list]
                    if anomaly_detection_for_file(single_file_data, log_dict, log_patterns[:],event_list[:],
                                                tf_idf[:], num_all_docs, num_all_log, event_pred_model, synant_dict, threshold=threshold):
                        tp+=1
                    else:
                        fn+=1
                prec=tp/(tp+fp)
                rec=tp/(tp+fn)
                print(f'thre: {j}, acc:{(tp+tn)/(tp+tn+fp+fn)}, f1: {2*prec*rec/(prec+rec)}, prec: {prec}, rec: {rec} ')
                tmp_f1.append(2*prec*rec/(prec+rec))
            f1_data[i]=tmp_f1
        df=pd.DataFrame(f1_data, index=repeat_index)
        df.columns.name='Value threshold'
        df=df.rename_axis('Repeat threshold')
        sns_fig = sns.heatmap(df, annot=True, cmap='Blues')
        fig=sns_fig.get_figure()
        fig.savefig('../results/ABonly_repeatbase_heatmap.png')
        plt.clf()'''
        
        # Test Time based
        f1_data={}
        time_index=[0.5, 1, 3, 5,10,20]
        for i in [12, 11, 10, 9,8,7,6,5,4]:
            print(f'single threshold is {i}')
            tmp_f1=[]
            for j in time_index:
                threshold=[i,j]
                tp, fp, fn, tn = 0, 0, 0, 0
                for single_file_data in normal_log_data:
                    #(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
                    event_list=[x[:] for x in ori_event_list]
                    if anomaly_detection_for_file(single_file_data, log_dict, log_patterns[:],event_list[:],tf_idf[:], 
                                                num_all_docs, num_all_log, event_pred_model, synant_dict, threshold=threshold, way='time'):
                        fp+=1
                    else:
                        tn+=1
                for single_file_data in abnormal_log_data:
                    #(log_dict, synant_dict, log_patterns,event_list),(Q, sigma, delta, initialState, F_) = data
                    event_list=[x[:] for x in ori_event_list]
                    if anomaly_detection_for_file(single_file_data, log_dict, log_patterns[:],event_list[:],tf_idf[:], 
                                                num_all_docs, num_all_log, event_pred_model,synant_dict, threshold=threshold, way='time'):
                        tp+=1
                    else:
                        fn+=1
                prec=tp/(tp+fp)
                rec=tp/(tp+fn)
                print(f'thre: {j}, acc:{(tp+tn)/(tp+tn+fp+fn)}, f1: {2*prec*rec/(prec+rec)}, prec: {prec}, rec: {rec} ')
                tmp_f1.append(2*prec*rec/(prec+rec))
            f1_data[i]=tmp_f1
        df=pd.DataFrame(f1_data, index=time_index)
        df.columns.name='Value threshold'
        df=df.rename_axis('Time threshold')
        sns_fig = sns.heatmap(df, annot=True, cmap='Blues')
        fig=sns_fig.get_figure()
        fig.savefig('../results/ABonly_timebase_heatmap.png')

    AD(normal_data, abnormal_data, log_dict, log_patterns, event_list, tf_idf, num_all_docs, num_all_log, event_pred_model, synant_dict)

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
