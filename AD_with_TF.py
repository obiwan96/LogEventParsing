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
        del json_data['time']
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
            tmp_data=pd.Series(json_read_content(file_path+'/'+switch_type+'/'+file, switch_type), name=file_date)
            find_file=True
            data=(tmp_data if data.empty else pd.concat([data, tmp_data], axis=1))
    #print(data.index.tolist())
    return data.transpose(), find_file

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
    whole_netconf_features=[]
    for date in date_list:
        # read log data
        tmp_data={'log':[], 'netconf':[]}
        all_log=[]
        for file in log_file_list:
            if os.path.exists(log_file_path+'/'+file+'/'+date):
                if file =='1st':
                    dir_list_=os.listdir(log_file_path+'/'+file+'/'+date)
                    for dir_name in dir_list_:
                        file_list_=os.listdir(log_file_path+'/'+file+'/'+date+'/'+dir_name)
                        for file_name_ in file_list_:
                            log_=read_file(log_file_path+'/'+file+'/'+date+'/'+dir_name+'/'+file_name_)
                            all_log.extend(log_)
                file_list_=os.listdir(log_file_path+'/'+file+'/'+date)
                for file_name_ in file_list_:
                    log_=read_file(log_file_path+'/'+file+'/'+date+'/'+file_name_)
                    all_log.extend(log_)
            all_log=sorted(all_log, key=lambda x:x['date'])
            tmp_data['log']=all_log
            #print(all_log)
        
        # read netconf data
        netconf_data=pd.DataFrame()
        date_month=int(date[:2])
        date_day=int(date[3:])

        # Cisco
        tmp_netconf_data, find_file_1=read_netconf_data(netconf_file_path, (date_month, date_day), 'cisco')
        netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=1)
        # Juniper
        tmp_netconf_data, find_file_2=read_netconf_data(netconf_file_path, (date_month, date_day), 'juniper')
        netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=1)
        if not find_file_1 and not find_file_2:
            tmp_netconf_data, _ =read_netconf_data(netconf_data_path, (date_month, date_day), 'cisco')
            netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=1)
            tmp_netconf_data, _ =read_netconf_data(netconf_data_path, (date_month, date_day), 'juniper')
            netconf_data=pd.concat([netconf_data, tmp_netconf_data], axis=1)
        #print(netconf_data.columns.tolist())
        tmp_data['netconf']=netconf_data
        data[date]=tmp_data
        whole_netconf_features.extend(netconf_data.columns.tolist())
    print(f'{len(date_list)} num of date readed')
    return data, list(set(whole_netconf_features))

if __name__ == '__main__':
    loading_dict_data=True
    loading_model=True
    calculate_ab_score=False
    test_ab_score=False
    loading_AD_data=True

    model_path='../model/eventnum313_input10_acc93_transformer'
    normal_log_path='../normal_data'
    abnormal_log_path='../overloaded_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not loading_dict_data or not loading_model or not loading_AD_data or test_ab_score or calculate_ab_score: 
        print('rading log and NETCONF data')
        print('normal data read')
        normal_data, whole_netconf_features=read_data(normal_log_path)
        print('abnormal data read')
        abnormal_data, netconf_features =read_data(abnormal_log_path)
        whole_netconf_features.extend(netconf_features)
        whole_netconf_features=list(set(whole_netconf_features))
        print(f'whole netconf features are {len(whole_netconf_features)}')

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
    ori_event_list=[x[:] for x in event_list]

    # Load Event Prediction Model or Train the model
    # After training, calculate and plot the abnormal score for each log file
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
    
    if calculate_ab_score:
        #normal_log_data=[normal_data[date]['log'] for date in normal_data.keys()]
        #abnormal_log_data=[abnormal_data[date]['log'] for date in abnormal_data.keys()]
        
        occurrence_porb_list, repeat_rate_list, abnormal_score_list, tf_idf_list = calculate_abnormal_score_for_files(normal_data, 
            log_dict, log_patterns, event_list, tf_idf, num_all_docs, num_all_log, event_pred_model)
        plt.xlabel('Date')
        plt.ylabel('Values')
        x=[]
        y=[]
        for i, date_ in enumerate(normal_data.keys()):
            x.extend([i]*len(abnormal_score_list[date_]))
            y.extend(abnormal_score_list[date_])
        plt.scatter(x,y)
        #plt.scatter([x for x in range(len(abnormal_score_list))], abnormal_score_list, s=0.1)
        plt.savefig('../results/normal_data_abnormal_score.png')
        plt.clf()
        '''plt.scatter([x for x in range(len(occurrence_porb_list))], occurrence_porb_list, s=0.1)
        plt.savefig('../results/normal_data_occurrence_probability.png')
        plt.clf()
        plt.scatter([x for x in range(len(tf_idf_list))], tf_idf_list, s=0.1)
        plt.savefig('../results/normal_data_tf_idf.png')
        plt.clf()'''
        print('normal ab score plot saved')

        occurrence_porb_list, repeat_rate_list, abnormal_score_list, tf_idf_list = calculate_abnormal_score_for_files(abnormal_data, 
            log_dict, log_patterns, event_list, tf_idf, num_all_docs, num_all_log, event_pred_model)
        plt.xlabel('Date')
        plt.ylabel('Values')
        x=[]
        y=[]
        for i, date_ in enumerate(abnormal_data.keys()):
            x.extend([i]*len(abnormal_score_list[date_]))
            y.extend(abnormal_score_list[date_])
        plt.scatter(x,y)
        #plt.scatter([x for x in range(len(abnormal_score_list))], abnormal_score_list, s=0.1)
        plt.savefig('../results/abnormal_data_abnormal_score.png')
        plt.clf()
        '''plt.scatter([x for x in range(len(occurrence_porb_list))], occurrence_porb_list, s=0.1)
        plt.savefig('../results/abnormal_data_occurrence_probability.png')
        plt.clf()
        plt.scatter([x for x in range(len(tf_idf_list))], tf_idf_list, s=0.1)
        plt.savefig('../results/abnormal_data_tf_idf.png')
        plt.clf()'''
        print('abnormal ab score plot saved')

    #Test f1 score with AB score only
    if test_ab_score:
        normal_log_data=[normal_data[date]['log'] for date in normal_data.keys()]
        abnormal_log_data=[abnormal_data[date]['log'] for date in abnormal_data.keys()]
        # Test Repeat based
        print('Test Repeat based AB score Anomaly Detection')
        ori_event_list=[x[:] for x in event_list]
        f1_data={}
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
        plt.clf()
        
        # Test Time based
        print('Test Time based AB score Anomaly Detection')
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
        plt.clf()

    # Test RNN model with AB score and NETCONF
    if loading_AD_data:
        with open('AD_data.pkl', 'rb') as f:
            ad_data=pkl.load(f)
        x_data, y_data = ad_data
        x_normal=pd.concat(([x_data[i] for i in range(len(y_data)) if not y_data[i]]))
        input_feature_num=len(ad_data[0][0].columns)
        #print(input_feature_num)
    else:
        event_list=[x[:] for x in ori_event_list]
        ad_data = Make_data_for_AD(normal_data, abnormal_data, log_dict, log_patterns, event_list, 
            tf_idf, num_all_docs, num_all_log, event_pred_model, whole_netconf_features, synant_dict)
        with open('AD_data.pkl','wb') as f:
            pkl.dump(ad_data,f)
        print('AD data saved')
    dataloader = seperate_data(ad_data, input_feature_num, batch_size=40)

    losses_trans, f1_trans = AD(dataloader, input_feature_num, 'transformer', lr=0.0001)
    losses_simple, f1_simple = AD(dataloader, input_feature_num, 'simple')
    losses_lstm, f1_lstm = AD(dataloader, input_feature_num, 'lstm', lr=0.007)
    losses_heavy, f1_heavy = AD(dataloader, input_feature_num, 'heavy' )

    # plot losses graph for four model
    epoches=[i*50 for i in range(800/50)]
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoches')
    ax.set_ylabel('Losses')
    plt.plot(epoches, losses_simple, label='simple_RNN')
    plt.plot(epoches, losses_heavy, label='heavy_RNN')
    plt.plot(epoches, losses_lstm, label='LSTM')
    plt.plot(epoches, losses_trans, label='Transformer')
    plt.legend()
    plt.savefig('../results/AD_losses_compare.png')
    plt.clf()

    # plot f1 score graph for four model
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoches')
    ax.set_ylabel('F1 Score')
    plt.plot(epoches, f1_simple, label='simple_RNN')
    plt.plot(epoches, f1_heavy, label='heavy_RNN')
    plt.plot(epoches, f1_lstm, label='LSTM')
    plt.plot(epoches, f1_trans, label='Transformer')
    plt.legend()
    plt.savefig('../results/AD_f1s_compare.png')
    plt.clf()
                                   
                                   

