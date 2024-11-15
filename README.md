Log Event Parser for Anomaly Detection
===
Requirements (python3.7)
```shell
p4ml> pip install -r requirements.txt
p4ml> pip install torch torchvision torchaudio tensorboard transformers pytest pandas nltk
```

This is a part of DPNM-SNIC project for log anaomaly detection.

We will use log event parser to detect anomaly logs in logs.

     - log_parser_lib.py 
        : have functions for parsing
     - check_sent_sim.py
        : check sentence similarity of parsed event form based on Sentence BERT. Just my curiosity. Not related to SNIC project
     - event_FSA.py
        : Build dictionary, event list and FSA. Save in 'data.pkl'
     - event_LSTM.py
        : Seperate train and test data, train the LSTM model. use PyTorch
     - test_LSTM.py
        : train the LSTM model for 500 epoch changing input window size
     - read_and_save_log.py
        : read logs and combine logs to 'all.log'
     - AD_module.py
        : AB based Anomaly Detection Module
     - module_test.py
        : Test the AB based Anomaly Detection Module
     - LogNetconfAD_lib.py
        : Log and NETCONF based AD Module
     - AD_with_TF.py
        : Test the Log and NETCONF based AD Module
     - tf-idf ~ .ipynb
        : Log-TF-IDF based AD experiments file

    
ex)

```shell
p4ml> python3 AD_module.py 

>>> reading 02_21_overloaded.log file... <<<
-----------------------------------------
transition from q11 to q134 does not exist in FSA!
Prev: Feb 21 00:04:22PVIDB: Attribute 'cosd.support_xellent_qfx_feature' not present in Db
Now: Feb 21 00:04:22feature cos_fc_defaults num elems 4 rc 0
-----------------------------------------
transition from q134 to q11 does not exist in FSA!
Prev: Feb 21 00:04:22feature cos_fc_defaults num elems 4 rc 0
Now: Feb 21 00:04:22PVIDB: Attribute 'cosd.support_xellent_qfx_feature' not present in Db
----------------------------------
LSTM model detects abnormal log continuosly :
Feb 21 05:28:03UI_CHILD_STATUS: Cleanup child '/usr/bin/netstat', PID 2937, status 0
```
next image is example of our FSA.

![FSA example](./DFA/DFA_for_all.gv.png)
