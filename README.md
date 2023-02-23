Log Event Parser for Anomaly Detection
===
This is a part of DPNM-SNIC project for log anaomaly detection.

We will use log event parser to detect anomaly logs in logs.

     - log_parser_lib.py 
        : have functions for parsing
     - check_sent_sim.py
        : check sentence similarity of parsed event form based on Sentence BERT
     - event_FSA.py
        : Build dictionary, event list and FSA. Save in 'data.pkl'
     - put_new_log_to_fsa.py
        : check new log exist in DB and log event transition is valid with FSA
     - read_and_save_log.py
        : read logs and combine logs to 'all.log'
    
ex)

```shell
p4ml> python3 check_sent_sim.py

Let's calculate the similarity based on Sent-BERT
object sentence : authentication from for admin success

## Based on cosine similarity
authentication from for admin success : 1.0
pr system warm reboot : 0.9776268601417542
login on pts/ from as admin ssh : 0.9769018888473511
smtl initialization : 0.9761740565299988
can t read snmp encrypted community : 0.9740574955940247
process nbr on tengi/ from full to down neighbor down inactivitytimer : 0.9739202260971069
new session of user root : 0.9738637804985046
[path] schedule timeout x/xc : 0.9736156463623047
[path] wake up x/x : 0.9724854230880737
echo [date] task timeout secs [UNK] message : 0.9714299440383911

## Based on euclidian similarity
authentication from for admin success : 2.7712725568562746e-05
pr system warm reboot : 3.415301561355591
login on pts/ from as admin ssh : 3.489677906036377
can t read snmp encrypted community : 3.672403335571289
new session of user root : 3.6902873516082764
[path] schedule timeout x/xc : 3.7656421661376953
smtl initialization : 3.7789106369018555
process nbr on tengi/ from full to down neighbor down inactivitytimer : 3.7853009700775146
[path] wake up x/x : 3.9372076988220215
imi error checking is ok : 3.9435954093933105
```

![FSA example](DFA_for_all.gv.png)