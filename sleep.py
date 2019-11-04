import numpy as np
import pandas as pd
import random
# import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


TIME_OF_W_TO_KEEP_SEC = 1800


nb_subjects = 1 
# CAN BE UP TO 20
nights = [1] 
# [1,2]

subjects = range(0,nb_subjects) 

# Get path to local copies of data from Physionet files are PSG.edf and Hypnogram.edf
files = fetch_data(subjects=subjects, recording=nights)

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

sleep_stages = {
    'Sleep stage W' : 0,
    'Sleep stage 1' : 1,
    'Sleep stage 2' : 2,
    'Sleep stage 3' : 3,
    'Sleep stage 4' : 3,
    'Sleep stage R' : 4,
    'Sleep stage ?' : 5
}

for no_subject in subjects:
    # Returns RawGDF and RawEDF
    raw_data = mne.io.read_raw_edf(files[no_subject][0], preload=True)
    # Returns a mne.Annotations object
    annot_data = mne.read_annotations(files[no_subject][1])

    raw_data.set_annotations(annot_data, emit_warning=False)
    raw_data.set_channel_types(mapping)

    df = pd.DataFrame(np.array(raw_data._data).transpose())

    # # keep only the EEG signals data
    subject_data = df.drop([2,3,4,5,6],axis=1)

    nb_entries = subject_data.shape[0]

    classifications = ['Sleep stage W']*nb_entries
    
    # add the classification for every row
    for i, sleep_stage in enumerate(annot_data.description):
        onset_ms = int(annot_data.onset[i] / 0.01)
        duration_ms = int(annot_data.duration[i] / 0.01)
        if (sleep_stage != 'Sleep stage ?' and sleep_stage != 'Sleep stage W'):
            # print(sleep_stage, 'started at ', onset_ms, 'ms and lasted for ', duration_ms, 'ms')
            for j in range(onset_ms, onset_ms + duration_ms):
                classifications[j] = sleep_stage

    print('added classifications.')
    subject_data.insert(2, 'Class', classifications)

    #30 MINS BEFORE SLEEP AND 30 MINS AFTER
    i_start = (int(annot_data.duration[0]) - TIME_OF_W_TO_KEEP_SEC) * 100

    i_end = (int(annot_data.duration[len(annot_data.duration) - 2]) + TIME_OF_W_TO_KEEP_SEC) * 100


    all_seq = []

    for i in range(0,32):
        start_i = random.randrange(i_start, i_end - 24000)

        seq = subject_data[start_i:start_i + 24000, :]

        all_seq.append(seq)

    np.savetxt(file_name, all_seq, delimiter=',', header='EEG1,EEG2,Class', comments='', fmt='%s')





    

    # # Get how long his first W stage is
    # awake_time = int(annot_data.duration[0])
    # nb_of_rows = awake_time * 100
    # # keep only one hour of wake stage
    # subject_data.drop(subject_data.index[:(nb_of_rows - TIME_OF_W_TO_KEEP_SEC*100)], inplace=True)

    # # Get how long last W stage is 
    # final_wake_time = int(annot_data.duration[len(annot_data.duration) - 2])
    # nb_of_rows = final_wake_time * 100
    # # Keep only one hour of wake time after 'waking up'
    # subject_data.drop(subject_data.tail(nb_of_rows - TIME_OF_W_TO_KEEP_SEC*100).index, inplace=True)

    # # Convert Sleep Stage X to corresponding integer
    # subject_data['Class'].replace(sleep_stages, inplace=True)

    # file_name = 'subject' + str(no_subject) + '.csv'
    # np.savetxt(file_name, subject_data, delimiter=',', header='EEG1,EEG2,Class', comments='', fmt='%s')
     
    # # needs to get raw_data._data => (7,7 950 000) array each channel, each time instant
    # # annot_data._description => (1,154) array with all sleep cycles in order
    # #           ._duration => time of each sleep cycle
    # #           ._onset => start time of sleep cycle
