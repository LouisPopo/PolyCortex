import numpy as np
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
    'Sleep stage 4' : 4,
    'Sleep stage R' : 5,
    'Sleep stage ?' : 6
}

for no_subject in subjects:
    # Returns RawGDF and RawEDF
    raw_data = mne.io.read_raw_edf(files[no_subject][0], preload=True)
    # Returns a mne.Annotations object
    annot_data = mne.read_annotations(files[no_subject][1])

    raw_data.set_annotations(annot_data, emit_warning=False)
    raw_data.set_channel_types(mapping)

    subject_data = np.array(raw_data._data).transpose()

    nb_entries = subject_data.shape[0]

    classifications = ['Sleep stage W']*nb_entries
    
    for i, sleep_stage in enumerate(annot_data.description):
        onset_ms = int(annot_data.onset[i] / 0.01)
        duration_ms = int(annot_data.duration[i] / 0.01)
        if (sleep_stage != 'Sleep stage ?' and sleep_stage != 'Sleep stage W'):
            # print(sleep_stage, 'started at ', onset_ms, 'ms and lasted for ', duration_ms, 'ms')
            for j in range(onset_ms, onset_ms + duration_ms):
                classifications[j] = sleep_stage
    
    classifications = np.array(classifications)
    result = np.c_[subject_data, classifications]
    
    file_name = 'subject' + str(no_subject) + '.csv'
    np.savetxt(file_name,result, delimiter=',', header='string', comments='', fmt='%s')
    x = 0
     
    # needs to get raw_data._data => (7,7 950 000) array each channel, each time instant
    # annot_data._description => (1,154) array with all sleep cycles in order
    #           ._duration => time of each sleep cycle
    #           ._onset => start time of sleep cycle
