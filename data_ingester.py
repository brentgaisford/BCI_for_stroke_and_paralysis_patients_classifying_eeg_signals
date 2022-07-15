from scipy.io import loadmat
import pandas as pd
import mne
import numpy as np

#Create function to ingest subject data
#Returns session 1 data, session 1 trial times, session 1 task list per trial,
#Session 2 data, session 2 trial times, session 2 task list per trial
def data_ingester(filename):
    '''This function ingests all the data from one of the nine data files that came in this dataset.\n
    Each data file contains all the data for one of the nine subjects.\n
    For the selected data file, it returns six objects - the eeg data, the sample numbers at which each trial began, \n
    and what task was performed in each trial. It returns those three objects for session 1 followed by session 2.\n
    Note this function is run automatically by this script, and a dictionary of data, trial times, and trial types\n
    are created.'''
    #open file in scipy
    annots = loadmat(filename)
    #dig into data to get to data for each session
    [data] = annots['data']
    [sesh_1] = data[0]
    [sesh_2] = data[1]
    #Creating dataframes of each sessions data
    sesh_1 = pd.DataFrame(sesh_1)
    sesh_2 = pd.DataFrame(sesh_2)

    #save eeg recording data from session 1
    [a] = sesh_1['X'] 
    #save the sample numbers at which each trial began in session 1
    [b] = sesh_1['trial']
    #save the list of what task was performed in each trial in session 1
    [c] = sesh_1['y']

    #save the raw eeg recording data from session 2
    [d] = sesh_2['X']
    #save the sample numbers at which each trial began in session 2
    [e] = sesh_2['trial']
    #save the list of what task was performed in each trial in session 2
    [f] = sesh_2['y']
    
    #return all needed data
    return a, b, c, d, e, f

#Create dictionaries to populate with our data
data_dict = {}
trial_times_dict = {}
y_dict = {}

#extract data for subject A
(data_dict['sub_A_sesh_1'], 
 trial_times_dict['sub_A_sesh_1'], 
 y_dict['sub_A_sesh_1'], 
 data_dict['sub_A_sesh_2'], 
 trial_times_dict['sub_A_sesh_2'], 
 y_dict['sub_A_sesh_2']) = data_ingester('data/A.mat')

 #extract data for subject C
(data_dict['sub_C_sesh_1'], 
 trial_times_dict['sub_C_sesh_1'], 
 y_dict['sub_C_sesh_1'], 
 data_dict['sub_C_sesh_2'], 
 trial_times_dict['sub_C_sesh_2'], 
 y_dict['sub_C_sesh_2']) = data_ingester('data/C.mat')

 #extract data for subject D
(data_dict['sub_D_sesh_1'], 
 trial_times_dict['sub_D_sesh_1'], 
 y_dict['sub_D_sesh_1'], 
 data_dict['sub_D_sesh_2'], 
 trial_times_dict['sub_D_sesh_2'], 
 y_dict['sub_D_sesh_2']) = data_ingester('data/D.mat')

 #extract data for subject E
(data_dict['sub_E_sesh_1'], 
 trial_times_dict['sub_E_sesh_1'], 
 y_dict['sub_E_sesh_1'], 
 data_dict['sub_E_sesh_2'], 
 trial_times_dict['sub_E_sesh_2'], 
 y_dict['sub_E_sesh_2']) = data_ingester('data/E.mat')

 #extract data for subject F
(data_dict['sub_F_sesh_1'], 
 trial_times_dict['sub_F_sesh_1'], 
 y_dict['sub_F_sesh_1'], 
 data_dict['sub_F_sesh_2'], 
 trial_times_dict['sub_F_sesh_2'], 
 y_dict['sub_F_sesh_2']) = data_ingester('data/F.mat')

 #extract data for subject G
(data_dict['sub_G_sesh_1'], 
 trial_times_dict['sub_G_sesh_1'], 
 y_dict['sub_G_sesh_1'], 
 data_dict['sub_G_sesh_2'], 
 trial_times_dict['sub_G_sesh_2'], 
 y_dict['sub_G_sesh_2']) = data_ingester('data/G.mat')

 #extract data for subject H
(data_dict['sub_H_sesh_1'], 
 trial_times_dict['sub_H_sesh_1'], 
 y_dict['sub_H_sesh_1'], 
 data_dict['sub_H_sesh_2'], 
 trial_times_dict['sub_H_sesh_2'], 
 y_dict['sub_H_sesh_2']) = data_ingester('data/H.mat')

 #extract data for subject J
(data_dict['sub_J_sesh_1'], 
 trial_times_dict['sub_J_sesh_1'], 
 y_dict['sub_J_sesh_1'], 
 data_dict['sub_J_sesh_2'], 
 trial_times_dict['sub_J_sesh_2'], 
 y_dict['sub_J_sesh_2']) = data_ingester('data/J.mat')

 #extract data for subject L
(data_dict['sub_L_sesh_1'], 
 trial_times_dict['sub_L_sesh_1'], 
 y_dict['sub_L_sesh_1'], 
 data_dict['sub_L_sesh_2'], 
 trial_times_dict['sub_L_sesh_2'], 
 y_dict['sub_L_sesh_2']) = data_ingester('data/L.mat')


#Create info file used to create MNE raw objects
#channels used from study documentation
ch_names = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T3', 
            'C3', 'Cz', 'C4', 'T4', 'CP3', 'CPz', 'CP4', 'P7', 'P5', 'P3', 
            'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']
#per the study documentation data was collected at 256 hertz
sfreq = 256
#all channel types are eeg sensors
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

#Create event dictionary
events_explained = {'brainteaser/word': 1, 'brainteaser/subtraction': 2, 
               'brainteaser/navigation': 3, 'motor/hand': 4, 'motor/feet': 5}

#Create event dictionary for later use in creating epochs objects
#event array combines stimulus times (3 sec after start of trial),
#A middle column of zeros since our data is digital,
#And a final column to indicate which kind of stimulus occurred
event_dict = {}
for key, value in trial_times_dict.items():
    event_dict[key] = np.concatenate((np.reshape(value.flatten() + (256 * 3), 
                                                 (len(value), 1)),
                                      np.zeros((y_dict[key].shape[0], 1)),
                                      y_dict[key]), axis=1).astype(int)


#Delete trial times dictionary, done with it
del trial_times_dict