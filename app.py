import os
import numpy as np
import pandas as pd
import streamlit as st
import mne
import scipy.io

import picard
import pickle
import random

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from sklearn.linear_model import LogisticRegression,RidgeCV
from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler
from mne_features.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

import streamlit as st      

st.header('EEG Tweaking Labs 🎲')

#---------------------------------- Side Bar ---------------------------------------------


with st.sidebar:
    st.header("EEG Tweaking Labs")
    page=st.radio('Select one:', ["Welcome","Preparing the Data", "Synthetic Data","About"])
    atrpx=st.file_uploader('Upload Data')
    
    # ate=st.file_uploader('Upload Test Data')


#---------------------------------- P300 ---------------------------------------------
if page == "Welcome":

    st.write("""Non-invasive electroencephalogram (EEG)-based brain-computer interfaces (BCI) 
    can be characterized by the technique used to measure brain activity and by the way  
    that different brain signals are translated into commands that control an effector 
    (e.g., controlling a computer cursor for word processing and accessing the internet)""")

if page == "Preparing the Data":
    atr=scipy.io.loadmat("Subject_A_Train.mat")
    bandpassflag=st.checkbox("Bandpass Filter Data")
    icaflag=st.checkbox("Apply ICA to denoise data")
    baselineepochflag=st.checkbox("Baseline epochs")
    
    chvalues = st.slider('Select number of channels',0, 64, value=64)  
    st.write(chvalues," channels selected") 

    runp300=st.button("Run Program")
    if runp300:
        st.write("converting data...")
        signal=atr['Signal']
        tarchar=atr['TargetChar'][0]
        flashing=atr['Flashing']
        stimcode=atr['StimulusCode']
        stimtype=atr['StimulusType']
        st.write("creating events...")
        for char in range(85):
            for i in range(len(flashing[0])):
                if stimcode[char,i]!=0:
                    if stimtype[char,i]==1:
                        stimtype[char,i]=2
                else:
                    stimtype[char,i]=1

        st.write("merging data...")
        signal_m=signal[0]
        stimcode_m=stimcode[0]
        stimtype_m=stimtype[0]
        for i in range(1,85):
            signal_m=np.concatenate((signal_m,signal[i]))
            stimcode_m=np.concatenate((stimcode_m,stimcode[i]))
            stimtype_m=np.concatenate((stimtype_m,stimtype[i]))

        st.write("creating info structure...")
        channel = {1:'FC5',2: 'FC3',3:'FC1',4:'FCz',5:'FC2',6:'FC4',7:'FC6',8:'C5',9:'C3',10:'C1',11:'Cz',12:'C2',13:'C4',14:'C6',15:'CP5',16:'CP3',17:'CP1',18:'CPz',19:'CP2',20:'CP4',21:'CP6',22:'Fp1',23:'Fpz',24:'Fp2',25:'AF7',26:'AF3',27:'AFz',28:'AF4',29:'AF8',30:'F7',31:'F5',32:'F3',33:'F1',34:'Fz',35:'F2',36:'F4',37:'F6',38:'F8',39:'FT7',40:'FT8',41:'T7',42:'T8',43:'P9',44:'P10',45:'TP7',46:'TP8',47:'P7',48:'P5',49:'P3',50:'P1',51:'Pz',52:'P2',53:'P4',54:'P6',55:'P8',56:'PO7',57:'PO3',58:'POz',59:'PO4',60:'PO8',61:'O1',62:'Oz',63:'O2',64:'Iz'}
        cn = [*channel.values()]
        #channel types
        ch_types= ['eeg'] * signal_m.shape[1]
        
        stimcoden= np.expand_dims(stimcode_m,axis=1)
        stimtypen = np.expand_dims(stimtype_m,axis=1)
        signal_new = np.append(signal_m,stimcoden,axis=1)
        signal_new = np.append(signal_new,stimtypen,axis=1)

        ch_names_events = cn + ['stimcode']+ ['stimtype']
        ch_types_events = ch_types + ['misc'] + ['misc']

        info_events = mne.create_info(ch_names_events,240, ch_types_events)
        st.write("creating mne compatible array...")
        raw_arr = mne.io.RawArray(signal_new.T, info_events)
        if bandpassflag:
            st.write("bandpass filtering data...")
        raw_arr_f=raw_arr.copy().filter(1,20)
        st.write("creating channel montage...")
        raw_arr_f.set_montage('biosemi64',on_missing='warn')

        if icaflag:
            st.write("applying ICA algorithm...")

        st.write("finding events in signal...")
        events_st = mne.find_events(raw_arr_f, stim_channel='stimtype', initial_event=True)
        
        tmin1 = 0
        tmax1 = 0.7
        if baselineepochflag:
            st.write("creating epochs with baseline...")
            bline=(tmin1,tmax1)
        else:
            st.write("creating epochs...")
            bline=(None)

        bline=(tmin1,tmax1)
        epochs = mne.Epochs(raw_arr_f, events_st, { 'n300':1, 'p300':2 }, tmin1, tmax1, baseline=bline)
        
        st.write("splitting epochs...")      
        p300ss = epochs['p300']
        n300ss = epochs['n300']
        p300s = epochs['p300'].average()
        n300s = epochs['n300'].average()
        with open('p300.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(p300ss.get_data(), file)
        
        l=[]
        for i in range(64):
            l.append(i)

        roi_dict = dict(p300=l)
        roi_evoked = mne.channels.combine_channels(p300s, roi_dict, method='mean')
        print(roi_evoked.info['ch_names'])
        # roi_evoked.plot()


        st.write("done")
        
#---------------------------------- Synthetic Data ---------------------------------------------


elif page== "Synthetic Data":
    st.subheader("Generate Synthetic EEG Data")
    
    st.selectbox('Select a GAN Model', ['CTGAN', 'TimeGAN','VAE'])
    st.selectbox('Select what data to generate', ['p300','n300'])
    with open('p300.pkl', 'rb') as file:
        # A new file will be created
        p300ss=pickle.load(file)
    p300ss=np.moveaxis(p300ss,1,2)
    p300ss2=np.mean(p300ss[:,:,:64],axis=2,keepdims=False)
    df = pd.DataFrame(p300ss2, columns = [i for i in range(169)])
    df.drop([df.index[1], df.index[100]])
    df.to_csv('p300s.csv',index=False)
    st.write(df.shape)

    if st.button("Loaded Data"):
            if os.path.exists("file1.csv") == True:
                st.write("Dataset Loaded")
                loadddata = pd.read_csv(
                    "file1.csv")
                st.dataframe(loadddata)
                st.write(loadddata.shape)
            else:
                st.write("No Dataset")
    
    st.selectbox('Channels', ['1', '4','all'])

    st.button("Generate")
    with st.expander("About Gretel"):
            st.image(
                "https://uploads-ssl.webflow.com/5ea8b9202fac2ea6211667a4/5eb59ce904449bf35dded1ab_gretel_wordmark_gradient.svg")

            st.write("""Generate synthetic data to augment your datasets.
                This can help you create AI and ML models that perform
                and generalize better, while reducing algorithmic bias.""")

            st.write("""No need to snapshot production databases to share with your team.
                Define transformations to your data with software,
                and invite team members to subscribe to data feeds in real-time""")
    



    with st.expander("Analysis"):
        st.write("Accuracy: 60%")            
    st.number_input('Generate Records', 0, 1000)
    st.button('Download file')

    if st.button("Show  Data"):
            if os.path.exists("syndata.csv") == True:
                st.write("Dataset Generated")
                syntheddata = pd.read_csv(
                    "syndata.csv")
                st.dataframe(syntheddata)
                st.write(syntheddata.shape)
            else:
                st.write("Dataset not yet Generated")
    
    if st.button("Generate Plots"):
                st.write("Graphs")
                with st.expander("Show"):
                    st.image("1.png")
                    st.image("2.png")
#---------------------------------- About ---------------------------------------------

elif page== "About":
    st.write("Made by People")
    st.subheader('PPT, Paper at')
    st.button("Click!")
