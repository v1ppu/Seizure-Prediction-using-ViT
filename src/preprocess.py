import mne
import numpy as np
import os
import re
import torch
import datetime
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import mode_dot

# 30 second chunks per segemnt with 15 second overlap between adjacent segments

def load_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    if(raw.info['sfreq'] != 256):
        raw.resample(256)
    return raw

def filter_eeg(raw, l_freq=0.5, h_freq=40.0):
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

def segment_eeg(raw, segment_length=30, stepsize = 15, fs=256):
    data = raw.get_data()
    n_samples = int(segment_length * fs)
    n_samples_step = int(stepsize * fs)
    n_samples_total = data.shape[1]

    segments = []

    start = 0
    while start + n_samples <= n_samples_total:
        end = start + n_samples
        segment = data[:, start:end]
        segments.append(segment)
        start += n_samples_step

    return np.array(segments)

def label_segments(segments, file_path, summary_path, segment_length=30, stepsize=15, fs=256):
    n_segments = segments.shape[0]
    labels = np.zeros(n_segments)
    file_name = os.path.basename(file_path)
    #summary_path = 'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01-summary.txt'
    
    with open(summary_path, 'r') as f:
        summary_text = f.read()
    

    file_pattern = f"File Name: {file_name}\n.*?Number of Seizures in File: (\d+)"
    file_match = re.search(file_pattern, summary_text, re.DOTALL)
    
    if file_match:
        num_seizures = int(file_match.group(1))
        
        if num_seizures > 0:
            seizure_section = re.search(f"File Name: {file_name}.*?(Seizure Start Time.*?)(?:File Name:|$)", 
                                      summary_text, re.DOTALL)
            
            if seizure_section:
                seizure_info = seizure_section.group(1)
                
                #find seizure times
                start_times = re.findall(r"Seizure Start Time: (\d+) seconds", seizure_info)
                end_times = re.findall(r"Seizure End Time: (\d+) seconds", seizure_info)
                
                for start_time, end_time in zip(start_times, end_times):
                    start_sec = int(start_time)
                    end_sec = int(end_time)
                    
                    #make segments
                    start_segment = start_sec // stepsize
                    end_segment = (end_sec // stepsize) + 1
                    
                    
                    for i in range(start_segment, min(end_segment, n_segments)):
                        segment_start = i * stepsize
                        segment_end = segment_start + segment_length
                        if segment_end > start_sec and segment_start < end_sec:
                            labels[i] = 1
    
    return labels

