from msilib.schema import Error
import torch
import math
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt 

from DFT_filter_decompse import Filter_creating, Perfect_filter_decompose, drawing_group_filter_frequency_timedomian_response
from Adaptive_control_filter_generator_batch_normal import adaptive_control_filter_batch_normal, train_adaptive_gain_aglrithm_based_batch

#--------------------------------------------------------------------
# Function : Generating_broadband_noise()
#--------------------------------------------------------------------
def Generating_broadband_noise(low_cutoff_freq, high_cutoff_freq, T, fs=16000):
    """Generating a brodband noise as a given frequency band. 

    Args:
        low_cutoff_freq (int): the low end frequnecy of the broadband noise. 
        high_cutoff_freq (int): the high end frequency of the boadband noise.
        T (int): the simulation duration (seconds)
        fs (int, optional): _description_. Defaults to 16000.

    Returns:
        float64: the broadband noise.
    """
    Len_data = T*fs 
    Len      = 1024
    b1       = signal.firwin(Len, [low_cutoff_freq, high_cutoff_freq], pass_zero='bandpass', window ='hamming',fs=fs)
    xin      = np.random.randn(Len_data)
    y        = signal.lfilter(b1, 1, xin)
    return y[Len:] 

#--------------------------------------------------------------------
def Construt_filter_from_labels(sub_filters, pre_result=None):
    threshold    = 0.5 
    labels       = pre_result >= threshold 
    label_num    = np.expand_dims(labels,axis=0)
    novel_filter = np.matmul(label_num, sub_filters)

    pre_vector   = np.expand_dims(pre_result, axis=0)
    const_filter = np.matmul(pre_vector, sub_filters)

    return novel_filter, const_filter, label_num

def Noise_cancellor(Filter, Fx, Disturbance):
    y_anti_noise = signal.lfilter(Filter,1,Fx)
    error        = Disturbance-y_anti_noise
    
    return error 

#--------------------------------------------------------------------
# Function : Converse_from_numpyarray_to_tensor()
#--------------------------------------------------------------------
def Converse_from_numpyarray_to_tensor(xin):
    return torch.from_numpy(xin).type(torch.float)

def additional_noise(signal, snr_db):
    signal_power     = signal.norm(p=2)
    length           = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power      = additional_noise.norm(p=2)
    snr              = math.exp(snr_db / 10)
    scale            = snr * noise_power / signal_power
    noisy_signal     = signal + additional_noise/scale
    return noisy_signal

#---------------------------------------------------------------------
if __name__=="__main__":
    c_filter       = 20 
    fs             = 16000
    control_filter = Filter_creating(Len = 512, low_cut_normal_fre = 200, high_cut_normal_fre=7800, fs=fs)
    sub_filters    = Perfect_filter_decompose(control_filter,c_filter)
    sub_filters_T  = Converse_from_numpyarray_to_tensor(sub_filters)
    drawing_group_filter_frequency_timedomian_response(sub_filters=sub_filters, fs=fs)
    
    T                = 10
    low_cutoff_freq  = 1731  
    high_cutoff_freq = 7386
    
    Noise       = Generating_broadband_noise(low_cutoff_freq, high_cutoff_freq, T, fs)
    Disturbance = signal.lfilter(control_filter,1,Noise)
    Xin         = Converse_from_numpyarray_to_tensor(Noise)
    Xin         = additional_noise(Xin.unsqueeze(0),90)[0,:].unsqueeze(0)
    Dis         = Converse_from_numpyarray_to_tensor(Disturbance).unsqueeze(0)
    Xin2        = torch.cat((Xin,Xin,Xin,Xin,Xin),dim=0)
    Dis2        = torch.cat((Dis,Dis,Dis,Dis,Dis),dim=0)
    
    # Determind the processor: CUP or GPU 
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'<<===This program used {device}====>>')
    Generator = adaptive_control_filter_batch_normal(sub_filters_T,Batch_size=5, muw=0.01, device=device)
    error = train_adaptive_gain_aglrithm_based_batch(Generator,Xin2, Dis2, device=device)
    
    plt.plot(error)
    plt.title('The residual error of adaptive gain control')
    plt.grid()
    plt.show()