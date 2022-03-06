from cmath import pi
from scipy.fft import fft, fftfreq, ifft
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt 

#---------------------------------------------------------------------------------
# Function : Filter_creating()
# Description : Creating the desired boardband filter 
#---------------------------------------------------------------------------------
def Filter_creating(Len = 512, low_cut_normal_fre = 200, high_cut_normal_fre=7800, fs=16000):
    """_summary_

    Args:
        Len (int, optional): The length of the control filter. Defaults to 512.
        low_cut_normal_fre (float, optional): The low cut-off normalized frequency. Defaults to 0.3.
        high_cut_normal_fre (float, optional): The high cut-off normalized frequency. Defaults to 0.8.

    Returns:
        float: The coefficients of the designed filter.
    """
    b1     = signal.firwin(Len, [low_cut_normal_fre, high_cut_normal_fre], pass_zero='bandpass', window ='hamming',fs=fs)
    
    w1, h1 = signal.freqz(b1)
    
    plt.title('Digital filter frequnecy response')
    plt.plot(w1*fs/(2*pi), 20*np.log10(np.abs(h1)),'b')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()
    
    return b1 

#---------------------------------------------------------------------------------
# Function: drawing_spectrum_via_fft()
#---------------------------------------------------------------------------------
def drawing_spectrum_via_fft(filter_response,fs=160000):
    N = len(filter_response)
    T = 1/fs 
    Fre_response = fft(filter_response)
    # print(Fre_response[0] + Fre_response[0])
    xf = fftfreq(N,T)
    #print(xf)
    yf =np.abs(Fre_response)**2
    plt.plot(yf)
    plt.grid()
    plt.show()

def drawing_group_filter_frequency_timedomian_response(sub_filters,fs=16000):
    N_filter = sub_filters.shape[0]
    filter_total = np.sum(sub_filters,axis=0)
    
    
    for i in range(N_filter):
        plt.subplot(N_filter+1,2,2*i+1)
        plt.plot(sub_filters[i,:])
        plt.title(f'Impulse response of the {i}th control filter')
        plt.grid()
        plt.xlabel('Taps')
        
        w1, h1 = signal.freqz(sub_filters[i,:])
        plt.subplot(N_filter+1,2,2*(i+1))
        plt.plot(w1*fs/(2*pi), np.abs(h1)**2,'b')
        plt.title(f'Digital filter frequnecy response of the {i}th control filter')
        plt.grid()
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude Response (dB)')
    
    plt.subplot(N_filter+1,2,2*(N_filter+1)-1)
    plt.plot(filter_total)
    plt.title(f'Impulse response of the main control filter')
    plt.grid()
    plt.xlabel('Taps')
        
    w1, h1 = signal.freqz(filter_total)
    plt.subplot(N_filter+1,2,2*(N_filter+1))
    plt.plot(w1*fs/(2*pi), np.abs(h1)**2,'b')
    plt.title(f'Digital filter frequnecy response of the main control filter')
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude Response (dB)')
    
    # plt.grid()
    plt.show()

#---------------------------------------------------------------------------------
# Function: Perfect_filter_decompose()
#---------------------------------------------------------------------------------
def Perfect_filter_decompose(filter,Num_cmb,fs=16000):
    """_summary_

    Args:
        filter (float64): The impulse resoponse of the control filter.
        Num_cmb (int): The number of the sub filter.
        fs (int, optional): The system sampling rate. Defaults to 16000.
    
    Returns:
        float64: The control filter's group.
    """
    N = len(filter)
    sub_filters = np.zeros((Num_cmb, N))
    sub_num     = N//(Num_cmb*2)
    Fre_filter  = fft(filter) 
    
    for ii in range(Num_cmb):
        Temper_spectrum = np.zeros_like(Fre_filter)
        start_index   = ii*sub_num+1
        end_index     = (ii+1)*sub_num+1
        start_n_index = -(ii+1)*sub_num 
        end_n_index   = -start_index +1 
        
        if ii != Num_cmb-1:
            Temper_spectrum[start_index:end_index] = Fre_filter[start_index:end_index]
            if end_n_index == 0 :
                Temper_spectrum[start_n_index:] = Fre_filter[start_n_index:]
                # print(Fre_filter[start_index:end_index]-np.flip(np.conj(Fre_filter[start_n_index:])))
            else:
                Temper_spectrum[start_n_index:end_n_index] = Fre_filter[start_n_index:end_n_index]
                # print(Fre_filter[start_index:end_index]-np.flip(np.conj(Fre_filter[start_n_index:end_n_index])))
            
        else:
            Temper_spectrum[start_index:end_n_index] = Fre_filter[start_index:end_n_index] 
        
        sub_filters[ii,:] = ifft(Temper_spectrum)
    return sub_filters

if __name__=="__main__":
    b1 = Filter_creating()
    print(f'The lenght of the contorl filter is {b1.shape[0]}')
    b_frequency = fft(b1)
    print(b_frequency.dtype)
    drawing_spectrum_via_fft(b1,fs=160000)
    sub_filters = Perfect_filter_decompose(b1,6)
    print(sub_filters.shape)
    # print(sub_filters[1,1].dtype)
    
    drawing_group_filter_frequency_timedomian_response(sub_filters,fs=160000)
    
    
    # c =np.array([1,2,4,5,6,7,8])
    # print(c[1:3])
    # print(np.flip(c)[1:3])
    print(b1.shape)
    print(np.sum(sub_filters,axis=0).shape)
    c = np.sum(sub_filters,axis=0) - b1 
    print(c[1:].max())
    x_inex = range(512)
    plt.subplot(3,1,1)
    plt.plot(x_inex,b1)
    plt.grid()
    plt.subplot(3,1,2)
    plt.plot(np.sum(sub_filters,axis=0))
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(c)
    plt.grid()
    plt.show()