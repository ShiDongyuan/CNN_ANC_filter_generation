import torch 
import progressbar
import numpy       as np 
import torch.optim as optim 

#--------------------------------------------------------------------
# class : Adaptive_control_filter_generator()
#--------------------------------------------------------------------
class Adaptive_control_filter_generator():
    
    def __init__(self, Control_filter_groups):
        """Adaptive algorithm is used to generate the control filter from the pre-trained filters. 

        Args:
            Control_filter_groups (float32 tensor): The group of control filters [number of filter x length].
        """
        self.filter_number = Control_filter_groups.shape[0]
        self.W_gain        = torch.zeros(1, self.filter_number, requires_grad=True, dtype=torch.float)
        self.Y_outs        = torch.zeros(1, self.filter_number, dtype=torch.float)
        self.Xd            = torch.zeros(1, Control_filter_groups.shape[1], dtype=torch.float)
        self.Filters       = Control_filter_groups
    
    def forward(self, xin):
        """Construting the anti-noise by combing the different output signals of the pre-trainned control filter.

        Args:
            xin (Tensor.float32): the filtered reference signal 
        
        Returns:
            float32 : The anti-noise signal 
            float32 : The power of the output control signal 
        """
        self.Xd      = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = xin 
        self.Y_outs  = self.Filters    @ self.Xd.t() 
        y_anti       = self.W_gain     @ self.Y_outs
        power        = self.Y_outs.t() @ self.Y_outs
        
        return  y_anti,  power
    
    def LossFunction(self, y_anti, disturbance, power):
        """Computing the sqaure error signal of the ANC system. 

        Args:
            y_anti (Tensor.float32): The anti-noise signal
            disturbance (Tensor.float32): The disturbance signal 
            power (Tensor.float32): The power of the output signals

        Returns:
            Tensor.float32: the sequare of the error sgianl.
            Tensor.float32: the error signal. 
        """
        e = y_anti - disturbance
        
        return e**2/(2*power), e
    
    def get_coeffiecients_(self):
        """Extrating the coefficients from the generators. 

        Returns:
            float32: the coeffients of the adaptive gain vector.
        """
        return self.W_gain

#--------------------------------------------------------------------
# Function : train_adaptive_gain_algorithm()
#--------------------------------------------------------------------
def train_adaptive_gain_aglrithm(model, filter_ref, Disturbance, stepsize=0.01):
    """The gain of the output signal from the subband control filters is computed based on the NLMS algorithm.

    Args:
        model (Adaptive_control_filter_generator): The model of the adaptive gain algorithm
        filter_ref (Tensor.float32): the filtered reference signal [1 x Len] 
        Disturbance (Tensor.float32): the disturbance signal [1 x Len]
        stepsize (float, optional): the step size of the NLMS algorithm. Defaults to 0.01.

    Returns:
        float32: the error signal vector
    """
    Len_data = len(Disturbance)
    
    bar = progressbar.ProgressBar(maxval=Len_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # Setting the optimizer 
    optimizer = optim.SGD([model.W_gain], lr=stepsize)
    
    Erro_signal = []
    for itera in range(Len_data):
        # Feedforward
        xin = filter_ref[itera]
        dis = Disturbance[itera]
        y_anti, power = model.forward(xin)
        loss, e       = model.LossFunction(y_anti,dis,power)
        
        # Backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        
        # Updating the progress bar
        bar.update(itera+1)
        
    bar.finish()
    return Erro_signal
