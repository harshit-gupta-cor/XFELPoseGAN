"""Module to corrupt the projection with noise."""
import torch
from transforms import downsample_avgpool

class Noise(torch.nn.Module):
    """Class to corrupt the projection with noise.
    Written by J.N. Martel and H. Gupta.
    Parameters
    ----------
    config: class
        contains parameters of the noise distribution
    """

    def __init__(self, config):

        super(Noise, self).__init__()
        self.config=config
        self.scalar = (torch.Tensor([0]))
        self.current_snr=100
        
        print("noise module sigma kept constant to 1")
    def forward(self, proj, noise_params ):
        """Add noise to projections.
        Currently, only supports additive white gaussian noise.
        Parameters
        ----------
        proj: torch.Tensor
            input projection of shape (batch_size,1,side_len,side_len)
        Returns
        -------
        out: torch.Tensor
            noisy projection of shape (batch_size,1,side_len,side_len)
        """
        
        noise_sigma = torch.exp(self.scalar[0].data) 
        
        
        
        
        if  noise_params is not None and  "noise" in noise_params:
                noise= noise_params["noise"]
        
        else:
            
            noise_randn=torch.randn(proj.shape[0], 1, self.config.gt_side_len,  self.config.gt_side_len).to(proj.device)
            if proj.shape[-1] != noise_randn.shape[-1]:
                noise=downsample_avgpool(noise_randn, size=proj.shape[-1])
            else:
                noise=noise_randn
        
                    
        out = proj + noise_sigma * noise
        self.current_snr=self.snr_calculator(proj, out)
           
        return out

    def snr_calculator(self, clean,noisy, scale=5):
        clean=downsample_avgpool(clean, size=2**scale)
        noisy=downsample_avgpool(noisy, size=2**scale)
        snr = 10 * torch.log10(clean.flatten(1).pow(2).sum(1) / (noisy - clean).flatten(1).pow(2).sum(1))
        snr=snr.mean().item()
        return snr



