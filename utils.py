import torch
from modules import ExplicitVolume
import mrcfile
from pytorch3d.transforms import rotation_6d_to_matrix, random_rotations, quaternion_to_matrix, matrix_to_quaternion
import numpy as np

def to_numpy(t):
    return t.detach().cpu().numpy()

def sigmoid(x):
    return  1/(1 + np.exp(-x))

def cryoposenet_weight_scheduler(config, iteration, max_iter, current_rel_angle=None,max_rel_angle=None ):
    
    if current_rel_angle is not None and max_rel_angle is not None and config.supervised_loss==True:
        rel_angle_ratio=current_rel_angle/max_rel_angle
    else:
        rel_angle_ratio=0
    iteration_ratio=float(iteration)/max_iter
    relu=5*np.max(((iteration_ratio-0.1), 0))

    cryoposenet_weight=sigmoid(np.tan( np.pi *(0.5-rel_angle_ratio)))*np.min((relu ,1) ) if config.supervised_loss else 1.0
    
    return cryoposenet_weight



def mean_snr_calculator(clean, noisy):
    snr=10*torch.log10(clean.flatten(1).pow(2).sum(1)/(noisy-clean).flatten(1).pow(2).sum(1))
    return snr.mean().item()


    
def get_samps_simulator(module, params, grad=True):
    ctf_params=params["ctf_params"] if "ctf_params" in params else None
    shift_params=params["shift_params"] if "shift_params" in params else None
    noise_params=params["noise_params"] if "noise_params" in params else None
    
    with torch.set_grad_enabled(grad):
        proj_dict=module( params["rotmat"],  ctf_params, shift_params, noise_params)
    return proj_dict
        

class Dict_to_Obj(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(Dict_to_Obj, self).__init__(*args, **kwargs)
        self.__dict__ = self


def update_config(config):
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        config["device"]="cpu"

    if config["multires"]:
        num=config["scale"][0]-1
    else:
        num=int(np.log2(config["gt_side_len"])-1) if config["simulated"] else int(np.log2(128)-1)
    
    config["num_layer_Discriminator"]=np.max((num, 4))
    num=config["num_layer_Discriminator"]
    print(f"\nNumber of layers in the discriminator is {num}.\n")
    print("converting 1e type notation to float")
    

    for keys in config:
    
        if  isinstance(config[keys], str) :
            if "e-" in config[keys] or "1e" in config[keys] :
                config[keys]=float( config[keys])

    return config




def mean_snr(GT, scalar_gt, config):

    save_mode_down=GT.downsample
    GT.downsample=False
    
    
    gt_data=get_gt_samps(GT, scalar_gt, config)
    mean_snr_val=mean_snr_calculator(gt_data["samps_clean"], gt_data["samps"])
   
    GT.downsample=save_mode_down

    return mean_snr_val
    
    
def sigma_estimator(GT, config):
    
    
    save_mode_down=GT.downsample
    GT.downsample=False
    
    
    gt_data=get_samps(GT, torch.zeros(2).to(config.device), config)
    energy_ratio=(gt_data["samps_clean"].flatten(1).pow(2).sum(1).sqrt())/torch.randn_like(gt_data["samps_clean"]).flatten(1).pow(2).sum(1).sqrt()
    sigma_val=energy_ratio*(10**(-config.snr_val/20.0))
    
    GT.downsample=save_mode_down
    return sigma_val.mean().item()
    

def ExpandVolume( X, n):
        Y=torch.zeros(n,n,n)
        x0=(n-X.shape[0])//2
        x1=x0+X.shape[0]
            
        y0=(n-X.shape[1])//2
        y1=y0+X.shape[1]
            
        z0=(n-X.shape[2])//2
        z1=z0+X.shape[2]
            
        Y[x0:x1, y0:y1, z0:z1]=X
        return Y
    

def dict2cuda(a_dict,verbose=False):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value,torch.Tensor):
            tmp.update({key: value.cuda()})
            if verbose:
                print(key,value.shape)
        elif isinstance(value, dict):
            tmp.update({key: dict2cuda(value)})
        else:
            tmp.update({key: value})
    return tmp

def downsample_dict(dict, scale):
    for keys, val in dict.items():
        if keys in ["clean", "tomo", "proj", "noise"]:
            kernel_size=int(val.shape[-1]//2**scale)
            val=torch.nn.AvgPool2d(kernel_size=kernel_size)(val)
            dict.update({keys:val})

    return dict


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value,torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp