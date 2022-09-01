import sys
import os
import torch
import torch.fft
import mrcfile
import starfile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix, random_rotations, quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix
from simulator_utils import LinearSimulator, XFELSimulator
from projector_utils import ProjectorMultiRes, XFELMultiRes, EwaldProjector
import matplotlib.pyplot as plt
from transforms import half_so3, downsample_avgpool, primal_to_fourier_3D, downsample_avgpool_3D,downsample_fourier_crop_3D, fourier_to_primal_3D,fourier_to_primal_2D, vol_to_autocorr
sys.path.insert(0, "/sdf/home/h/hgupta/ondemand/CryoPoseGAN/")
from utils import dict2cuda



def dataloader_from_dataset(dataset, config):

    out_dataloader = DataLoader(dataset,
                                shuffle=True, batch_size=config.batch_size,
                                pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    return out_dataloader

def get_ctf_params(config, max_defocus=1.2, min_defocus=0.6):
        """Get the parameters for the CTF of the particle from a distribution.
        If config.ctf is True else returns None
        Returns
        -------
        ctf_params: dict of type str to {tensor}
            dictionary containing:
            "defocus_u": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the major
                defocus value in microns
            "defocus_v": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the minor
                defocus value in microns
            "defocus_angle": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the astigatism
                angle in radians
        """
        if config.ctf:
            defocus_u = (
                min_defocus
                + (max_defocus - min_defocus)
                * torch.zeros(config.batch_size)[:, None, None, None].uniform_()
            )
            defocus_v = defocus_u.clone().detach()
            defocus_angle = torch.zeros_like(defocus_u)
            return {
                "defocus_u": defocus_u.to(config.device),
                "defocus_v": defocus_v.to(config.device),
                "defocus_angle": defocus_angle.to(config.device),
            }
        else:
            return None
        
def get_shift_params(config, std=6):
    if config.shift:
        
        return {"shift_x": std*torch.randn(config.batch_size).to(config.device),
            "shift_y": std*torch.randn(config.batch_size).to(config.device)
           }

    else:
        return None
    
    
    

def dataloader( config):
    if config.simulated:
        if "xsdfsdfray" in config.exp_name:
            data_loader=  SimulatedXFELDataLoader(config)
            dataset=data_loader
            noise_loader=SimulatedXFELDataLoader(config, fake_params=True)
        elif hasattr(config, "cryo") and config.noise_type=="cryo":
            data_loader=SimulatedDataLoader(config)
            dataset=data_loader
            noise_loader=SimulatedDataLoader(config, fake_params=True)
        else:
            data_loader=SimulatedDataLoader(config)
            dataset=data_loader
            noise_loader=SimulatedDataLoader(config, fake_params=True)
            
    else:
       
        config.relion_path=config.data_relion_path
        config.relion_star_file=config.data_star_path
        
        dataset=RelionDataLoader(config, noise_loader=False)
        
        config.relion_path=config.background_relion_path
        config.relion_star_file=config.background_star_path
        
        
        noise_dataset=RelionDataLoader(config, noise_loader=True)
        
        data_loader = dataloader_from_dataset(dataset, config)
        noise_loader = dataloader_from_dataset(noise_dataset, config)
        
        
        
    return data_loader, noise_loader, dataset


def remove_none_from_dict(dictionary):
    output = {k: v for k, v in dictionary.items() if v is not None}
    return output

def rotmat_generator(num, config):
#         if config.protein== "betagal"  and "conditional_gan" not in config.exp_name :
#             rotmat=filter_rotation(random_rotations(num))
#         else:
        rotmat=random_rotations(num)
        return rotmat

def init_gt_generator(config):
        L = config.side_len

        print("Protein is "+config.protein )
        if config.protein == "betagal" :
            with mrcfile.open("/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs/GroundTruth_Betagal-256.mrc") as m:
                vol = torch.Tensor(m.data.copy()) / 10


        elif config.protein == "ribo": 
            with mrcfile.open("/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs/80S.mrc") as m:
                vol = torch.Tensor(m.data.copy()) / 1000


        elif config.protein == "cube":
            vol = torch.Tensor(init_cube(L)) / 50

        elif config.protein == "splice":
            with mrcfile.open("/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs/GroundTruth_Splice-128.mrc") as m:
                vol = torch.Tensor(m.data.copy()) / 1e6
                

        else:
            vol = torch.Tensor(init_cube(L)) / 50

        if config.gt_side_len != vol.shape[-1]:
            vol = downsample_fourier_crop_3D(vol.to(config.device), size=config.gt_side_len)
            
            name="/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs/GroundTruth_"+config.protein+"_"+str(config.gt_side_len)+".mrc"
            with mrcfile.new(name, overwrite=True) as m:
                m.set_data(vol.cpu().numpy())


        return vol



class SavedDataLoader(Dataset):
    def __init__(self, config, fake_params=False):
        self.config=config

        self.counter = 0
        self.dictionary = {}

        dataset_name = self.config.protein + "_snr_" + str(self.config.snr_val) + "_projsize_" + str(self.config.gt_side_len)
        self.mrc_path = "./datasets/" + dataset_name
        self.mrcs=mrcfile.mmap(self.mrc_path+"/gt_data.mrcs")
        self.rotmat = torch.from_numpy(mrcfile.mmap(self.mrc_path + "/gt_data_rotmat.mrc").data)

    def make_vol(self):
        if not hasattr(self, "vol"):
            self.vol = torch.from_numpy(mrcfile.open(self.mrc_path + "/gt_vol.mrc").data)
        return self.vol.to(self.config.device)



    def __len__(self):
        return self.config.datasetsize // self.config.batch_size

    def __next__(self):
        self.counter += 1
        output_dict={}
        indices=np.random.randint(0, self.__len__(), self.config.batch_size)
        output_dict["proj"]= torch.from_numpy(self.mrcs.data[indices])
        output_dict["rotmat"]=self.rotmat.data[indices]
        return output_dict

    def __iter__(self):
        return self




class SimulatedDataLoader(Dataset):
    def __init__(self, config, fake_params=False):
        self.config=config
        self.fake_params=fake_params
        self.config.side_len=self.config.gt_side_len
        
        
        if not self.fake_params:
            self.sim=LinearSimulator(self.config).to(self.config.device)

            self.sim.projector.vol.requires_grad=False
            self.sim.proj_scalar.requires_grad=False
            with torch.no_grad():

                vol=init_gt_generator(self.config)[:,:,:].to(self.config.device)
                self.sim.projector.vol[:, :, :]=vol

                if "autocorr" in self.config.exp_name:
                    self.sim.projector.vol.data=vol_to_autocorr(self.sim.projector.vol.data)
                elif "ewald" in self.config.exp_name:
                    print("here")
                    self.sim.projector = EwaldProjector(self.config).to(self.config.device)
                    self.sim.projector.vol.data = primal_to_fourier_3D(vol).abs().pow(2)
            self.snr_specifier()
        
        self.normalizer=torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)
    
        print("dataloader uses gaussian assumption for snr calculation\n")
        print("clean proj value scaled to change snr in gt\n")
        print(f"GT sidelen {self.config.side_len}x{self.config.side_len} ProjSize: {self.config.ProjectionSize}x{self.config.ProjectionSize}")
        
        
        self.counter=0
        self.dictionary={}

    def __len__(self):
        return self.config.datasetsize//self.config.batch_size

    def make_vol(self):
        if "ewald" in self.config.exp_name:
            return fourier_to_primal_3D(self.sim.projector.vol.data).real

        else:
            return self.sim.projector.make_vol()

    
    def get_samps_sim(self):
        ctf_params  =get_ctf_params(self.config) if self.config.ctf else None
        shift_params =get_shift_params(self.config) if self.config.shift else None
        rotmat =rotmat_generator(self.config.batch_size, self.config).to(self.config.device)
        noise_params =None
        if not self.fake_params:
            with torch.no_grad():
                output_dict=self.sim(rotmat, ctf_params, shift_params, noise_params)
            
            proj=output_dict["proj"]
            
            if proj.shape[-1]!= self.config.ProjectionSize:
                proj=downsample_avgpool(proj, size=self.config.ProjectionSize)
                
            if self.config.normalize_gt:
                 proj=self.normalizer(proj)
            if self.config.flip_sign:
                 proj=-proj
            output_dict["proj"]=proj
            output_dict["rotmat"] = rotmat
            
        
        else:
            output_dict={"noise_params": None,
                       "rotmat":  half_so3(rotmat),
                       "shift_params": shift_params,
                       "ctf_params": ctf_params

                      }

        return output_dict
        
    def __next__(self):
        self.counter+=1
        output_dict=self.get_samps_sim()
        return output_dict

    def __iter__(self):
        return self
        
        
    def snr_specifier(self):
        save_mode=self.config.normalize_gt
        self.config.normalize_gt=False

        output_dict=self.get_samps_sim() 
        signal_energy=output_dict["clean"].pow(2).flatten(1).sum(1).sqrt()
        noise_energy=torch.randn_like(output_dict["clean"]).pow(2).flatten(1).sum(1).sqrt()
        mean_energy_ratio=(signal_energy/noise_energy).mean()
            
        sigma_val=mean_energy_ratio*(10**(-self.config.snr_val/20.0))
        inv_sigma_val=1/sigma_val
        with torch.no_grad():
            self.sim.proj_scalar[0]=torch.log(inv_sigma_val)
        self.config.normalize_gt=save_mode

    def __getitem__(self, idx):
         pass



class SimulatedXFELDataLoader(Dataset):
    def __init__(self, config, fake_params=False):
        self.config = config
        self.fake_params = fake_params
        self.config.side_len = self.config.gt_side_len
        self.vol=init_gt_generator(self.config).to(self.config.device)

        if not self.fake_params:
            self.sim = LinearSimulator(self.config).to(self.config.device)
            self.sim.projector=EwaldProjector(self.config).to(self.config.device)
            self.sim.projector.vol.data = primal_to_fourier_3D(self.vol).abs().pow(2)
            # self.sim.projector = ProjectorMultiRes(self.config).to(self.config.device)
            # print(self.sim.projector)
            # for keys in self.sim.projector.vol_dict:
            #     self.sim.projector.vol_dict[keys].requires_grad = False
            #     with torch.no_grad():
            #         self.sim.projector.vol_dict[keys][:, :, :] = fourier_to_primal_3D(primal_to_fourier_3D(init_gt_generator(self.config).to(self.config.device))).real
            #self.sim.projector = ProjectorMultiRes(self.config).to(self.config.device)
        print("Data generated with ewald sphere in XFEL. ")

        self.normalizer = torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)

        print(
            f"GT sidelen {self.config.side_len}x{self.config.side_len} ProjSize: {self.config.ProjectionSize}x{self.config.ProjectionSize}")

        self.counter = 0

        if self.config.mask_2d:
            length = self.config.ProjectionSize
            ax = torch.linspace(-length / 2.0, length / 2.0, length)
            diameter = self.config.mask_2d_diameter
            xx, yy = torch.meshgrid(ax, ax)

            self.mask_2d = ((xx ** 2 + yy ** 2).sqrt() < length / 2.0 * diameter).to(self.config.device)

    def __len__(self):
        return 41000 // self.config.batch_size

    def get_samps_sim(self):

        rotmat = rotmat_generator(self.config.batch_size, self.config).to(self.config.device)
        noise_params = None

        if not self.fake_params:
            with torch.no_grad():
                output_dict = self.sim(rotmat, None, None, noise_params)

            proj = output_dict["proj"]

            if proj.shape[-1] != self.config.ProjectionSize:
                proj = downsample_avgpool(proj, size=self.config.ProjectionSize)

            if self.config.normalize_gt:
                proj = self.normalizer(proj)
            if self.config.flip_sign:
                proj = -proj

            if self.config.mask_2d:
                proj = proj * self.mask_2d

            output_dict.update({"proj": proj,
                                "f_proj": proj,
                                "rotmat": rotmat,
                                "shift_params": None,
                                "ctf_params": None,
                                "noise_params": None
                                })

        else:
            output_dict = {"noise_params": None,
                           "rotmat": half_so3(rotmat),
                           "shift_params": None,
                           "ctf_params": None

                           }

        return output_dict

    def __next__(self):
        self.counter += 1

        output_dict = self.get_samps_sim()

        return output_dict

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        pass

    def make_vol(self):

        return fourier_to_primal_3D(self.sim.projector.vol.data ).real



    

    
    
    
    
    
