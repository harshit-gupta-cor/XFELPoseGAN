from supervised_xfelposegan import XFELPoseGAN
from saveimage_utils import save_fig_double
import os
import mrcfile
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import mean_snr_calculator, dict2cuda,  Dict_to_Obj, downsample_dict
import pytorch3d
from pytorch3d.transforms import so3_relative_angle
from writer_utils import writer_image_add_dict, writer_scalar_add_dict
from src.dataio import dataloader
from src.transforms import downsample_avgpool, downsample_avgpool_3D,downsample_fourier_crop_3D
from src.summary_functions import write_summary
import time
from torch.profiler import profile, record_function, ProfilerActivity




class SupervisedXFELposeganWrapper():
    def __init__(self, config):
        super(SupervisedXFELposeganWrapper, self).__init__()

        self.config = config

        config_dataio=Dict_to_Obj(config.copy())
        config_dataio.multires=False
        print(f"batchsize for dataio {config_dataio.batch_size}")

        self.gt_loader, self.noise_loader, dataset=dataloader(config_dataio)

        if not self.config.simulated:
            self.config=dataset.get_df_optics_params(self.config)

        self.xfelposegan = XFELPoseGAN(config)
        self.xfelposegan.init_gen()
        self.xfelposegan.init_dis()
        self.xfelposegan.init_encoder()
        print(self.xfelposegan.encoder)
        print(self.xfelposegan.dis)
        self.xfelposegan.to(self.config.device)
        #New copy of config to detach it from config for reconstruction

        self.init_scheduler(self.xfelposegan)
        self.init_path()


        if config.init_with_gt:

            print("init with GT")

            vol_gt=self.gt_loader.sim.projector.vol.data
            size=int(vol_gt.shape[-1]*2.0/float(config.down_res_in_pixel))
            vol_down=downsample_fourier_crop_3D(vol_gt, size=size)
            print(f"init volume downsampled to size: {vol_down.shape[0]}")
            vol_up=torch.nn.functional.interpolate(vol_down[None, None, :,:,:], size=vol_gt.shape[-1]).squeeze()
            vol_up=(vol_gt.pow(2).sum().sqrt()/vol_up.pow(2).sum().sqrt())* vol_up

            self.xfelposegan.gen.projector.vol.data=vol_up.clone()

            self.xfelposegan.gen.proj_scalar.data=self.gt_loader.sim.proj_scalar.clone()

    def run(self):


        max_iter = self.meta_bools()


        #self.write_meta_bools(max_iter)

        per_epoch_iteration=len(self.gt_loader)
        total_epochs=int(np.max((max_iter//per_epoch_iteration, 1)))


        print(f"Total epochs is {total_epochs}")

        start_time=time.time()

        times={}
        total_time=0
        total_computation_time=0
        total_dataloading_time=0
        total_summary_time=0
        iteration=-1


        for epoch in range(total_epochs):
            
            for _ in range(per_epoch_iteration):


                iteration+=1
                if iteration==max_iter:
                    break



                train_all = ( (iteration% self.config.dis_iterations)==self.config.dis_iterations-1)



                if iteration==(self.config.gan_iteration+1) and self.config.use_3d_volume_encoder:
                    self.xfelposegan.encoder.projection_images= fake_data["clean"].data.squeeze()[None, :, :, :]
                    self.xfelposegan.encoder.projection_images.to(self.config.device)

                self.assign_bool_iteration(iteration)

                scale = self.meta_scheduler_dict["scale_array"][iteration]

                start_data_loading_time = time.time()


                if not self.xfelposegan.config.supervised_loss:

                    gt_data = next(self.gt_loader)
                    if "cuda" in self.config.device:
                      gt_data = dict2cuda(gt_data, verbose=iteration==0)
                 

                    gt_data = downsample_dict(gt_data, scale)

                else:
                    gt_data= None
                if not self.xfelposegan.config.tomography:

                    fake_params = next(self.noise_loader)
                    if "cuda" in self.config.device:
                      fake_params = dict2cuda(fake_params, verbose=iteration==0)
                    fake_params = downsample_dict(fake_params, scale)

                else:
                    fake_params= {}

                end_data_loading_time = time.time()

                if (iteration+1< max_iter)  and self.meta_scheduler_dict["zero_volume"][iteration+1]:
                    print("saving fake params for adjusting snr")
                    self.xfelposegan.gen.save_snr_configuration(fake_params)

                if self.meta_scheduler_dict["fill_volume_upscale"][iteration]:
                    print(f"iter: {iteration} upscaling the volume")
                    self.xfelposegan.gen.projector.fill_volume(scale - 1, scale)
                    print(f"adjusting snr")
                    self.xfelposegan.gen.adjust_snr()

                if self.meta_scheduler_dict["zero_volume"][iteration] and not "autoencoder" in self.config.exp_name:
                    print(f"iter: {iteration} zeroing the volume")
                    if self.config.multires:
                        self.xfelposegan.gen.projector.vol_dict["vol_" + str(int(scale))].data.fill_(0.0)
                    else:
                        self.xfelposegan.gen.projector.vol.data.fill_(0.0)
                    print("scalar not zeroed")
                    #self.cryoposegan.gen.proj_scalar.data.fill_(0.0)
                    #self.cryoposegan.gen.scalar.data.fill_(0.0)

                start_computation_time = time.time()
                loss_dict, rec_data, fake_data, self.writer = self.xfelposegan.train(gt_data, fake_params, max_iter, iteration,
                                                                                     self.writer, train_all)


                end_computation_time=time.time()

                if self.xfelposegan.config.gan:
                    self.scheduler_dis.step()
                    self.scheduler_gen.step()
                if self.xfelposegan.config.supervised_loss:
                    self.scheduler_encoder.step()
                    self.scheduler_scalar.step()
                if self.xfelposegan.config.tomography:
                    self.scheduler_gen.step()

                computation_time=end_computation_time-start_computation_time
                data_loading_time=end_data_loading_time-start_data_loading_time
                total_computation_time+=computation_time
                total_dataloading_time+=data_loading_time



                alpha_local=self.meta_scheduler_dict["alpha_array"][iteration]
                scale_local =self.meta_scheduler_dict["scale_array"][iteration]
                key=[keys for keys, val in self.meta_scheduler_dict.items() if keys in ["gan_bool","supervised_bool", "tomo_bool" ] and val[iteration]>0 ]
                print(f"iter: {iteration} algo: {key} scale: {scale_local} alpha:{ alpha_local}")

                summary_time=0
                summary_iteration_number=499
                if ((iteration+1)%summary_iteration_number==0 ) or (iteration==0):
                    start_summary_time=time.time()
                    volume_dict ={"gt":self.gt_loader.make_vol().cpu(),
                                  "rec":self.xfelposegan.gen.projector.make_vol().cpu()}


                    write_summary(gt_data, fake_data, rec_data, volume_dict,
                                  self.writer, iteration,
                                  summary_prefix='', compute_fsc= not self.config.supervised_loss, multi_res_factor=2**(self.config.scale[-1] - self.xfelposegan.gen.projector.scale), computation_time=total_computation_time)

                    self.writer = writer_image_add_dict(self.writer, gt_data, fake_data, rec_data, self.config, iteration)
                    if not self.xfelposegan.config.supervised_loss:
                        volume_path=self.OUTPUT_PATH + '/'+str(iteration).zfill(6)+"_volume.mrc"
                        with mrcfile.new(volume_path, overwrite=True) as m:
                            m.set_data(self.xfelposegan.gen.projector.make_vol().cpu().numpy())

                        curr_volume_path=self.OUTPUT_PATH + '/current_volume.mrc'
                        shutil.copy(volume_path, curr_volume_path)
                    if hasattr(self.xfelposegan, "encoder"):
                        torch.save(self.xfelposegan.encoder, self.OUTPUT_PATH + "/Encoder.pt")

                    if hasattr(self.xfelposegan, "gen"):
                        torch.save(self.xfelposegan.gen.noise.scalar, self.OUTPUT_PATH + "/scalar.pt")
                        torch.save(self.xfelposegan.gen.proj_scalar, self.OUTPUT_PATH + "/proj_scalar.pt")
                    end_summary_time=time.time()

                    summary_time=end_summary_time-start_summary_time





                    print(f"iter: {iteration}" )#loss_wass: {wass_loss}")
                total_summary_time+=summary_time
                times.update({"computation":total_computation_time,
                              "total/computation": total_computation_time,
                              "data_loading": data_loading_time,
                              "total/dataloading": total_dataloading_time,
                              "summary_time": summary_time,
                              "total/summary_time": total_summary_time
                             })
                self.writer=writer_scalar_add_dict(self.writer, times, iteration, prefix="time/")

        self.writer.close()


   
    def init_path(self):
        for path in ["/logs/", "/figs/"]:
            OUTPUT_PATH = os.getcwd() + path
            if os.path.exists(OUTPUT_PATH) == False:    os.mkdir(OUTPUT_PATH)
            OUTPUT_PATH = OUTPUT_PATH + self.config.exp_name
            if os.path.exists(OUTPUT_PATH) == False:    os.mkdir(OUTPUT_PATH)
            if "logs" in path:
                self.writer = SummaryWriter(log_dir=OUTPUT_PATH)

        self.OUTPUT_PATH = OUTPUT_PATH
        shutil.copy(self.config.config_path, self.OUTPUT_PATH)
  



    def assign_bool_iteration(self, iteration):
        self.xfelposegan.config.gan=self.meta_scheduler_dict["gan_bool"][iteration]
        self.xfelposegan.config.supervised_loss=self.meta_scheduler_dict["supervised_bool"][iteration]
        self.xfelposegan.config.tomography=self.meta_scheduler_dict["tomo_bool"][iteration]

        scale=self.meta_scheduler_dict["scale_array"][iteration]
        alpha = self.meta_scheduler_dict["alpha_array"][iteration]
        self.xfelposegan.gen.projector.scale=scale
        self.xfelposegan.encoder.scale=scale
        self.xfelposegan.gen.projector.alpha=alpha
        self.xfelposegan.encoder.alpha=alpha



    def meta_bools(self):

        cumulative_iteration = np.zeros((len(self.config.scale), 2))
        local_iteration = self.config.gan_iteration

        for i in range(len(self.config.scale)):
            local_iteration += self.config.sup_iteration_scale[i]
            cumulative_iteration[i, 0] = local_iteration
            local_iteration += self.config.tomo_iteration_scale[i]
            cumulative_iteration[i, 1] = local_iteration

        max_iter = local_iteration

        supervised_bool = np.zeros((max_iter,))
        xfelgan_bool = np.zeros((max_iter,))
        tomo_bool = np.zeros((max_iter,))
        scale_array = np.zeros((max_iter,))
        alpha_array = np.ones((max_iter,))
        fill_volume_upscale = np.zeros((max_iter,))
        zero_volume = np.zeros((max_iter,))
        total=0
        for iteration in range(max_iter):

            if iteration < self.config.gan_iteration:
                xfelgan_bool[iteration] = 1
                scale_array[iteration] = self.config.scale[0]
            else:
                for i, scale in enumerate(self.config.scale):

                    if iteration < cumulative_iteration[i, 0]:
                        supervised_bool[iteration] = 1
                        if i > 0:
                            diff = cumulative_iteration[i, 0] - iteration
                            ratio = diff / float(self.config.sup_iteration_scale[i])
                            ratio = 1 - ratio
                            if ratio < 0.5:
                                alpha_array[iteration] = 2 * ratio
                        break

                    if iteration == cumulative_iteration[i, 0]:
                        zero_volume[iteration] = 1

                    if iteration < cumulative_iteration[i, 1]:
                        tomo_bool[iteration] = 1
                        break

                    if iteration == cumulative_iteration[i, 1]:
                        fill_volume_upscale[iteration] = 1

                scale_array[iteration] = scale


        self.meta_scheduler_dict=({"supervised_bool" : supervised_bool,
                                    "gan_bool" : xfelgan_bool,
                                    "tomo_bool" : tomo_bool,
                                    "scale_array" : scale_array,
                                    "alpha_array" : alpha_array,
                                    "fill_volume_upscale" : fill_volume_upscale,
                                    "zero_volume" : zero_volume})


        return max_iter


    def write_meta_bools(self, max_iter):

        for local_iter in range(max_iter):
            meta_dict = {}
            for keys, val in self.meta_scheduler_dict.items():
                meta_dict.update({keys: val[local_iter]})
            self.writer = writer_scalar_add_dict(self.writer, meta_dict, local_iter, prefix="meta_bools/")

    def init_scheduler(self, xfelposegan):
        
        if hasattr(xfelposegan, "dis_optim"):

            self.scheduler_dis = torch.optim.lr_scheduler.StepLR(self.xfelposegan.dis_optim,
                                                                 step_size=self.config.scheduler_step_size * self.config.dis_iterations,
                                                                 gamma=self.config.scheduler_gamma)
        if hasattr(xfelposegan, "gen_optim"):

            self.scheduler_gen = torch.optim.lr_scheduler.StepLR(self.xfelposegan.gen_optim,
                                                                 step_size=self.config.scheduler_step_size,
                                                                 gamma=self.config.scheduler_gamma)
        if hasattr(xfelposegan, "encoder_optim"):
            self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(self.xfelposegan.encoder_optim,
                                                                     step_size=self.config.scheduler_step_size,
                                                                     gamma=self.config.scheduler_gamma)
        if hasattr(xfelposegan, "scalar_optim"):
            self.scheduler_scalar = torch.optim.lr_scheduler.StepLR(self.xfelposegan.scalar_optim,
                                                                    step_size=self.config.scheduler_step_size,
                                                                    gamma=self.config.scheduler_gamma)
        
#     def init_with_gt(self):
#         if self.config.init_with_gt:
#             with torch.no_grad():
#                 self.cryoposegan.gen.projector.vol[:, :, :] = self.GT.vol[:, :, :]
       
    def plot_images(self, gt_data, fake_data, rec_data, iteration):

       
        
        self.writer = writer_image_add_dict(self.writer, gt_data, fake_data, rec_data, iteration) 
        
        
        if fake_data is not None:
            save_fig_double(gt_data["proj"].cpu().data, fake_data["proj"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj", iteration=str(iteration).zfill(6),
                        Title1='Real', Title2='Fake_' + str(iteration),
                        doCurrent=True, sameColorbar=False)
            
            gt_data_clean=gt_data["clean"].cpu().data if "clean" in gt_data else gt_data["proj"].cpu().data
            save_fig_double(gt_data_clean, fake_data["clean"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj_clean", iteration=str(iteration).zfill(6),
                        Title1='Real_clean', Title2='fake_clean' + str(iteration),
                        doCurrent=True, sameColorbar=False)
        
        if rec_data is not None:
             
            with torch.no_grad():
                if  "clean" in gt_data:
                    down_gt_clean=downsample_avgpool(gt_data["clean"],size=rec_data["clean"].shape[-1])
                    loss_rec_clean=(rec_data["clean"]-down_gt_clean ).pow(2).sum()/self.config.batch_size
                    self.writer.add_scalar("loss/loss_rec_clean", loss_rec_clean, iteration)

            gt_data_clean=gt_data["clean"].cpu().data if "clean" in gt_data else gt_data["proj"].cpu().data
            save_fig_double(gt_data_clean, rec_data["clean"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj_rec", iteration=str(iteration).zfill(6),
                        Title1='Real_clean', Title2='rec_clean' + str(iteration),
                        doCurrent=True, sameColorbar=False)
        
        volume_path=self.OUTPUT_PATH + '/'+str(iteration).zfill(6)+"_volume.mrc"
        with mrcfile.new(volume_path, overwrite=True) as m:
            m.set_data(self.xfelposegan.gen.projector.vol.detach().cpu().numpy())
            
        curr_volume_path=self.OUTPUT_PATH + '/current_volume.mrc'
        shutil.copy(volume_path, curr_volume_path)
            
        torch.save(self.xfelposegan.encoder, self.OUTPUT_PATH + "/Encoder.pt")
        torch.save(self.xfelposegan.gen.noise.scalar, self.OUTPUT_PATH + "/scalar.pt")
        torch.save(self.xfelposegan.gen.proj_scalar, self.OUTPUT_PATH + "/proj_scalar.pt")


    