import numpy as np
import torch
import torch.fft
from torch import nn
from pytorch3d.transforms import random_rotations, rotation_6d_to_matrix, so3_relative_angle
from modules import  Discriminator, Encoder, weights_init, RotDiscriminator
from writer_utils import writer_image_add_dict, writer_update_weight, dict_from_dis, dict_from_gen, norm_of_weights, writer_scalar_add_dict
from loss_utils import calculate_loss_dis, calculate_loss_supervised, calculate_loss_unsupervised, calculate_loss_tomography, dict_to_loss, dict_to_loss_dis, calculate_loss_gan_gen, calculate_loss_mean_std
from utils import get_samps_simulator
from src.simulator_utils import LinearSimulator, XFELSimulator, DenoiseSimulator
from src.summary_functions import write_euler_histogram
from src.plot3d_utils import align_rotmat
import time
class XFELPoseGAN(nn.Module):
    """
    This object initializes all the modules and
    contains access to all the sub methods need to run the XFELposeGAN algorithm.
    """
    def __init__(self, config):
        super(XFELPoseGAN, self).__init__()
        """
        Initialize the config and time variables.
        """
        self.config = config
        self.max_rel_angle=0
        self.save_itr=0
        self.saved_fake_data={}
        self.time={}
        self.time.update({"total/gan_complete":0,
                         "gan/total_dis_forward":0,
                         "gan/total_dis_backward":0,
                          "gan/total_gen_forward":0,
                         "total/sup_complete":0,
                         "sup/total_forward":0,
                         "sup/total_backward":0,
                         "total/tomo_complete":0,
                         "tomo/total_forward":0,
                         "tomo/total_backward":0})
                         
    def init_dis(self):
        """
        Initialize the discriminator architecture and optimizer.

        """

        self.dis = Discriminator(self.config)
        if not self.config.encoder_equalized_lr:
         self.dis.apply(lambda m: weights_init(m, self.config))
        self.dis_optim = torch.optim.Adam(self.dis.parameters(),
                                          lr=self.config.dis_lr,
                                          betas=(self.config.dis_beta_1, self.config.dis_beta_2),
                                          eps=self.config.dis_eps,
                                          weight_decay=self.config.dis_weight_decay)

    def init_gen(self):
        """
        Initialize the generator components like simulator and scalar (noise level estimator) and optimizer.

        """
        self.config.side_len=self.config.rec_side_len
        

        if "conditional_gan" in self.config.exp_name:
            self.gen= DenoiseSimulator(self.config)
        else:
            mode="new_idea" if "new_idea" in self.config.exp_name else "projection"
            self.gen = LinearSimulator(self.config,mode=mode )
        
        list_params=list(self.gen.projector.parameters())+list({self.gen.proj_scalar})
        
        self.gen_optim = torch.optim.Adam(list_params,
                                          lr=self.config.gen_lr,
                                           betas=(self.config.gen_beta_1, self.config.gen_beta_2),
                                           eps=self.config.gen_eps,
                                           weight_decay=self.config.gen_weight_decay)
        
        self.scalar_optim = torch.optim.Adam([{"params":self.gen.scalar}],
                                           lr=self.config.scalar_lr,
                                            betas=(self.config.scalar_beta_1, self.config.scalar_beta_2),
                                            eps=self.config.scalar_eps,
                                            weight_decay=self.config.scalar_weight_decay)

        print("scalar optim is off since everything is normalized for now")

       
    def init_encoder(self):
        """
        Initialize the encoder architecture and optimizer.

        """

        self.encoder = Encoder(self.config)
        if not self.config.encoder_equalized_lr:
            self.encoder.apply(lambda m: weights_init(m, self.config))
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(),
                                              lr=self.config.encoder_lr,
                                              betas=(self.config.encoder_beta_1, self.config.encoder_beta_2),
                                              eps=self.config.encoder_eps,
                                              weight_decay=self.config.encoder_weight_decay, amsgrad=True)


    def get_fake_data(self, params):
        """

        Parameters
        ----------
        params: dictionary containing parameters of the fake data like rotation matrices.
                The parameters are used by the generator to create fake data.

        Returns
        -------
        fake_data: dictionary containing fake data (including images and the rotation matrices associated)

        """
        save_fake_data=False
        if save_fake_data:
            self.save_itr+=1
            max_saved_data=5000//self.config.batch_size
            if self.save_itr<max_saved_data:
                fake_data = get_samps_simulator(self.gen, params, grad=True)

                if self.saved_fake_data:
                    for keys, val in  fake_data.items():
                        if val is not None:
                            self.saved_fake_data[keys]=torch.cat([self.saved_fake_data[keys], val], dim=0)
                        else:
                            self.saved_fake_data[keys]=val


                else:
                    self.saved_fake_data= fake_data

            else:
                print(f"taking from saved")
                fake_data={}
                indices=np.random.randint(0, max_saved_data, self.config.batch_size)
                for keys, val in self.saved_fake_data.items():
                    if val is not None:
                        fake_data[keys] = self.saved_fake_data[keys][indices]
                    else:
                        fake_data[keys]=None

            return fake_data
        else:
            fake_data = get_samps_simulator(self.gen, params, grad=True)
            return fake_data


    def train(self, real_data, params, max_iter, iteration, writer, train_all=True):
        """
        Runs the corresponding method depending on the value of
        self.config.gan
        self.config.supervised_loss
        self.config.tomography

        For each method calls the forward, loss functions, backpropagation, and registers the time taken.

        Parameters
        ----------
        real_data: ground truth data with images and rotation matrices (if simulated data).
        params: params to create fake data.
        max_iter: maximum number of iterations.
        iteration: current iteration number.
        writer: tensorboard writer.
        train_all: bool which specifies if the generator needs to be backprogable at a given iteration.

        Returns
        -------
        loss_dict: dictionary containing values of all the loss functions estimated at the current iteration.
        rec_data: dictionary containing data reconstructed from the tomography step. None if tomography method is not called.
        fake_data: dictionary containing data reconstructed from the gan step. None if gan method is not called.
        writer: tensorboard writer.
        """
        
        config=self.config
        rec_data=None
        fake_data=None
        loss_dict={}

        
        weight_dict_gen={"weight_loss_gan_gen":1* self.config.gan}
        weight_dict_mean_std={"weight_loss_mean_std":1* self.config.gan*self.config.weigt_loss_mean_std}
        weight_dict_dis={"weight_loss_dis":1* self.config.gan}
         
        weight_dict_supervised={"weight_loss_supervised":1*self.config.supervised_loss }
        weight_dict_tomography={"weight_loss_tomography":1 * self.config.tomography}
     


        if self.config.gan:
            
            
            start_gan=time.time()
            start_gen_forward=time.time()
            
            if "conditional_gan" in self.config.exp_name:
                if "noise_params" in params:
                    params["noise_params"]={}
                params["noise_params"].update({"real_proj":real_data["proj"]})
                
            

            fake_data=get_samps_simulator(self.gen, params, grad=train_all)

            end_gen_forward=time.time()
            
            
            start_dis_forward=time.time()
            loss_dis_dict = calculate_loss_dis(self.dis, real_data, fake_data, self.config)
            loss_dis=dict_to_loss(loss_dis_dict, weight_dict_dis)
            end_dis_forward=time.time()
            
            start_dis_backward=time.time()
            loss_dis.backward()
            end_dis_backward=time.time()
    
            
            
            if train_all and iteration%10==11:
                 writer=writer_update_weight(self.dis, writer, iteration, name="Dis")
            if  iteration%10==11 and "conditional_gan" in self.config.exp_name:
                  writer=writer_update_weight(self.gen.projector, writer, iteration, name="unet")
                    
            
            self.train_dis()
            self.zero_grad()
            loss_dict.update(**loss_dis_dict)
            
            if train_all:
                loss_gen_dict=calculate_loss_gan_gen( self.dis, real_data, fake_data, self.config)
                loss_gen=dict_to_loss(loss_gen_dict, weight_dict_gen)



                if (not "new_idea" in self.config.exp_name) and weight_dict_mean_std["weight_loss_mean_std"]:
                    loss_mean_std_dict=calculate_loss_mean_std(real_data, fake_data)
                    loss_mean_std=dict_to_loss(loss_mean_std_dict, weight_dict_mean_std)
                    loss_dict.update(**loss_mean_std_dict)
                    loss_gen+=loss_mean_std
                loss_gen.backward()
                self.train_gen()
                self.zero_grad()



                loss_dict.update(**loss_gen_dict)

            end_gan=time.time()
            
            complete_time=end_gan-start_gan
            dis_forward_time=end_dis_forward-start_dis_forward
            dis_backward_time=end_dis_backward-start_dis_backward
            gen_forward_time=end_gen_forward-start_gen_forward
            self.time["total/gan_complete"]+=complete_time
            self.time["gan/total_dis_forward"]+=dis_forward_time
            self.time["gan/total_dis_backward"]+=dis_backward_time
            self.time["gan/total_gen_forward"]+=gen_forward_time
            self.time.update({"gan/complete": complete_time,
                              "gan/dis_forward":dis_forward_time,
                              "gan/dis_backward":dis_backward_time,
                              "gan/gen_forward": gen_forward_time})


            
        if  self.config.supervised_loss:
            start_supervise=time.time()
            
            start_supervise_forward=time.time()

            fake_data=self.get_fake_data(params)
                        
            if self.config.progressive_supervision:
                ratio=float(iteration-self.config.gan_iteration)/float(self.config.supervised_loss_iteration)
                alpha_min=np.max((0.0, ratio-0.1))
                alpha=np.min((1.0, alpha_min/0.9))
              
            else:
                alpha=1
            
            rotmat_true_fake, rotmat_pred_fake, loss_supervised_dict=calculate_loss_supervised(self.encoder, self.gen, real_data, fake_data, alpha, self.config)
            loss_supervised=dict_to_loss(loss_supervised_dict, weight_dict_supervised)
            
            end_supervise_forward=time.time()
            start_supervise_backward=time.time()

            loss_supervised.backward()
            
            end_supervise_backward=time.time()
             
            if iteration%2000==0:
                writer=writer_update_weight(self.encoder, writer, iteration, name="Encoder")
                
            self.train_enc()
            self.zero_grad()
            loss_dict.update(**loss_supervised_dict)
               
            end_supervise=time.time()
            
            
            if iteration%2000==0:
                    write_euler_histogram(writer, iteration,rotmat_true_fake, rotmat_pred_fake, name="Fake" )
         
            complete_time=end_supervise-start_supervise
            supervise_forward_time=end_supervise_forward-start_supervise_forward
            supervise_backward_time=end_supervise_backward-start_supervise_backward
            
            self.time["total/sup_complete"]+=complete_time
            self.time["sup/total_forward"]+=supervise_forward_time
            self.time["sup/total_backward"]+=supervise_backward_time
            self.time.update({"sup/complete": complete_time,
                         "sup/forward":supervise_forward_time,
                         "sup/backward":supervise_backward_time
                                })


                
        if self.config.tomography:  
            #==========================
            start_tomo=time.time()
            
            start_tomo_forward=time.time()
                
            rotmat_true_real, rotmat_pred_real, rec_data,loss_tomography_dict=calculate_loss_tomography(self.encoder, self.gen, real_data, config)
            loss_tomography=dict_to_loss(loss_tomography_dict, weight_dict_tomography)


            end_tomo_forward=time.time()
            
            start_tomo_backward=time.time()
        
            loss_tomography.backward()
            end_tomo_backward=time.time()
            
            if iteration%2000==0:
                writer=writer_update_weight(self.gen, writer, iteration, name="Gen")
                writer = writer_update_weight(self.encoder, writer, iteration, name="Encoder")

            self.train_gen()
            if "autoencoder" in self.config.exp_name:
                self.train_enc()

            self.zero_grad()

            end_tomo=time.time()
            
            
            if iteration%2000==0:
                    if rotmat_true_real is not None:
                        aligned_rotmat_pred_real=align_rotmat(rotmat_true_real,rotmat_pred_real)
                        write_euler_histogram(writer, iteration,rotmat_true_real,aligned_rotmat_pred_real , name="Real" )
                        loss_tomography_angle=180*so3_relative_angle(rotmat_true_real, aligned_rotmat_pred_real).mean()/np.pi
                        loss_tomography_dict.update({"loss_tomography_angle":loss_tomography_angle.item()})
            loss_dict.update(**loss_tomography_dict)
           
            complete_time=end_tomo-start_tomo
            tomo_forward_time=end_tomo_forward-start_tomo_forward
            tomo_backward_time=end_tomo_backward-start_tomo_backward
            
            self.time["total/tomo_complete"]+=complete_time
            self.time["tomo/total_forward"]+=tomo_forward_time
            self.time["tomo/total_backward"]+=tomo_backward_time
            self.time.update({"tomo/complete": complete_time,
                              "tomo/forward":tomo_forward_time,
                              "tomo/backward":tomo_backward_time
                                })
            


        weight_dict={**weight_dict_gen, **weight_dict_dis, **weight_dict_supervised, **weight_dict_tomography }
        writer=writer_scalar_add_dict(writer, loss_dict, iteration, prefix="loss/")
        writer=writer_scalar_add_dict(writer, weight_dict, iteration, prefix="coefficients/")
        writer=writer_scalar_add_dict(writer, self.time, iteration, prefix="time/")


       
        writer.add_scalar("sigma_snr", torch.exp(-self.gen.proj_scalar.data[0]), iteration)
        writer.add_scalar("sigma_scaling", torch.exp(self.gen.scalar.data[0]), iteration)
        writer.add_scalar("SNR_estimated", self.gen.noise.current_snr, iteration)
        
    
        return loss_dict, rec_data, fake_data, writer
       

    def writer_image_add(self, writer, real_data, fake_data):
        if writer is not None:
            writer = writer_image_add_dict(writer, real_data, fake_data)
        return writer

    def train_dis(self):
        
        if self.config.dis_clip_grad == True:
            dictionary = list(self.dis.parameters())
            torch.nn.utils.clip_grad_norm_(dictionary, max_norm=self.config.dis_clip_norm_value)

        self.dis_optim.step()

        
    def zero_grad(self):
        """
        Zeros the gradient after backpropogration for tensors.

        """
        self.dis_optim.zero_grad()
        self.gen_optim.zero_grad()
        self.encoder_optim.zero_grad()
        self.scalar_optim.zero_grad()
        


    def train_gen(self):
        """
        Optimizer step for generator
        """
    
        if self.config.gen_clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.gen.projector.parameters(), max_norm=self.config.gen_clip_norm_value)
            torch.nn.utils.clip_grad_norm_(self.gen.proj_scalar, max_norm=self.config.scalar_clip_norm_value)


        self.gen_optim.step()
        self.constraint()
            
    def train_scalar(self):
        """
        Optimizer step for scalar (noise level estimator)
        """
        if self.config.scalar_clip_grad == True:
                torch.nn.utils.clip_grad_norm_(self.gen.scalar, max_norm=self.config.scalar_clip_norm_value)
                

        self.scalar_optim.step()
        self.constraint()
        

    def train_enc(self):
        """
        Optimizer step for encoder.
        """

        if self.config.encoder_clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.config.encoder_clip_norm_value)
        self.encoder_optim.step()


    def constraint(self):
        """
        enforces the constraints on the projector.
        """

        self.gen.projector.constraint()


    



