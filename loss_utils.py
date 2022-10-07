import torch
from utils import get_samps_simulator
import pytorch3d
from pytorch3d.transforms import so3_relative_angle, matrix_to_rotation_6d
import numpy as np
from src.transforms import  rotmat_pi_added


def pairwise_cos_sim(x):
    """

    Parameters
    ----------
    x: input tensor

    Returns
    -------
    pairwise_sim: pairwise cosine similarity of a tensor elements.
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # --> B,(6/3)
    # multiply row i with row j using transpose
    # element wise product
    pairwise_sim = torch.matmul(x_norm, torch.transpose(x_norm, 0, 1))

    return pairwise_sim

def contrastive_loss(rotmat):
    """

    Parameters
    ----------
    rotmat: batch_sizex3x3

    Returns
    -------
    loss: contrastive loss

    """
    flat_rotmat=rotmat.flatten(1)
    batch_sz=flat_rotmat.shape[0]
    latent_loss = (pairwise_cos_sim(flat_rotmat) - torch.eye(batch_sz).cuda()) ** 2
    loss =  latent_loss.mean()
    
    return loss



def select_min_rotmat(rotmat_true, rotmat_pred):
    #Selects min  among n-th and (n+batch_size)-th error
    #input 2Bx3x3  output--> Bx3x3
    error_full = (rotmat_pred - rotmat_true).detach().flatten(1).pow(2).sum(1)[:,None]
    error = torch.cat([error_full[:len(rotmat_true) // 2, :], error_full[len(rotmat_true) // 2:, :]], 1)
    val, ind = torch.min(error, -1)
    ind = ind * error.shape[0] + torch.arange(0, error.shape[0]).to(rotmat_true.device)
    rotmat_pred = rotmat_pred[ind]
    rotmat_true = rotmat_true[ind]
    return rotmat_true, rotmat_pred


def select_min_rotmat_tomography(image_true, image_pred, rotmat_true, rotmat_pred):
    #Selects min  among n-th and (n+batch_size)-th error
    #input 2Bx3x3  output--> Bx3x3

    error_full = (image_true - image_pred).detach().flatten(1).pow(2).sum(1)[:,None]
    error = torch.cat([error_full[:len(image_true) // 2, :], error_full[len(image_true) // 2:, :]], 1)
    val, ind = torch.min(error, -1)
    ind = ind * error.shape[0] + torch.arange(0, error.shape[0]).to(image_true.device)

    image_true = image_true[ind]
    image_pred = image_pred[ind]
    rotmat_true= rotmat_true[ind] if rotmat_true is not None else rotmat_true
    rotmat_pred= rotmat_pred[ind]


    return image_true, image_pred, rotmat_true, rotmat_pred




def calculate_loss_dis(dis, real_data, fake_data, config):
    """

    Parameters
    ----------
    dis: discriminator
    real_data
    fake_data
    config

    Returns
    -------
    loss_dict: contains
    loss_dis= Dis(real)-Dis(fake)+gradient_penalty (search WGAN gradient penalty for more details)
    loss_wass=Dis(real)-Dis(fake)
    loss_gp=gradient_penalty
    """
    fake_samps=fake_data["proj"].detach()
    real_samps=real_data["proj"]

    fake_out = dis(fake_samps)
    real_out = dis(real_samps)
    
    val_dis_fake= torch.mean(fake_out)
    val_dis_real=torch.mean(real_out)
    
    loss_wass=val_dis_real-val_dis_fake

    gp = config.lambdapenalty * stable_gradient_penalty_cryoposegan(dis, real_samps, fake_samps.detach())
    loss_dis=-loss_wass+gp
        
    loss_dict = {"loss_dis":loss_dis,
                "loss_wass": loss_wass.item(),
                "loss_gp": gp.item()}

    return loss_dict

def calculate_loss_gan_gen( dis, real_data, fake_data, config):
    """

       Parameters
       ----------
       dis: discriminator
       real_data
       fake_data
       config

       Returns
       -------
       loss_dict: contains
       loss_gan_gen= Dis(fake)

    """
    loss_dict={}
    
    fake_samps=fake_data["proj"]
    fake_out = dis(fake_samps)    
    val_dis_fake= torch.mean(fake_out)
    loss_gen_fake=-val_dis_fake
    
    loss_dict.update({"loss_gan_gen":loss_gen_fake
                     })
    
    return loss_dict

def calculate_loss_mean_std(real_data, fake_data):
    loss_dict={}
    loss_mean=(real_data["proj"].mean()-fake_data["proj"].mean()).pow(2)
    loss_std=(real_data["proj"].std()-fake_data["proj"].std()).pow(2)
    loss_mean_std=(real_data["proj"].flatten(1).pow(2).sum(1).sqrt().mean()-
                   fake_data["proj"].flatten(1).pow(2).sum(1).sqrt().mean()).pow(2)#loss_mean+loss_std
    loss_dict.update({
                      "loss_mean":loss_mean,
                      "loss_std": loss_std,
                      "loss_mean_std": loss_mean_std
        
                     })
    
    return loss_dict

def calculate_loss_supervised( encoder, gen, real_data, fake_data, alpha, config):
    """

    Parameters
    ----------
    encoder
    gen
    real_data
    fake_data
    alpha
    config

    Returns
    -------
    loss_pose_fake: L2 norm of the difference between pose predicted by encoder for the fake data and the true
                    poses used to generate the fake data. Poses are parameterized using rotation matrices.
    loss_rel_angle_fake: Same as loss_pose_fake but in SO3 pose space.
    """
   
    identity=torch.eye(3,3).to(config.device)
    rotmat_true = fake_data["rotmat"]

    image = (alpha) * fake_data["proj"].data + (1-alpha) *  fake_data["clean"].data

    if config.symmetrized_loss_sup:
        image_flipped=torch.flip(image, dims=(-2, -1))
        image=torch.cat([image, image_flipped], 0)

        rotmat_true = rotmat_pi_added(rotmat_true)


    rotmat_pred_fake=encoder(image)

    if config.symmetrized_loss_sup:
        rotmat_true, rotmat_pred_fake=select_min_rotmat(rotmat_true, rotmat_pred_fake )



    loss_pose_fake=(matrix_to_rotation_6d(rotmat_true)-matrix_to_rotation_6d(rotmat_pred_fake)).pow(2).sum()/rotmat_true.shape[0]
    with torch.no_grad():
       loss_rel_angle_fake= 180*so3_relative_angle(rotmat_true, rotmat_pred_fake).mean()/np.pi
    loss_dict={"loss_supervised":loss_pose_fake,
               "loss_supervised_angle"   :loss_rel_angle_fake}
    
    return rotmat_true, rotmat_pred_fake, loss_dict

def calculate_loss_tomography(encoder, gen, real_data, config):
    """

    Parameters
    ----------
    encoder
    gen
    real_data
    config

    Returns
    -------
    rotmat_true_real: True rotmatrix for ground truth data (also called real)
    rotmat_pred_real: predicted rotmatrix for ground truth data (also called real)
    rec_data: Data reconstructed using predicted rot matrix
    loss_dict: Dictionary containing loss variables as following
        loss_tomography: L2 norm between true real image and predicted image.


    """
    loss_dict={}
    rotmat_true_real=real_data["rotmat"] if "rotmat" in real_data else None
    if rotmat_true_real is not None and config.symmetrized_loss_tomo:
        rotmat_true_real = rotmat_pi_added(rotmat_true_real)

    with torch.set_grad_enabled("autoencoder" in config.exp_name):
        image_true=real_data["proj"]

        if config.symmetrized_loss_tomo:
            image_true=real_data["proj"]
            image_true_flipped = torch.flip(image_true, dims=(-2, -1))
            image_true = torch.cat([image_true, image_true_flipped], 0)

        rotmat_pred_real=encoder(image_true)
        rec_params={k:v for k, v in real_data.items() if "params" in k}
        rec_params.update({"rotmat":rotmat_pred_real})
        
    rec_data=get_samps_simulator(gen, rec_params, grad=True)
    if config.symmetrized_loss_tomo:
        rec_data["rotmat"]=rotmat_pred_real[:len(rotmat_pred_real)//2]
    image_pred = rec_data["clean"]


    if config.symmetrized_loss_tomo:
        image_true, image_pred, rotmat_true_real, rotmat_pred_real = select_min_rotmat_tomography(image_true, image_pred, rotmat_true_real, rotmat_pred_real)

    loss_tomography=(image_true-image_pred).flatten(1).pow(2).sum(1).mean()
    loss_dict.update({ "loss_tomography":loss_tomography})

    return  rotmat_true_real, rotmat_pred_real, rec_data, loss_dict
    
    
def calculate_loss_unsupervised( dis, encoder, gen, real_data, fake_data, config):
    """

    Parameters
    ----------
    dis
    encoder
    gen
    real_data
    fake_data
    config

    Returns
    -------
    loss_dict: dictionary containing loss values from various unsupervised approaches.
    """
    loss_dict={}
    rec_data=None
    
    if config.cryogan:
        fake_samps=fake_data["proj"]
        fake_out = dis(fake_samps)    
        val_dis_fake= torch.mean(fake_out)
        loss_gen_fake=-val_dis_fake
        
     
        loss_mean_std=(real_data["proj"].mean()-fake_data["proj"].mean()).pow(2)+(real_data["proj"].std()-fake_data["proj"].std()).pow(2)
        
        loss_dict.update({"loss_gan_gen":loss_gen_fake})
        loss_dict.update({"loss_mean_std":loss_mean_std})
    #=======================
    if config.cryoposenet :

        rotmat_pred_real=encoder(real_data["proj"])
        rec_params={k:v for k, v in real_data.items() if "params" in k}
        rec_params.update({"rotmat":rotmat_pred_real})
        
        rec_data=get_samps_simulator(gen, rec_params, grad=True)
        loss_cryoposenet=(real_data["proj"]-rec_data["clean"]).pow(2).flatten(1).sum(1).mean()
        loss_dict.update({"loss_cryoposenet":loss_cryoposenet})
        
        
        loss_dict.update({"loss_contrastive":contrastive_loss(encoder.output_cnn) })

    
    return rec_data, loss_dict
    

    
    
def dict_to_loss_dis(loss_dict, weight_dict):
    return loss_dict["loss_dis"]

def dict_to_loss(loss_dict, weight_dict):
    """

    Parameters
    ----------
    loss_dict: dict containing loss values
    weight_dict: dict containg weight for each loss value

    Returns
    -------
    summation of the weighted loss.

    """
    loss=0
    weight_total=0
    for keys in weight_dict:
        
        if weight_dict[keys]<0:
            raise AssertionError(keys+" val is negative")
     
            
        if weight_dict[keys]>0:
            loss+=weight_dict[keys]*loss_dict[keys[7:]]
        weight_total+=weight_dict[keys]
    loss=loss/weight_total
    return loss
    
    
    

def stable_gradient_penalty_cryoposegan(dis, real_samps, fake_samps):
    """
    private helper for calculating the gradient penalty
    :param real_samps: real samples
    :param fake_samps: fake samples
    :param reg_lambda: regularisation lambda
    :return: tensor (gradient penalty)
    """
    batch_size = real_samps.shape[0]

    # generate random epsilon
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

    # create the merge of both real and fake samples
    merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
    merged.requires_grad_(True)

    # forward pass
    op = dis(merged)

    # perform backward pass from op to merged for obtaining the gradients
    gradients = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # Return gradient penalty
    #print("wrong gradient penalty computations")
    return ((gradients_norm  - 1) ** 2).mean()

