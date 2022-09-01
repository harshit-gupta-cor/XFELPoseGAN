import torchvision
from src.summary_functions import normalize_proj
import torch
from src.transforms import downsample_avgpool
import matplotlib.pyplot as plt

def writer_scalar_add_dict(writer, dict_weights, iteration, prefix=None):
    
    for keys in dict_weights:
            if prefix is not None:
                name=prefix+keys
            writer.add_scalar(name, dict_weights[keys], iteration)
    return writer
            

    
def dict_to_grid(data, config):
    grid=None
    
    def im_to_grid(im):
        if im.shape[-1]>config.ProjectionSize:
            im=downsample_avgpool(im, size=config.ProjectionSize)
        return torchvision.utils.make_grid(normalize_proj(im).data)
    keys=["tomo", "clean","proj"]
    plot=False
    for key in keys:

        if key in data:
            plot=True
            out=im_to_grid(data[key][:16], )
            break
    if plot:
        for key in keys:
            out=im_to_grid(data[key][:16]) if key in data else torch.zeros_like(out)
            grid=out if grid is None else torch.cat([grid, out], 1)


        fig = plt.figure(dpi=96)
        plt.imshow(grid[0].cpu().numpy(), cmap='Greys_r')
        plt.tight_layout()
        return grid.cpu(), fig
    else:
        return None, None
    
        
        
    
def writer_image_add_dict(writer, real_data, fake_data, rec_data, config, iteration):
    if real_data is not None:
        grid, fig= dict_to_grid(real_data, config)
        if grid is not None and fig is not None:
            writer.add_figure("images/real", fig, global_step=iteration)
        #writer.add_image("images/real",, iteration)
    if fake_data is not None:
        grid, fig= dict_to_grid(fake_data, config)
        if grid is not None and fig is not None:
            writer.add_figure("images/fake", fig, global_step=iteration)

    if rec_data is not None:

        grid, fig= dict_to_grid(rec_data, config)
        if grid is not None and fig is not None:
            writer.add_figure("images/rec", fig, global_step=iteration)
    
    return writer


    
def norm_of_weights(module):
        dictionary={}
        for params in module.named_parameters():
            if ("weight" in params[0] and  any(name in params[0] for name in ["conv", "mlp"])) or  "vol" in params[0]:
                dictionary.update({params[0]+"/_weight":params[1].data })
                dictionary.update({params[0]+"/_weight_norm":params[1].data.norm() })
                if params[1].grad is not None:
                    dictionary.update({params[0]+"/_grad": params[1].grad.data})
                    dictionary.update({params[0]+"/_grad_norm_rel": (params[1].grad.data.norm().item()/params[1].data.norm()).item()})
        return dictionary
    
def writer_update_weight(module, writer, iteration, name):
    dict_weights=norm_of_weights(module)
    for keys in dict_weights:
        if "norm" in keys:
            writer.add_scalar(name+"/"+keys, dict_weights[keys], iteration)
            
        else:
            writer.add_histogram(name+"/"+keys, dict_weights[keys], iteration)
    return writer



def dict_from_gen(gen):
    weights_dict = {}
    weights_dict.update({"gen/vol":gen.projector.vol.data.cpu()})
    if gen.vol.grad is not None:
        weights_dict.update({"gen/vol_grad": gen.projector.vol.grad.data.cpu()})
    return weights_dict

def dict_from_dis(dis):
    weights_dict = {}
    weights_dict.update({"dis/weight/mlp_image/layer_1":dis.fully_linear_image[0].weight.data.cpu()})
    if dis.fully_linear_image[0].weight.grad is not None:
        weights_dict.update({"dis/grad/mlp_image/layer_1": dis.fully_linear_image[0].weight.grad.data.cpu()})

    for num in [0,2]:
        layer_num = str(1 + num // 2)
        weights_dict.update({"dis/weight/mlp_rotmat/layer_"+layer_num: dis.fully_linear_rotmat[num].weight.data.cpu()})
        if dis.fully_linear_rotmat[num].weight.grad is not None:
            weights_dict.update({"dis/grad/mlp_rotmat/layer_"+layer_num: dis.fully_linear_rotmat[num].weight.grad.data.cpu()})

    weights_dict.update({"dis/grad/mlp_final_layer": dis.linear.weight.grad.data.cpu()})
    weights_dict.update({"dis/weight/mlp_image_output": dis.output_image.data.cpu()})
    weights_dict.update({"dis/weight/mlp_rotmat_output": dis.output_rotmat.data.cpu()})
    return weights_dict

def dict_from_encoder(encoder):
    cnn=encoder.cnn_encoder.net
    weights_dict={}
    for num in [0,2,4]:
        layer_num=str(1+num//2)
        weights_dict.update({"encoder/weight/cnn/layer_"+layer_num+"_conv1":cnn[num].conv1.weight.data.cpu()})
        weights_dict.update({"encoder/weight/cnn/layer_" + layer_num + "_conv2": cnn[num].conv2.weight.data.cpu()})
        if cnn[num].conv1.weight.grad is not None:
            weights_dict.update({"encoder/grad/cnn/layer_" + layer_num + "_conv1": cnn[num].conv1.weight.grad.data.cpu()})
            weights_dict.update({"encoder/grad/cnn/layer_" + layer_num + "_conv2": cnn[num].conv2.weight.grad.data.cpu()})
    mlp1=encoder.orientation_mlp_1.net
    for num in [0,2]:
        layer_num=str(1+num//2)
        weights_dict.update({"encoder/weight/mlp_1/layer_" + layer_num + "_linear": mlp1[num].weight.data.cpu()})
        if mlp1[num].weight.grad is not None:
            weights_dict.update({"encoder/grad/mlp_1/layer_" + layer_num + "_linear": mlp1[num].weight.grad.data.cpu()})
            
    mlp2 = encoder.orientation_mlp_2.net
    for num in [0]:
        layer_num = str(1 + num // 2)
        weights_dict.update({"encoder/weight/mlp_2/layer_" + layer_num + "_linear": mlp2[num].weight.data.cpu()})
        if mlp2[num].weight.grad is not None:
            weights_dict.update({"encoder/grad/mlp_2/layer_" + layer_num + "_linear": mlp2[num].weight.grad.data.cpu()})
            
    weights_dict.update({"encoder/weight/output": encoder.output.data.cpu()})
    
    return weights_dict

def writer_hist_add_dict(writer, weights_dict, iteration):

    for keys in weights_dict:
        writer.add_histogram(keys,weights_dict[keys], iteration )
        if "output" in keys or "vol" in keys:
            writer.add_scalar("energy/"+keys,weights_dict[keys].abs().mean(), iteration)
            if weights_dict[keys].grad is not None:
                writer.add_scalar("energy/" + keys, weights_dict[keys].grad.abs().mean(), iteration)
    return writer
