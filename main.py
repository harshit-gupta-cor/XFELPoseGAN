import argparse
import os
import sys
import traceback
import yaml

sys.path.insert(0, "/sdf/home/h/hgupta/ondemand/XFELPoseGAN/")
sys.path.insert(0, "/sdf/home/h/hgupta/ondemand/XFELPoseGAN/src/")
from utils import Dict_to_Obj, update_config
from wrapper import SupervisedXFELposeganWrapper
from generate_data import GenerateData


# TODO: add pbar
def init_config():
    # takes the cfg file with parameters and creates a variable called config with those parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", metavar="FILE", help="Specify config file")
    parser.add_argument("--snr_val", type=float, help="Specify snr value")
    parser.add_argument("--symmetrized_loss_sup", type=int, help="Specify symmetrized loss sup")
    parser.add_argument("--symmetrized_loss_tomo", type=int, help="Specify symmetrized loss tomo")
    parser.add_argument("--progressive_supervision", type=int, help="Specify progressive supervision")
    parser.add_argument("--use_3d_volume_encoder", type=int, help="Specify if want to use encoder with proj of 3D vol")
    parser.add_argument("--gaussian_pyramid", type=int, help="Specify gaussian pramid")
    parser.add_argument("--down_res_in_pixel", type=float, help="min val 2.0. Pixel numbers till init_gt is downsampled. ")
    parser.add_argument("--suffix_name", type=str,default="", help="suffix added to the exp_name.")
    parser.add_argument("--ewald_radius", type=int, default=100000, help="suffix added to the exp_name.")

    args = parser.parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError("Please provide a valid .cfg file")

    with open(args.config_path, "r") as read:
        config_dict = yaml.safe_load(read)

    config = update_config(config_dict)
    config = Dict_to_Obj(config)
    config.config_path = args.config_path

    if args.snr_val is not None:
        config.snr_val = args.snr_val

    if args.symmetrized_loss_sup is not None:
        config.symmetrized_loss_sup = args.symmetrized_loss_sup

    if args.symmetrized_loss_tomo is not None:
        config.symmetrized_loss_tomo = args.symmetrized_loss_tomo

    if args.progressive_supervision is not None:
        config.progressive_supervision = args.progressive_supervision

    if args.down_res_in_pixel is not None:
        config.down_res_in_pixel= args.down_res_in_pixel

    if args.ewald_radius is not None:
        config.ewald_radius = args.ewald_radius



    config.exp_name = config.exp_name
    config.exp_name += "_ewald_" + str(config.ewald_radius)
    config.exp_name += "_sym_loss_sup_" + str(config.symmetrized_loss_sup)
    config.exp_name += "_sym_loss_tomo_" + str(config.symmetrized_loss_tomo)
    config.exp_name += "_progressive_sup_" + str(config.progressive_supervision)
    if config.init_with_gt:
        config.exp_name += "_down_pixel_" + str(config.down_res_in_pixel)

    config.exp_name += "_sidelen_" + str(config.gt_side_len)
    config.exp_name += "_projsize_" + str(config.ProjectionSize)
    config.exp_name += "_snr_" + str(config.snr_val)


    if args.suffix_name is not None:
        config.exp_name+="_"+args.suffix_name

    return config


def main():
    config = init_config()
    # dataset_generator=GenerateData(config)
    # dataset_generator.run()
    supervised_cryoposegan_wrapper = SupervisedCryoposeganWrapper(config)
    supervised_cryoposegan_wrapper.run()
    return 0, "Reconstruction completed."


if __name__ == "__main__":
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = "Error: Training failed."

    print(status_message)
    exit(retval)
