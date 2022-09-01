"""run dataset generator and save it."""
import os
import torch
import starfile
import mrcfile
from utils import mean_snr_calculator, dict2cuda,  Dict_to_Obj
import pytorch3d
from src.dataio import dataloader
"""Module to generate and save dataset (including metadata) in the output directory."""


class DatasetGenerator(torch.nn.Module):
    """class to generate and save dataset (including metadata) in the output directory.
    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generation and simulator.
    """

    def __init__(self, config):
        super(DatasetGenerator, self).__init__()

        self.config = config

        self.gt_loader, self.noise_loader, dataset = dataloader(Dict_to_Obj(config.copy()))

        if not self.config.simulated:
            self.config = dataset.get_df_optics_params(self.config)

        self.dataframe = []
        self.init_dir()  # initializing the output folder
        self.print_statements()

    def run(self):
        """Generate a chunk of projection and save it."""
        datalist = []
        iteration=-1
        for epoch in range(1):

            iter_loader = zip(self.gt_loader, self.noise_loader)
            iteration += 1
            print(iteration)
            gt_data, _ = next(iter_loader)



        rot_params= gt_data["rot_params"]
        ctf_params= gt_data["ctf_params"] if "ctf_params" in gt_data else None
        shift_params = gt_data["shift_params"] if "shift_params" in gt_data else None
        projections = self.simulator(rot_params, ctf_params, shift_params)
        save_mrc(self.config.output_path, projections, iterations)
        datalist = starfile_data(
            datalist, rot_params, ctf_params, shift_params, iterations, self.config
        )

        print(f"Saving star file with the parameters of the generated dataset..")
        write_metadata_to_starfile(
            self.config.output_path, datalist, self.config, save_name="Simulated"
        )

    def init_dir(self):
        """Make the output directory and puts the path in the config.output_path."""
        self.config.output_path = os.path.join(os.getcwd(), self.config.output_path)

        self.config.output_path = os.path.join(self.config.output_path, "Datasets")
        if os.path.exists(self.config.output_path) is False:
            os.mkdir(self.config.output_path)

        self.config.output_path = os.path.join(
            self.config.output_path, self.config.name
        )
        if os.path.exists(self.config.output_path) is False:
            os.mkdir(self.config.output_path)

