import os
from utils import Dict_to_Obj
from src.dataio import dataloader, SimulatedDataLoader
import mrcfile
import numpy as np
class GenerateData():
    def __init__(self, config):
        super(GenerateData, self).__init__()

        self.config = config.copy()

        self.config = Dict_to_Obj(self.config)
        self.config.multires = False
        print(f"batchsize for dataio {self.config.batch_size}")



    def need_generate(self):
        dataset_name=self.config.protein+"_snr_"+str(self.config.snr_val)+"_projsize_"+str(self.config.gt_side_len)
        self.output_path="./datasets/"+dataset_name
        if os.path.exists(self.output_path):
            print(f"The dataset aleady exists at {self.output_path}")
            return False
        else :
            os.makedirs(self.output_path)
            print(f"Generating data and saving at {self.output_path}")
            return True


    def run(self):


        if  self.need_generate():
            self.gt_loader=SimulatedDataLoader(self.config)
        else:
            return 0

        mrc_vol = mrcfile.new(self.output_path + "/gt_vol.mrc", overwrite=True)
        mrc_vol.set_data(self.gt_loader.sim.projector.make_vol().cpu().numpy())
        mrc_vol.close()


        per_epoch_iteration = len(self.gt_loader)
        shape=(per_epoch_iteration * self.config.batch_size, 1, self.config.gt_side_len, self.config.gt_side_len)
        rotmat = np.zeros((per_epoch_iteration * self.config.batch_size, 3, 3 ))
        mrcs=mrcfile.new_mmap(self.output_path+"/gt_data.mrcs", shape=shape, mrc_mode=2, overwrite=True)
        mrcs_rotmat = mrcfile.new(self.output_path + "/gt_data_rotmat.mrc", overwrite=True)

        print("Generating data")
        for i in range(per_epoch_iteration):
            print(100*float(i)/per_epoch_iteration)
            gt_data = next(self.gt_loader)
            ind=i*self.config.batch_size
            mrcs.data[ind:ind+self.config.batch_size]=gt_data["proj"].cpu().numpy()
            rotmat[ind:ind+self.config.batch_size]=gt_data["rotmat"].cpu().numpy()

        mrcs_rotmat.set_data(np.float32(rotmat))
        mrcs_rotmat.close()
        mrcs.close()

        print("Done")
        return 1