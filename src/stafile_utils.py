import os
import starfile
import numpy as np
import pandas as pd
def check_star_file(path):
    """Check if the starfile exists and is valid."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Input star file doesn't exist!")
    if ".star" not in path:
        raise FileExistsError("Input star file is not a valid star file!")


def starfile_opticsparams(config):
    """Update attributes of config with metadata from input starfile.
    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.
    Returns
    -------
    config: class
    """
    check_star_file(config.input_starfile_path)
    df = starfile.read(config.input_starfile_path)
    config.side_len = df["optics"]["rlnImageSize"][0]
    config.kv = df["optics"]["rlnVoltage"][0]
    config.pixel_size = df["optics"]["rlnImagePixelSize"][0]
    config.cs = df["optics"]["rlnSphericalAberration"][0]
    config.amplitude_contrast = df["optics"]["rlnAmplitudeContrast"][0]
    if hasattr(df["optics"], "rlnCtfBfactor"):
        config.b_factor = df["optics"]["rlnCtfBfactor"][0]

    return config


def return_names(config):
    """Return relion-convention names of metadata for starfile.
    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.
    Returns
    -------
    names: list of str
    """
    names = [
        "__rlnImageName",
        "__rlnAngleRot",
        "__rlnAngleTilt",
        "__rlnAnglePsi",
    ]
    if config.shift:
        names += ["__rlnOriginX", "__rlnOriginY"]
    if config.ctf:
        names += ["__rlnDefocusU", "__rlnDefocusV", "__rlnDefocusAngle"]


    return names


def starfile_data(datalist, rot_params, ctf_params, shift_params, iterations, config):
    """Append the datalist with the parameters of the simulator.
    Parameters
    ----------
    rot_params: dict of type str to {tensor}
        Dictionary of rotation parameters for a projection chunk
    ctf_params: dict of type str to {tensor}
        Dictionary of Contrast Transfer Function (CTF) parameters
         for a projection chunk
    shift_params: dict of type str to {tensor}
        Dictionary of shift parameters for a projection chunk
    iterations: int
        iteration number of the loop. Used in naming the mrcs file.
    config: class
         class containing parameters of the dataset generator.
    Returns
    -------
    datalist: list
        list containing the metadata of the projection chunks.
        This list is then used to save the starfile.
    """
    image_name = [
        str(idx).zfill(3) + "@" + str(iterations).zfill(4) + ".mrcs"
        for idx in range(config.batch_size)
    ]

    for num in range(config.batch_size):
        list_var = [
            image_name[num],
            rot_params["relion_angle_rot"][num].item(),
            rot_params["relion_angle_tilt"][num].item(),
            rot_params["relion_angle_psi"][num].item(),
        ]
        if shift_params:
            list_var += [
                shift_params["shift_x"][num].item(),
                shift_params["shift_y"][num].item(),
            ]
        if ctf_params:
            list_var += [
                1e4 * ctf_params["defocus_u"][num].item(),
                1e4 * ctf_params["defocus_v"][num].item(),
                np.degrees(ctf_params["defocus_angle"][num].item()),
            ]

        list_var += [
            config.kv,
            config.pixel_size,
            config.cs,
            config.amplitude_contrast,
            config.b_factor,
        ]
        datalist.append(list_var)
    return datalist

def save_starfile_cryoem_convention(output_path, datalist, config, save_name):
    """Save the metadata in a starfile in the output directory.
    Parameters
    ----------
    output_path: str
        path to save starfile
    datalist: list
         list containing data set generation variables
    config: class
        class containing bool values
        ctf: bool
            indicates if the CTF effect is to be used in the forward model
        shift: bool
            indicates if the shift operator is to be used in the forward model.
    """
    cryoem_variable_names = return_names(config)
    save_starfile(output_path, datalist, cryoem_variable_names, save_name)


def save_starfile(output_path, datalist, variable_names, save_name):
    """Save the metadata in a starfile in the output directory.
    Parameters
    ----------
    output_path: str
        path to save starfile
    datalist: list
        list containing data set generation variables
    variable_names: list of str
        list containing name of the variables contained in the datalist
    save_name: str
        name of the starfile to be saved.
    """
    indices = list(range(len(datalist)))
    df = pd.DataFrame(
        data=datalist,
        index=indices,
        columns=(variable_names),
    )
    starfile.write(df, os.path.join(output_path, save_name + ".star"), overwrite=True)
