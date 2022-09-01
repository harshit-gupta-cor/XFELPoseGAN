import numpy as np
import torch
import pytorch3d.transforms
from scipy import ndimage

def axis_angle_to_quaternion(axis, theta):
    """
    Convert an angular rotation around an axis series to quaternions.
    Note: this function is a temporary fix and should  be replaced by
    pytorch3d.transforms.axis_angle_to_quaternion.
    Parameters
    ----------
    axis : torch.Tensor, size (num_pts, 3)
        axis vector defining rotation
    theta : torch.Tensor, size (num_pts)
        angle in radians defining anticlockwise rotation around axis
    Returns
    -------
    quat : torch.Tensor, size (num_pts, 4)
        quaternions corresponding to axis/theta rotations
    """
    axis /= torch.norm(axis, dim=1)[:,None]
    angle = theta / 2

    quat = torch.zeros(theta.shape[0], 4)
    quat[:,0] = torch.cos(angle)
    quat[:,1:] = torch.sin(angle)[:,None] * axis

    return quat

def save_mrc(output, data, voxel_size=None, header_origin=None):
    """
    Save numpy array as an MRC file.

    Parameters
    ----------
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    data : numpy.ndarray
        image or volume to save
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom in the header
    header_origin : numpy.recarray
        if supplied, use the origin from this header object
    """
    mrc = mrcfile.new(output, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    if header_origin is not None:
        mrc.header['origin']['x'] = float(header_origin['origin']['x'])
        mrc.header['origin']['y'] = float(header_origin['origin']['y'])
        mrc.header['origin']['z'] = float(header_origin['origin']['z'])
        mrc.update_header_from_data()
        mrc.update_header_stats()
    mrc.close()
    return


def quaternion2rot3d(quat):
    q01 = quat[:, 0] * quat[:, 1]
    q02 = quat[:, 0] * quat[:, 2]
    q03 = quat[:, 0] * quat[:, 3]
    q11 = quat[:, 1] * quat[:, 1]
    q12 = quat[:, 1] * quat[:, 2]
    q13 = quat[:, 1] * quat[:, 3]
    q22 = quat[:, 2] * quat[:, 2]
    q23 = quat[:, 2] * quat[:, 3]
    q33 = quat[:, 3] * quat[:, 3]

    # Obtain the rotation matrix
    # rotation = torch.cat(((1. - 2. * (q22 + q33)), 2. * (q12 - q03), 2. * (q13 + q02),
    #                        2. * (q12 + q03), (1. - 2. * (q11 + q33)), 2. * (q23 - q01),
    #                        2. * (q13 - q02), 2. * (q23 + q01), (1. - 2. * (q11 + q22))),
    #                        dim=-1).reshape(-1,3,3)
    rotation = torch.zeros((quat.shape[0],3, 3)).to(quat.device)
    rotation[:,0, 0] = (1. - 2. * (q22 + q33))
    rotation[:,0, 1] = 2. * (q12 - q03)
    rotation[:,0, 2] = 2. * (q13 + q02)
    rotation[:,1, 0] = 2. * (q12 + q03)
    rotation[:,1, 1] = (1. - 2. * (q11 + q33))
    rotation[:,1, 2] = 2. * (q23 - q01)
    rotation[:,2, 0] = 2. * (q13 - q02)
    rotation[:,2, 1] = 2. * (q23 + q01)
    rotation[:,2, 2] = (1. - 2. * (q11 + q22))

    return rotation


def get_preferred_orientation_quat(num_pts, sigma, base_quat=None):
    """
    Sample quaternions distributed around a given or random position in a restricted
    range in SO(3), where the spread of the distribution is determined by sigma.
    Parameters
    ----------
    num_pts : int
        number of quaternions to generate
    sigma : float
        standard deviation in radians for angular sampling
    base_quat : torch.Tensor, size (4)
        quaternion about which to distribute samples, random if None
    Returns
    -------
    quat : torch.Tensor, size (num_quats, 4)
        quaternions with preferred orientations
    """
    if base_quat is None:
        base_quat = pytorch3d.transforms.random_quaternions(1)

    R_random = quaternion2rot3d(pytorch3d.transforms.random_quaternions(num_pts))
    unitvec = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    rot_axis = torch.bmm(unitvec.repeat(num_pts, 1, 1), R_random)
    theta = sigma * torch.randn(num_pts)
    rand_axis = theta[:,None] * torch.squeeze(rot_axis, axis=1)
    #pref_quat = pytorch3d.transforms.axis_angle_to_quaternion(rand_axis)
    pref_quat = axis_angle_to_quaternion(rand_axis, theta)
    quat = pytorch3d.transforms.quaternion_multiply(pref_quat, base_quat)

    return quat


def calc_fsc(rho1, rho2, side):
    """ Calculate the Fourier Shell Correlation between two electron density maps."""
    df = 1.0 / side
    n = rho1.shape[0]
    qx_ = np.fft.fftfreq(n) * n * df
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qx_max = qx.max()
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    F1 = np.fft.fftn(rho1)
    F2 = np.fft.fftn(rho2)
    numerator = ndimage.sum(np.real(F1 * np.conj(F2)), labels=qbin_labels,
                            index=np.arange(0, qbin_labels.max() + 1))
    term1 = ndimage.sum(np.abs(F1) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    term2 = ndimage.sum(np.abs(F2) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    denominator = (term1 * term2) ** 0.5
    FSC = numerator / denominator
    qidx = np.where(qbins < qx_max)
    return np.vstack((qbins[qidx], FSC[qidx])).T


def fsc2res(fsc, cutoff=0.5, return_plot=False):
    """Calculate resolution from the FSC curve using the cutoff given.
    fsc - an Nx2 array, where the first column is the x axis given as
          as 1/resolution (angstrom).
    cutoff - the fsc value at which to estimate resolution, default=0.5.
    return_plot - return additional arrays for plotting (x, y, resx)
    """
    x = np.linspace(fsc[0, 0], fsc[-1, 0], 1000)
    y = np.interp(x, fsc[:, 0], fsc[:, 1])
    if np.min(fsc[:, 1]) > cutoff:
        # if the fsc curve never falls below zero, then
        # set the resolution to be the maximum resolution
        # value sampled by the fsc curve
        resx = np.max(fsc[:, 0])
        resn = float(1. / resx)
        # print("Resolution: < %.1f A (maximum possible)" % resn)
    elif np.max(fsc[:, 1]) < cutoff:
        resx = np.min(fsc[:, 0])
        resn = np.nan
    else:
        idx = np.where(y >= cutoff)
        # resi = np.argmin(y>=0.5)
        # resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
        resx = np.max(x[idx])
        resn = float(1. / resx)
        # print("Resolution: %.1f A" % resn)
    if return_plot:
        return resn, x, y, resx
    else:
        return resn


def add_flipped_rotmat(pred_rotmat):
    euler_angles = matrix_to_euler_angles(pred_rotmat, 'ZYZ')  # B, 3
    additional_pi = torch.tensor([np.pi, 0., 0.]).reshape(1, 3).to(euler_angles.device)
    euler_angles = euler_angles + additional_pi
    pred_rotmat_flipped = euler_angles_to_matrix(euler_angles, 'ZYZ')
    return torch.cat([pred_rotmat, pred_rotmat_flipped], dim=0)


def rotate_volume(vol, quat):
    """
    Rotate copies of the volume by the given quaternions.

    Parameters
    ----------
    vol : torch.Tensor, shape (n,n,n)
        volume to be rotated
    quat : torch.Tensor, shape (n_quat,4)
        orientations to apply to the volume

    Returns
    -------
    rot_vol : torchTensor, shape (n_quat,n,n,n)
        rotated copies of volume
    """
    sidelen = vol.shape[0]
    lincoords = torch.linspace(-1., 1., sidelen)
    [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
    vol_coords = torch.stack([Y, X, Z], dim=-1).reshape(-1, 3)

    rotmat = quaternion2rot3d(quat)
    n_quat = rotmat.shape[0]
    rot_vol_coords = torch.bmm(vol_coords.repeat(n_quat, 1, 1), rotmat)  # --> Batch, sidelen^3, 3

    rot_vol = torch.nn.functional.grid_sample(vol.repeat(n_quat, 1, 1, 1, 1),  # = (Batch, C,D,H,W)
                                              rot_vol_coords[:, None, None, :, :],
                                              align_corners=True)
    rot_vol = rot_vol.reshape(n_quat, sidelen, sidelen, sidelen)

    return rot_vol


def score_orientations(mrc1, mrc2, quat):
    """
    Compute the Pearson correlation coefficient between the input volumes
    after rotating the first volume by the given quaternions.

    Parameters
    ----------
    mrc1 : torch.Tensor, shape (n,n,n)
        volume to be rotated
    mrc2 : torch.Tensor, shape (n,n,n)
        volume to be held fixed
    quat : torch.Tensor, shape (n_quat,4)
        orientations to apply to mrc2

    Returns
    -------
    ccs : torch.Tensor, shape (n_quat)
        correlation coefficients associated with quat
    """
    rmrc1 = rotate_volume(mrc1, quat)
    rmrc1_flat = rmrc1.flatten(start_dim=1)
    mrc2_flat = mrc2.unsqueeze(axis=0).flatten(start_dim=1)

    vx = rmrc1_flat - rmrc1_flat.mean(axis=-1)[:, None]
    vy = mrc2_flat - mrc2_flat.mean(axis=-1)[:, None]
    numerator = torch.sum(vx * vy, axis=1)
    denom = torch.sqrt(torch.sum(vx ** 2, axis=1)) * torch.sqrt(torch.sum(vy ** 2, axis=1))
    return numerator / denom


def scan_orientations(mrc1, mrc2, zoom=1, sigma=0, n_iterations=10, n_search=420):
    """
    Find the quaternion and its associated score that best aligns volume
    mrc1 to mrc2. Candidate orientations are scored based on the Pearson
    correlation coefficient. First a coarse search is performed, followed
    by a series of increasingly fine searches in angular space.

    Parameters
    ----------
    mrc1 : torch.Tensor, shape (n,n,n)
        volume to be rotated
    mrc2 : torch.Tensor, shape (n,n,n)
        volume to be held fixed
    zoom : float, default 1
        if not 1, sample by which to up or downsample volume
    sigma : int, default 0
        sigma of Gaussian filter to apply to each volume
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    Returns
    -------
    opt_q : torch.Tensor, shape (1,4)
        quaternion to apply to mrc1 to align it with mrc2
    prev_score : float
        cross-correlation between aligned mrc1 and mrc2
    """
    # perform a coarse alignment to start
    quat = pytorch3d.transforms.random_quaternions(n_search)
    ccs = score_orientations(mrc1, mrc2, quat)
    opt_q, prev_score = quat[torch.argmax(ccs)], torch.max(ccs)

    # perform a series of fine alignment, ending if CC no longer improves
    sigmas = 2 - 0.2 * torch.arange(1, 10)
    for n in range(1, n_iterations):
        print("Iteration " + str(n))
        quat = get_preferred_orientation_quat(n_search - 1, float(sigmas[n - 1]), base_quat=opt_q)
        quat = torch.vstack((opt_q, quat))
        ccs = score_orientations(mrc1, mrc2, quat)
        if torch.max(ccs) < prev_score:
            break
        else:
            opt_q = quat[torch.argmax(ccs)]
        # print(torch.max(ccs), opt_q) # useful for debugging
        prev_score = torch.max(ccs)
        print("Score " + str(prev_score))

    return opt_q, prev_score


def align_volumes(mrc1, mrc2, zoom=1, sigma=0, n_iterations=10, n_search=420, output=None, voxel_size=None):
    """
    Find the quaternion that best aligns volume mrc1 to mrc2. Volumes are
    optionally preprocessed by up / downsampling and applying a Gaussian
    filter.

    Parameters
    ----------
    mrc1 : torch.Tensor, shape (n,n,n)
        volume to be rotated
    mrc2 : torch.Tensor, shape (n,n,n)
        volume to be held fixed
    zoom : float, default 1
        if not 1, sample by which to up or downsample volume
    sigma : int, default 0
        sigma of Gaussian filter to apply to each volume
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom for output
    Returns
    -------
    opt_q : torch.Tensor, shape (1,4)
        quaternion to apply to mrc1 to align it with mrc2
    r_vol : np.array, shape (n,n,n)
        aligned array
    """
    # copy the input volume if saving later
    mrc1_original = torch.clone(mrc1)

    # optionally up/downsample volumes
    if zoom != 1:
        mrc1 = torch.Tensor(ndimage.zoom(np.array(mrc1), (zoom, zoom, zoom)))
        mrc2 = torch.Tensor(ndimage.zoom(np.array(mrc2), (zoom, zoom, zoom)))

    # optionally apply a Gaussian filter to volumes
    if sigma != 0:
        mrc1 = torch.Tensor(ndimage.gaussian_filter(np.array(mrc1), sigma=sigma))
        mrc2 = torch.Tensor(ndimage.gaussian_filter(np.array(mrc2), sigma=sigma))

    # evaluate both hands
    opt_q1, cc1 = scan_orientations(mrc1, mrc2, n_iterations, n_search)
    opt_q2, cc2 = scan_orientations(torch.flip(mrc1, [0, 1, 2]), mrc2, n_iterations, n_search)
    if cc1 > cc2:
        print("Not chiral")
        opt_q, cc, invert = opt_q1, cc1, False
    else:
        print("Chiral")
        opt_q, cc, invert = opt_q2, cc2, True

    print(f"Alignment CC is: {cc:.3f}")
    if invert:
        mrc1_original = torch.flip(mrc1_original, [0, 1, 2])
    r_vol = rotate_volume(mrc1_original, torch.unsqueeze(opt_q, axis=0))
    if output is not None:
        save_mrc(output, np.array(r_vol[0]), voxel_size=voxel_size)

    return opt_q, np.array(r_vol[0])