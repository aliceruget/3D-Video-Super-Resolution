
import numpy as np
from scipy.ndimage import median_filter

def create_histogram_from_binary_nods(pixel_map,num_bins):
    """
    Create histogram of photon counts from depth maps
    Args:
        pixels_map (numpy.ndarray): The input depth maps with dimensions (Nx, Ny, num_maps). Normalised between 0 and self.num_bins. 

    Returns:
        combined_histograms (numpy.ndarray): The 3D histogram with dimensions (Nx, Ny, num_bins), sum of the num_maps depth maps
    """
    if len(pixel_map.shape) ==3:
        num_maps = pixel_map.shape[2]
    else:
        num_maps = 1

    Nx, Ny = pixel_map.shape[0], pixel_map.shape[1]  

    pixel_map[pixel_map>num_bins-1] = num_bins-1
    
    # IRF matrix unitary 
    pixel_map = pixel_map.astype(int)
    IRF_matrix = np.zeros((num_bins, num_bins))
    np.fill_diagonal(IRF_matrix, 1)
    
    histograms = IRF_matrix[pixel_map.ravel()].reshape(Nx, Ny, num_maps, num_bins)
    # No depth at 0 
    histograms[pixel_map == 0] = 0
    # Sum over the num_maps
    combined_histograms = histograms.sum(axis=2)
    
    return combined_histograms

def dtof_hist_with_img(d, img,temp_res):
        # print('dtof_hist')
        """
        generate full dToF histogram using Eq.1 in paper
        """
        pitch = 16
        # temp_res = 100
        albedo = img
        hist = np.zeros((d.shape[0] // pitch, d.shape[1] // pitch, temp_res))

        for ii in range(d.shape[0] // pitch):
            for jj in range(d.shape[1] // pitch):
                ch, cw = ii * pitch, jj * pitch
                albedo_block = albedo[ch : ch + pitch, cw : cw + pitch]
  
                d_block = d[ch : ch + pitch, cw : cw + pitch]
                idx = np.round(d_block * (temp_res - 1)).reshape(-1)
                r =  (albedo_block / (1e-3 + d_block**2)).reshape(-1)
                r[d_block.reshape(-1) == 0] = 0
                idx = np.concatenate((idx, np.asarray([0, temp_res - 1]))).astype(
                    np.int64
                )
                r = np.concatenate((r, np.asarray([0, 0]))).astype(np.float32)
                hist[ii, jj] = np.bincount(idx, weights=r)
        return hist

def replace_zeros_with_median(matrix):
    # Create a mask for zero values
    zero_mask = matrix == 0

    # Use a median filter to compute the median of neighboring pixels
    median_filtered = median_filter(matrix, size=15)

    # Replace zeros with the median of neighboring pixels
    matrix[zero_mask] = median_filtered[zero_mask]
    return matrix

def center_of_mass_test(histogram):
    Nx_LR, Ny_LR, Nbins = histogram.shape

    # Find the position of the maximum value along the third axis
    pos_max = np.argmax(histogram, axis=2)

    # Define the range of indices
    index_range = np.arange(Nbins)

    # Create masks for valid ranges (index_bin > 0)
    valid_range_mask = (pos_max > 0) & (pos_max < Nbins - 1)

    # Calculate b (median along the third axis)
    b = np.median(histogram, axis=2)

    # Create the range_center_of_mass for all elements
    offsets = np.array([-1, 0, 1])
    range_center_of_mass = pos_max[..., np.newaxis] + offsets

    # Ensure the ranges are within bounds
    range_center_of_mass = np.clip(range_center_of_mass, 0, Nbins - 1)

    # Gather histogram values for the range_center_of_mass
    hist_values = np.take_along_axis(histogram, range_center_of_mass, axis=2)

    # Calculate max_diff (histogram values minus b, clipped at 0)
    max_diff = np.maximum(hist_values - b[..., np.newaxis], 0)

    # Calculate numerator and denominator
    numerator = np.sum(range_center_of_mass * max_diff, axis=2)
    denominator = np.sum(max_diff, axis=2)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depth_image = np.true_divide(numerator, denominator)
        depth_image[denominator == 0] = 0

    return np.float32(depth_image)