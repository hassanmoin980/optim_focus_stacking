from pickletools import uint8
import cv2, os, glob, numpy as np, cProfile, pstats
from multiprocessing import Process, Manager
from natsort import natsorted
import gc

def load_image_multiprocess(i, filename, images_list):
    image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
    images_list[i] = image

def extract_roi(images, x0, y0, w, h):
    return np.array([img[y0:y0+h, x0:x0+w] for img in images])

def lap_focus_stacking(images, N=5, kernel_size=5):

    def generating_kernel(a):
        kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
        return np.outer(kernel, kernel)
        
    def convolve(image, kernel=generating_kernel(0.4)):
        return cv2.filter2D(src=image.astype(np.float64), ddepth=-1, kernel=np.flip(kernel))
    
    def region_energy(laplacian):
        return convolve(np.square(laplacian))    
    
    def entropy(image, kernel_size):
        def get_probabilities(gray_image):
            levels, counts = np.unique(gray_image.astype(np.uint8), return_counts = True)
            probabilities = np.zeros((256,), dtype=np.float64)
            probabilities[levels] = counts.astype(np.float64) / counts.sum()
            return probabilities
        
        def _area_entropy(area, probabilities):
            levels = area.flatten()
            return -1. * (levels * np.log(probabilities[levels])).sum()
        
        probabilities = get_probabilities(image)
        pad_amount = int((kernel_size - 1) / 2)
        padded_image = cv2.copyMakeBorder(image,pad_amount,pad_amount,pad_amount,pad_amount,cv2.BORDER_REFLECT101)
        entropies = np.zeros(image.shape[:2], dtype=np.float64)
        offset = np.arange(-pad_amount, pad_amount + 1)
        for row in range(entropies.shape[0]):
            for column in range(entropies.shape[1]):
                area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
                entropies[row, column] = _area_entropy(area, probabilities)

        return entropies
    
    def deviation(image, kernel_size):
        def _area_deviation(area):
            average = np.average(area).astype(np.float64)
            return np.square(area - average).sum() / area.size

        pad_amount = int((kernel_size - 1) / 2)
        padded_image = cv2.copyMakeBorder(image,pad_amount,pad_amount,pad_amount,pad_amount,cv2.BORDER_REFLECT101)
        deviations = np.zeros(image.shape[:2], dtype=np.float64)
        offset = np.arange(-pad_amount, pad_amount + 1)
        for row in range(deviations.shape[0]):
            for column in range(deviations.shape[1]):
                area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
                deviations[row, column] = _area_deviation(area)

        return deviations

    def get_laplacian_pyramid(curr_img, N):
        lap_pyramids = []
        gaussian_pyramids = [curr_img,]

        # for every N pyramid level
        for i in range(N):
            down = cv2.pyrDown(curr_img)
            gaussian_pyramids.append(down)
            up = cv2.pyrUp(down, dstsize=(curr_img.shape[1], curr_img.shape[0]))
            lap = curr_img - up.astype('int16')
            lap_pyramids.append(lap)
            curr_img = down

            # top level laplacian be a gaussian downsampled
            if i == N-1:
                lap_pyramids.append(curr_img)
        return lap_pyramids    
    
    # 1: Generating Array of Laplacian Pyramid
    list_lap_pyramids = []
    for img in images:
        # Extract Laplacian and Gaussian Pyramids
        lap_pyr = get_laplacian_pyramid(img, N)
        base = lap_pyr[-1]
        lap_pyr = lap_pyr[:-1]
        list_lap_pyramids.append(lap_pyr)
    list_lap_pyramids = np.array(list_lap_pyramids, dtype=object)

    
    LP_f = []
    
    # 2: Regional Fusion using Laplacian Pyramids
    # fuse level=N laplacian pyramid, D=deviation, E=entropy

    D_N = np.array([deviation(lap, kernel_size) for lap in list_lap_pyramids[:, -1]])
    E_N = np.array([entropy(lap, kernel_size) for lap in list_lap_pyramids[:, -1]])

    # 2.a: Init level N fusion canvas
    LP_N = np.zeros(list_lap_pyramids[0, -1].shape)

    for m in range(LP_N.shape[0]):
        for n in range(LP_N.shape[1]):
            D_max_idx = np.argmax(D_N[:, m, n])
            E_max_idx = np.argmax(E_N[:, m, n])
            D_min_idx = np.argmin(D_N[:, m, n])
            E_min_idx = np.argmin(E_N[:, m, n])
            # if the image maximizes BOTH the deviation and entropy, use the pixel from that image
            if D_max_idx == E_max_idx:
                LP_N[m, n] = list_lap_pyramids[D_max_idx, -1][m, n]
            # if the image minimizes BOTH the deviation and entropy, use the pixel from that image
            elif D_min_idx == E_min_idx: 
                LP_N[m, n] = list_lap_pyramids[D_min_idx, -1][m, n]
            # else average across all images
            else:
                for k in range(list_lap_pyramids.shape[0]):
                    LP_N[m, n] += list_lap_pyramids[k, -1][m, n]
                LP_N[m, n] /= list_lap_pyramids.shape[0]

    LP_f.append(LP_N)

    # 2.b: Fusing other levels of laplacian pyramid (N-1 to 0)
    for l in reversed(range(0, N-1)):
        # level l final laplacian canvas
        LP_l = np.zeros(list_lap_pyramids[0, l].shape)

        # region energey map for level l
        RE_l = np.array([region_energy(lap) for lap in list_lap_pyramids[:, l]], dtype=object)

        for m in range(LP_l.shape[0]):
            for n in range(LP_l.shape[1]):
                RE_max_idx = np.argmax(RE_l[:, m, n])
                LP_l[m, n] = list_lap_pyramids[RE_max_idx, l][m, n]

        LP_f.append(LP_l)

    LP_f = np.array(LP_f, dtype=object)
    LP_f = np.flip(LP_f)

    # 3: Reconstruct final laplacian pyramid LP_f back to original image!
    # get the top-level of the gaussian pyramid
    fused_img = cv2.pyrUp(base, dstsize=(LP_f[-1].shape[1], LP_f[-1].shape[0])).astype(np.float64)

    for i in reversed(range(N)):
        # combine with laplacian pyramid at the level
        fused_img += LP_f[i]
        if i != 0:
            fused_img = cv2.pyrUp(fused_img, dstsize=(LP_f[i-1].shape[1], LP_f[i-1].shape[0]))
    
    return fused_img


# 0: Define Parameters --------------------------------------------------------------------
profiler = cProfile.Profile()
dir_path = '/home/hassan/mnt/Linux/repos/focus_stacking/test_imgs/Gold'
output_name = 'Gold_stacked.png'
pyramid_depth = 10
kernel_size = 5

# Region of Interest
x0 = 8000
y0 = 2000
w = 1000
h = 1000 



# 1: Read File Names --------------------------------------------------------------------
file_names = natsorted([img for img in glob.glob(os.path.join(dir_path, '*.png'))])

# 2: Load All Images --------------------------------------------------------------------
manager = Manager()
images_list = manager.dict()
processes = []
for i in range(len(file_names)):
    p = Process(target=load_image_multiprocess, args=(i, file_names[i], images_list))
    processes.append(p)
    p.start()
for process in processes:
    process.join()
images = np.array(images_list.values())

# ---------------------------------------------------------------------------------------
# 2.a: Extracting small area from main image
# images = extract_roi(images, x0, y0, w, h)

# 3: Laplacian Pyramid Fusion Focus Stacking --------------------------------------------
canvas = np.array(lap_focus_stacking(images, N=pyramid_depth, kernel_size=kernel_size))

profiler.enable()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

# # 4: Write to file
cv2.imwrite(output_name, canvas)


# # profiler.enable()
# # profiler.disable()
# # stats = pstats.Stats(profiler).sort_stats('cumtime')
# # stats.print_stats()
