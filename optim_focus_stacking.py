from pickletools import uint8
import cv2, os, glob, numpy as np, cProfile, pstats
from multiprocessing import Process, Manager
from natsort import natsorted
import time

class MultiProcess:
    def __init__(self):
        self.MANAGER = Manager()
        self.MP_LIST = self.MANAGER.dict()
        self.PROCESSES = []
        self.ITER = None
        self.TARGET_FUNC = None

    def run_on_multiprocessors(self):
        for i in range(len(self.ITER)):
            p = Process(target=eval(self.TARGET_FUNC), args=(i, self.ITER[i], self.MP_LIST))
            self.PROCESSES.append(p)
            p.start()
        for process in self.PROCESSES:
            process.join()
        return np.array(self.MP_LIST.values())

class Stacks(MultiProcess):
    def __init__(self, input_path, output_path):
        super().__init__()
        self.INPUT_PATH = input_path
        self.OUTPUT_PATH = output_path
        self.FILENAMES = self.read_images_from_path()
        self.ITER = self.FILENAMES
        self.TARGET_FUNC = 'self.load_image_multiprocess'
        self.IMAGES = self.run_on_multiprocessors()

    def load_image_multiprocess(self, i, filename, images_list):
        image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
        images_list[i] = image

    def read_images_from_path(self):
        file_names = natsorted([img for img in glob.glob(os.path.join(self.INPUT_PATH, '*.png'))])
        return file_names

class LaplacianPyramid():
    def __init__(self, images, input_path, output_path, pyramid_depth):
        self.INPUT_PATH = input_path
        self.OUTPUT_PATH = output_path
        self.PYRAMID_DEPTH = pyramid_depth
        self.LAP_PYR_LIST_BASE = self.lap_pyramid_initiate(images)

    def lap_pyramid_initiate(self, images):
        # images = Stacks(self.INPUT_PATH, self.OUTPUT_PATH).IMAGES
        list_lap_pyramids = []
        
        for img in images:
            lap_pyr = self.get_laplacian_pyramid(img, self.PYRAMID_DEPTH)
            base = lap_pyr[-1]
            lap_pyr = lap_pyr[:-1]
            list_lap_pyramids.append(lap_pyr)
        return [np.array(list_lap_pyramids, dtype=object), base]

    def get_laplacian_pyramid(self, curr_img, N):
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

class RegionalFusion():
    def __init__(self, images, input_path, output_path, pyramid_depth, kernel_size):
        self.INPUT_PATH = input_path
        self.OUTPUT_PATH = output_path
        self.PYRAMID_DEPTH = pyramid_depth
        self.KERNEL_SIZE = kernel_size
        list_lap_pyramids, base = self.extract_pyramids_and_base(images)
        
        LP_f = self.N_level_fusion(list_lap_pyramids)
        LP_f = self.other_levels_fusion(list_lap_pyramids, LP_f)
        
        LP_f = np.array(LP_f, dtype=object)
        LP_f = np.flip(LP_f)
        
        
        # 3: Reconstruct final laplacian pyramid LP_f back to original image!
        # get the top-level of the gaussian pyramid
        fused_img = cv2.pyrUp(base, dstsize=(LP_f[-1].shape[1], LP_f[-1].shape[0])).astype(np.int16)

        for i in reversed(range(self.PYRAMID_DEPTH)):
            # combine with laplacian pyramid at the level
            fused_img += LP_f[i]
            if i != 0:
                fused_img = cv2.pyrUp(fused_img, dstsize=(LP_f[i-1].shape[1], LP_f[i-1].shape[0]))

        self.CANVAS = fused_img


    def extract_pyramids_and_base(self, images):
        temp_list = LaplacianPyramid(images, self.INPUT_PATH, self.OUTPUT_PATH, self.PYRAMID_DEPTH).LAP_PYR_LIST_BASE
        return temp_list[0], temp_list[1]

    def other_levels_fusion(self, list_lap_pyramids, LP_f):
        # 2.b: Fusing other levels of laplacian pyramid (N-1 to 0)
        
        for l in reversed(range(0, self.PYRAMID_DEPTH-1)):
            
            start_time = time.time()
            print('Pyramid Level: ', l)
            RE_max_idx = self.region_energy_map(l, list_lap_pyramids)
            LP_l = self.compute_LP_l(l, list_lap_pyramids, RE_max_idx)
            LP_f.append(LP_l)
            print("--- %s seconds ---" % (time.time() - start_time))

        return LP_f     
    
    def region_energy_map(self, l, list_lap_pyramids):
        
        # region energey map for level l
        RE_l = np.array([self.region_energy(lap) for lap in list_lap_pyramids[:, l]], dtype='float32')
        RE_max_idx = np.argmax(RE_l,0)

        return RE_max_idx

    def compute_LP_l(self, l, list_lap_pyramids, RE_max_idx):

        stacks_at_l = list_lap_pyramids[:,l]
        stacks_at_l = np.reshape(np.concatenate(stacks_at_l), (len(stacks_at_l), stacks_at_l[0].shape[0], stacks_at_l[0].shape[1]))
        LP_l = stacks_at_l[RE_max_idx, np.arange(stacks_at_l[0].shape[0])[:,None], np.arange(stacks_at_l[0].shape[1])]

        return LP_l

    def N_level_fusion(self, list_lap_pyramids):    
        D_N = self.get_D_N(list_lap_pyramids)
        E_N = self.get_E_N(list_lap_pyramids)
        LP_f = []
        LP_N = np.zeros(list_lap_pyramids[0, -1].shape).astype('int16')
        for m in range(LP_N.shape[0]):
            for n in range(LP_N.shape[1]):
                D_max_idx = np.argmax(D_N[:, m, n]).astype('int16')
                E_max_idx = np.argmax(E_N[:, m, n]).astype('int16')
                D_min_idx = np.argmin(D_N[:, m, n]).astype('int16')
                E_min_idx = np.argmin(E_N[:, m, n]).astype('int16')
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
        return LP_f

    def get_D_N(self, list_lap_pyramids):
        return np.array([self.deviation(lap, kernel_size) for lap in list_lap_pyramids[:, -1]], dtype=np.float32)

    def get_E_N(self, list_lap_pyramids):
        return np.array([self.entropy(lap, kernel_size) for lap in list_lap_pyramids[:, -1]], dtype=np.float32)

    def generating_kernel(a):
        kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
        return np.outer(kernel, kernel)
        
    def convolve(self, image, kernel=generating_kernel(0.4)):
        return cv2.filter2D(src=image.astype(np.float64), ddepth=-1, kernel=np.flip(kernel))
    
    def region_energy(self, laplacian):
        return self.convolve(np.square(laplacian))    
    
    def entropy(self, image, kernel_size):
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
    
    def deviation(self, image, kernel_size):
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





profiler = cProfile.Profile()
pyramid_depth = 10
kernel_size = 5
input_path='/home/hassan/mnt/Linux/repos/focus_stacking/test_imgs_1/Gold'
output_path='/home/hassan/mnt/Linux/repos/focus_stacking/test_imgs/Gold'

profiler.enable()

images = Stacks(input_path, output_path).IMAGES
canvas = RegionalFusion(images, input_path, output_path, pyramid_depth, kernel_size).CANVAS

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

cv2.imwrite('output.jpg', canvas)