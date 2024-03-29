"""
    Define:
        + binary image: image with value 0s and 1s, since it is needed for convolution step
        + proper image: image with values 0s and 255s
                        You can use `Helper.binaryToProperImage(img)` to 
                        turn binary image into proper image.
        + grayscale image: image with value ranging in [0, 255], with np.uint8
                           Can be directly shown by cv.imshow()
"""

from functools import reduce
import cv2 as cv
import numpy as np
from typing import *
import random
import pickle

from contextlib import contextmanager
from timeit import default_timer
import argparse
import sys

from pathlib import Path

from scipy.signal import convolve2d
from tqdm import tqdm

# Helper class containing functions for testing
class Helper:
    @classmethod
    def makeSecret(cls, alphabet='S'):
        # make an image with main secret as black, and main background as white
        # note that the result will be binary
        img = np.zeros((128, 128), np.uint8)
        img = cv.putText(img, f"{alphabet}", (32, 96), cv.FONT_HERSHEY_SIMPLEX,
        3, 1, 10, cv.LINE_8)
        img = 1 - img
        return img

    @classmethod
    @contextmanager
    def elapsed_timer(cls):
        # use like this: 
        # ```
        #    with elapsed_timer() as elapsed:
        #        # do stuff, if want to get current time do `elapsed()`
        #    print("Task done in %.2f seconds" % elapsed() )
        # ```
        start = default_timer()
        elapser = lambda: default_timer() - start
        yield lambda: elapser()
        end = default_timer()
        elapser = lambda: end-start
    
    @classmethod
    def binaryToProperImage(cls, img):
        # turn image full of 0s and 1s to proper, show-able image
        img = img.copy()
        img[img > 0] = 255
        return img.astype(np.uint8)

    @classmethod
    def properToBinaryImage(cls, img):
        # turn image full of 0s and 1s to proper, show-able image
        img = img.copy()
        img[img > 0] = 1
        return img
    

class VacHvcAlgorithm:
    def __init__(self, secret_img):
        # secret_img must be binary
        # secret_img, img1 and img2 must be same size
        self.secret_img = secret_img

        self.BLACK = 0
        self.WHITE = 1

        self.B_region = (secret_img == self.BLACK)
        self.W_region = (secret_img == self.WHITE)

        self.kernel = self.makeGaussianKernel()
    
    def prepare(self):
        # prepare the part before actually doing thresholding on grayscale images
        self.step0()
        self.step1()
        self.step2()
    
    def execute(self, img1, img2):
        # do thresholding on grayscale images
        # img1, img2 should be grayscale (np.uint8)
        # and should be the same size as secret_img
        if img1.shape != self.secret_img.shape or img2.shape != self.secret_img.shape:
            raise Exception('img1, img2 is not as same shape as secret_img')
        return self.step3(img1, img2)

    # ---

    def makeGaussianKernel(self, kernel_sz=9, sigma=1.5):
        # kernel_sz must be odd
        r = kernel_sz // 2
        cs = np.zeros((kernel_sz, kernel_sz)) + np.arange(kernel_sz) - r
        rs = np.transpose(cs)

        kernel = np.exp(-(cs ** 2 + rs ** 2) / (2 * sigma))
        return kernel / np.sum(kernel)      # not necessary in our use cases though...
        
    def step0(self):
        # generate 2 random pattern: rp1 and rp2
        # s.t. np.bitwise_and(rp1, rp2) can reveal secret
        BLACK, WHITE = self.BLACK, self.WHITE
        self.rp1 = np.zeros(self.secret_img.shape, np.uint8)
        self.rp2 = np.zeros(self.secret_img.shape, np.uint8)
        rand_arr = np.random.rand(*self.secret_img.shape)

        # B -> (B, W) or (W, B)
        # W -> (W, W) or (B, B)
        self.rp1[np.where(np.logical_and(self.B_region, rand_arr < 0.5))] = BLACK
        self.rp2[np.where(np.logical_and(self.B_region, rand_arr < 0.5))] = WHITE

        self.rp1[np.where(np.logical_and(self.B_region, rand_arr > 0.5))] = WHITE
        self.rp2[np.where(np.logical_and(self.B_region, rand_arr > 0.5))] = BLACK

        self.rp1[np.where(np.logical_and(self.W_region, rand_arr < 0.5))] = WHITE
        self.rp2[np.where(np.logical_and(self.W_region, rand_arr < 0.5))] = WHITE

        self.rp1[np.where(np.logical_and(self.W_region, rand_arr > 0.5))] = BLACK
        self.rp2[np.where(np.logical_and(self.W_region, rand_arr > 0.5))] = BLACK
    

    class VACAlgorithm:
        # gives 2 manipulated array `ma1/2`, which will be manipulated by
        # flipPixel and swapPixel
        # Making it a class to avoid directly manipulate main class' variable
        # since it WILL be a mess.
        # ma1 and ma2 should be same shape, of course
        # `img_no` below specifies which `ma` to manipulate
        def __init__(self, ma1, ma2, B_region, W_region, kernel):
            self.ma1 = ma1.copy()
            self.ma2 = ma2.copy()
            self.shape = self.ma1.shape

            self.kernel = kernel.copy()
            self.kernel_size = self.kernel.shape

            self.B_region = B_region.copy()
            self.W_region = W_region.copy()

            self.cluster_score1 = None
            self.cluster_score2 = None
            self.void_score1 = None
            self.void_score2 = None
            self.initializeScore()

        def initializeScore(self):
            # do initialization on score array, in order to do findVAC and flipPixel
            # Must be able to find largest void & cluster, in white region, black region or all.
            
            # maybe do wrapped convolution from OpenCV? Refer to that github for detail...
            
            # Initialization of score array on ma1
            cluster_pattern = self.ma1
            void_pattern = np.logical_not(cluster_pattern).astype(np.int32)

            self.cluster_score1 = convolve2d(cluster_pattern, self.kernel, mode='same', boundary='wrap')
            self.void_score1 = convolve2d(void_pattern, self.kernel, mode='same', boundary='wrap')

            # Initialization of score array on ma2
            cluster_pattern = self.ma2
            void_pattern = np.logical_not(cluster_pattern).astype(np.int32)

            self.cluster_score2 = convolve2d(cluster_pattern, self.kernel, mode='same', boundary='wrap')
            self.void_score2 = convolve2d(void_pattern, self.kernel, mode='same', boundary='wrap')

            return
            
        def findVAC(self, img_no, region, vac):
            # find largest void or cluster, on black or white or all region
            # return image coordinate (written in (row, col))

            # do some np.min or max on score array should do the trick,
            # but take note of region problems...you can only consider points in specified region
            if vac == 'cluster':
                pattern = self.cluster_score1 if img_no == 1 else self.cluster_score2
                if region == 'white':
                    pattern[self.B_region] = 0
                elif region == 'black':
                    pattern[self.W_region] = 0
                largest_idx = np.argmax(pattern)
            else:
                pattern = self.void_score1 if img_no == 1 else self.void_score2
                if region == 'white':
                    pattern[self.B_region] = 0
                elif region == 'black':
                    pattern[self.W_region] = 0
                largest_idx = np.argmax(pattern)

            pos = np.unravel_index(largest_idx, self.shape)
                    
            return pos

        def update_score(self, img_no, pos):
            x,y = pos
            kx, ky = self.kernel_size
            nx, ny = kx // 2, ky //2
            off_x, off_y = (x-nx) % self.shape[0], (y-ny) % self.shape[1]

            img = self.ma1 if img_no == 1 else self.ma2
            img_pad = np.pad(img, (2*nx, 2*ny), mode='wrap')
            patch = img_pad[x:x+4*nx+1, y:y+4*ny+1]
            
            # Update cluster score
            cluster_patch = patch
            cluster_patch_score = convolve2d(cluster_patch, self.kernel, mode='valid')
            cluster_patch_score[cluster_patch[nx:-nx, ny:-ny]==0] = 0
            if img_no == 1:
                score = np.roll(np.roll(self.cluster_score1, -off_x, axis=0), -off_y, axis=1)
                score[:kx, :ky] = cluster_patch_score
                self.cluster_score1 = np.roll(np.roll(score, off_x, axis=0), off_y, axis=1)
            else:
                score = np.roll(np.roll(self.cluster_score2, -off_x, axis=0), -off_y, axis=1)
                score[:kx, :ky] = cluster_patch_score
                self.cluster_score2 = np.roll(np.roll(score, off_x, axis=0), off_y, axis=1)

            # Update void score
            void_patch = np.logical_not(patch).astype(np.int32)
            void_patch_score = convolve2d(void_patch, self.kernel, mode='valid')
            void_patch_score[void_patch[nx:-nx, ny:-ny]==0] = 0
            if img_no == 1:
                score = np.roll(np.roll(self.void_score1, -off_x, axis=0), -off_y, axis=1)
                score[:kx, :ky] = void_patch_score
                self.void_score1 = np.roll(np.roll(score, off_x, axis=0), off_y, axis=1)
            else:
                score = np.roll(np.roll(self.void_score2, -off_x, axis=0), -off_y, axis=1)
                score[:kx, :ky] = void_patch_score
                self.void_score2 = np.roll(np.roll(score, off_x, axis=0), off_y, axis=1)
            return
        
        def flipPixel(self, img_no, pos):
            # flip the pixel on pos, will update both self.ma1/2 and score matrix
            # Should be O(kernel.size) instead of O(ma1.size)

            img = self.ma1 if img_no == 1 else self.ma2
            img[pos] = np.logical_not(img[pos]).astype(np.int32)

            if img_no == 1:
                self.ma1 = img
            else:
                self.ma2 = img

            self.update_score(img_no, pos)
            return 

        def swapPixel(self, img_no, pos1, pos2):
            # swap the pixel on pos1 and pos2, will also update score matrix
            """
            if they are the same color: return
            else: flip them
            """
            img = self.ma1 if img_no == 1 else self.ma2
            img[pos1], img[pos2] = img[pos2], img[pos1]

            if img_no == 1:
                self.ma1 = img
            else:
                self.ma2 = img

            self.update_score(img_no, pos1)
            self.update_score(img_no, pos2)
            return
        
        def getMA(self):
            return self.ma1.copy(), self.ma2.copy()
    
    def findBelongingRegion(self, pos):
        # find which region / color `pos` is on in secret image
        return 'black' if self.B_region[pos] > 0 else 'white'

    def step1(self, max_iter=100000):
        # do VAC-operation 1 to generate threshold matrix
        # psuedocode for algorithm in step1, if I understood correctly
        img_no = random.choice([1, 2])
        vac = self.VACAlgorithm(self.rp1, self.rp2, self.B_region, self.W_region, self.kernel)
        for _ in tqdm(range(max_iter), desc='Step 1'):
            black_pos = vac.findVAC(img_no, 'all', 'cluster')
            vac.flipPixel(img_no, black_pos)
            region = self.findBelongingRegion(black_pos)
            white_pos = vac.findVAC(img_no, region, 'void')
            if white_pos == black_pos:
                vac.flipPixel(img_no, black_pos)
                break
            vac.flipPixel(img_no, white_pos)
            vac.swapPixel(3-img_no, black_pos, white_pos)

        sp1, sp2 = vac.getMA()
        self.sp1 = sp1; self.sp2 = sp2
        return
    
    def genDitherArray(self, sp):
        # use seed pattern `sp` to make dither array
        # which is a very complicated step...but simply following 2nd paper
        # should do the trick
        BLACK, WHITE = self.BLACK, self.WHITE
        num_ones = np.count_nonzero(sp == BLACK)
        da = np.zeros(sp.shape, int)

        # phase 1: enter RANK values between Ones and 0
        vac = self.VACAlgorithm(sp, sp, self.B_region, self.W_region, self.kernel) # for abusing the findVAC() method
        rank = num_ones - 1
        for r in tqdm(range(rank, -1, -1), desc='Step 2.1'):
            cluster_pos = vac.findVAC(1, "all", "cluster")
            vac.flipPixel(1, cluster_pos)
            da[cluster_pos] = r

        # phase 2: enter RANK values between Ones and the half-way point
        vac = self.VACAlgorithm(sp, sp, self.B_region, self.W_region, self.kernel) # for abusing the findVAC() method
        rank = num_ones
        for r in tqdm(range(rank, sp.size), desc='Step 2.2'):
            void_pos = vac.findVAC(1, "all", "void")
            vac.flipPixel(1, void_pos)
            da[void_pos] = r

        return da
    
    def genThresholdArray(self, da):
        # use dither array `da` to make threshold array
        return ((da.astype(float) + 0.5) / da.size * 255).astype(np.uint8)

    def step2(self):
        # do VAC-operation 2 on 2 seed images. 
        # I guess it is exactly the same as second paper w/o modification...?
        self.da1 = self.genDitherArray(self.sp1)
        self.da2 = self.genDitherArray(self.sp2)
        self.ta1 = self.genThresholdArray(self.da1)
        self.ta2 = self.genThresholdArray(self.da2)
    
    def thresholding(self, img, ta):
        # do thresholding using threshold array `ta` on image `img`
        # makes proper image
        img = img.copy()
        n, m = ta.shape
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                img[r, c] = 255 if img[r, c] >= ta[r%n, c%m] else 0
        return img

    def step3(self, img1, img2):
        # do thresholding on both images, maybe returning or store the 2 result image...?
        result1 = self.thresholding(img1, self.ta1)
        result2 = self.thresholding(img2, self.ta2)
        return result1, result2
    
# Test class running tests
# No step 2 because displaying dither / threshold array is not really helpful
class Test:
    @classmethod
    def Test_Step0(cls):
        vh = VacHvcAlgorithm(Helper.makeSecret())
        with Helper.elapsed_timer() as elapsed:
            vh.step0()
        print(f'Step 0 uses {elapsed():.5f} second')
        cv.imshow('secret', Helper.binaryToProperImage(vh.secret_img))
        cv.imshow('rp1', Helper.binaryToProperImage(vh.rp1))
        cv.imshow('rp2', Helper.binaryToProperImage(vh.rp2))
        cv.imshow('reveal secret', Helper.binaryToProperImage(np.bitwise_and(vh.rp1, vh.rp2)))
        cv.waitKey(0)

    @classmethod
    def Test_Step1(cls):
        vh = VacHvcAlgorithm(Helper.makeSecret())
        with Helper.elapsed_timer() as elapsed:
            vh.step0()
            vh.step1()
        print(f'Step 0~1 uses {elapsed():.5f} second')
        cv.imshow('secret', Helper.binaryToProperImage(vh.secret_img))
        cv.imshow('sp1', Helper.binaryToProperImage(vh.sp1))
        cv.imshow('sp2', Helper.binaryToProperImage(vh.sp2))
        cv.imshow('reveal secret', Helper.binaryToProperImage(np.bitwise_and(vh.sp1, vh.sp2)))
        cv.waitKey(0)

    @classmethod
    def Test_All(cls, img1, img2):
        vh = VacHvcAlgorithm(Helper.makeSecret())
        with Helper.elapsed_timer() as elapsed:
            vh.step0()
            vh.step1()
            vh.step2()
            result1, result2 = vh.step3(img1, img2)
        print(f'Step 0~3 uses {elapsed():.5f} second')
        cv.imshow('secret', Helper.binaryToProperImage(vh.secret_img))
        cv.imshow('result1', Helper.binaryToProperImage(result1))
        cv.imshow('result2', Helper.binaryToProperImage(result2))
        cv.imshow('reveal secret', Helper.binaryToProperImage(np.bitwise_and(result1, result2)))
        cv.waitKey(0)
        

def main():
    description = (
        "   VAC-based Halftoned Visual Cryptography (HVC-VAC)\n"
        "   \n"
        "   Given 2 (grayscale) images and 1 secret binary image,\n"
        "   generate 2 binary images s.t. if you bitwise-AND those 2 images,\n"
        "   you can see the secret image.\n"
        "   All 3 input images should be the same size.\n"
        "   \n"
        "   Or if you just want to save the VAC result (i.e. the thresholding arrays and stuff),\n"
        "   you can choose to save the pickled model, so that you can skip the lengthy VAC part and\n"
        "   do thresholding directly on your images later on, by loading back the model.\n"
        "   (The thresholding part is very fast, the time is mostly spent on making the thresholding arrays)\n"
        "   \n"
        "   Refer to examples (in folder `example/`) if you need more examples on how to use this.\n"
    )

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", help="2 grayscale images to be halftoned", nargs=2)
    parser.add_argument("--output", "-o", help="2 binary output image path, should be 2 different path", nargs=2)
    parser.add_argument("--save_model", "-m", help=""
        "Save the pickled model into some destination for further usage.\n"
        "The pickled model is dependent to the secret image, but not input image")
    parser.add_argument("--and_result", "-a", help="Save the bitwise-AND result into the designated path")
    parser.add_argument("--color", "-c", help="Do color version HVC, default grayscale.", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--secret", "-s", help="Binary secret image to be embedded")
    group.add_argument("--load_model", "-l", help="Load the pickled model")
    args = parser.parse_args()

    # check argument
    if bool(args.input) ^ bool(args.output):
        print('Must either specify both input & output, or neither of them.')
        exit(1)
    if bool(args.and_result) and not bool(args.input):
        print('Must input image to do bitwise-AND on result.')
        exit(1)
    
    # load input image & output path
    if args.input is not None:
        input_paths = [Path(p) for p in args.input]
        output_paths = [Path(p) for p in args.output]

        mode = cv.IMREAD_COLOR if args.color else cv.IMREAD_GRAYSCALE
        input_imgs = [cv.imread(str(p), mode) for p in input_paths]
        # check if any of the images are None (i.e. read failed)
        if reduce(lambda x,y: x or y, filter(lambda x: x is None, input_imgs), False):
            print('Failed to read input images!', file=sys.stderr)
            exit(1)

    # load secret image
    if args.secret is not None:
        secret_path = Path(args.secret) if args.secret is not None else None
        secret_img = cv.imread(str(secret_path), cv.IMREAD_GRAYSCALE)
        if secret_img is None:
            print('Failed to read secret images!', file=sys.stderr)
            exit(1)
        secret_img = Helper.properToBinaryImage(secret_img)

    # make / load model
    if args.load_model is not None:
        with open(str(Path(args.load_model)), 'rb') as f:
            vachvc = pickle.load(f)
    else:
        vachvc = VacHvcAlgorithm(secret_img)
        vachvc.prepare()

    # save model
    if args.save_model is not None:
        with open(str(Path(args.save_model)), 'wb') as f:
            pickle.dump(vachvc, f)
    
    # infer images
    if args.input:
        output_imgs = []

        img1, img2 = input_imgs[0], input_imgs[1]
        if not args.color:
            # grayscale
            output_imgs = vachvc.execute(img1, img2)
        else:
            # color
            for chnl in range(3):
                img1[:,:,chnl], img2[:,:,chnl] = vachvc.execute(img1[:,:,chnl], img2[:,:,chnl])
            output_imgs = [img1, img2]
                
        for output_img, output_path in zip(output_imgs, output_paths):
            cv.imwrite(str(output_path), output_img)
        
        if args.and_result:
            cv.imwrite(str(Path(args.and_result)), np.bitwise_and(*output_imgs))
    
    return

if __name__ == '__main__':
    main()