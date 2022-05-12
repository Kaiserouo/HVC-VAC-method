"""
    Define:
        + binary image: image with value 0s and 1s, since it is needed for convolution step
        + proper image: image with values 0s and 255s
                        You can use `Helper.binaryToProperImage(img)` to 
                        turn binary image into proper image.
        + grayscale image: image with value ranging in [0, 255], with np.uint8
                           Can be directly shown by cv.imshow()
"""

import cv2 as cv
import numpy as np
from typing import *
import random

from contextlib import contextmanager
from timeit import default_timer

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
    

class VacHvcAlgorithm:
    def __init__(self, secret_img, img1, img2):
        # secret_img must be binary
        # img1, img2 should be grayscale (np.uint8)
        # secret_img, img1 and img2 must be same size
        self.secret_img = secret_img
        self.img1 = img1
        self.img2 = img2

        self.BLACK = 0
        self.WHITE = 1

        self.B_region = (secret_img == self.BLACK)
        self.W_region = (secret_img == self.WHITE)
    
    def execute(self):
        # ...or run those yourself.
        self.step0()
        self.step1()
        self.step2()
        self.step3()

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
        def __init__(self, ma1, ma2, kernel):
            self.ma1 = ma1.copy()
            self.ma2 = ma2.copy()
            self.kernel = kernel.copy()
            self.initializeScore()

        def initalizeScore(self):
            # do initialization on score array, in order to do findVAC and flipPixel
            # Must be able to find largest void & cluster, in white region, black region or all.
            
            # maybe do wrapped convolution from OpenCV? Refer to that github for detail...
            pass
            
        def findVAC(self, img_no: Literal[1, 2], region: Literal['white', 'black', 'all'],
                          vac: Literal['void', 'cluster']) -> tuple[int, int]:
            # find largest void or cluster, on black or white or all region
            # return image coordinate (written in (row, col))

            # do some np.min or max on score array should do the trick,
            # but take note of region problems...you can only consider points in specified region
            pass
        
        def flipPixel(self, img_no, pos):
            # flip the pixel on pos, will update both self.ma1/2 and score matrix
            # Should be O(kernel.size) instead of O(ma1.size)

            img = self.ma1 if img_no == 1 else self.ma2
            if img[pos] == 1:
                img[pos] = 0
            else:
                img[pos] = 1
            
            return 

        def swapPixel(self, img_no, pos1, pos2):
            # swap the pixel on pos1 and pos2, will also update score matrix
            """
            if they are the same color: return
            else: flip them
            """
            img = self.ma1 if img_no == 1 else self.ma2
            img[pos1], img[pos2] = img[pos2], img[pos1]

            return
        
        def getMA(self):
            return self.ma1.copy(), self.ma2.copy()
    
    def findBelongingRegion(self, pos):
        # find which region / color `pos` is on in secret image
        return 'black' if self.B_region[pos] > 0 else 'white'

    def step1(self, max_iter=100):
        # do VAC-operation 1 to generate threshold matrix
        # psuedocode for algorithm in step1, if I understood correctly
        img_no = random.choice([1, 2])
        vac = self.VACALgorithm(self.rp1, self.rp2)
        for _ in range(max_iter):
            black_pos = vac.findVAC(img_no, 'all', 'cluster')
            vac.flipPixel(img_no, black_pos)
            region = self.findBelongingRegion(black_pos)
            white_pos = self.findVAC(img_no, region, 'void')
            if white_pos == black_pos:
                vac.flipPixel(img_no, black_pos)
                break
            vac.flipPixel(img_no, white_pos)
            vac.swapPixel(2-img_no, black_pos, white_pos)

        sp1, sp2 = vac.getMA()
        self.sp1 = sp1; self.sp2 = sp2
        return
    
    def genDitherArray(self, sp):
        # use seed pattern `sp` to make dither array
        # which is a very complicated step...but simply following 2nd paper
        # should do the trick
        pass
    
    def genThresholdArray(self, da):
        # use dither array `da` to make threshold array
        return ((da.astype(float) + 0.5) / da.size * 255).astype(np.uint8)

    def step2(self):
        # do VAC-operation 2 on 2 seed images. 
        # I guess it is exactly the same as second paper w/o modification...?
        self.da1 = self.genDitherMatrix(self, self.sp1)
        self.da2 = self.genDitherMatrix(self, self.sp2)
        self.ta1 = self.genThresholdArray(self, self.da1)
        self.ta2 = self.genThresholdArray(self, self.da2)
    
    def thresholding(self, img, ta):
        # do thresholding using threshold array `ta` on image `img`
        # makes proper image
        img = img.copy()
        n, m = ta.shape
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                img[r, c] = 255 if img[r, c] >= ta[r%n, c%m] else 0
        return img

    def step3(self):
        # do thresholding on both images, maybe returning or store the 2 result image...?
        self.result1 = self.thresholding(self.img1, self.ta1)
        self.result2 = self.thresholding(self.img2, self.ta2)
    
# Test class running tests
# No step 2 because displaying dither / threshold array is not really helpful
class Test:
    @classmethod
    def Test_Step0(cls):
        vh = VacHvcAlgorithm(Helper.makeSecret(), None, None)
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
        vh = VacHvcAlgorithm(Helper.makeSecret(), None, None)
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
        vh = VacHvcAlgorithm(Helper.makeSecret(), img1, img2)
        with Helper.elapsed_timer() as elapsed:
            vh.step0()
            vh.step1()
            vh.step2()
            vh.step3()
        print(f'Step 0~3 uses {elapsed():.5f} second')
        cv.imshow('secret', Helper.binaryToProperImage(vh.secret_img))
        cv.imshow('result1', Helper.binaryToProperImage(vh.result1))
        cv.imshow('result2', Helper.binaryToProperImage(vh.result2))
        cv.imshow('reveal secret', Helper.binaryToProperImage(np.bitwise_and(vh.result1, vh.result2)))
        cv.waitKey(0)
        
if __name__ == '__main__':
    Test.Test_Step0()