"""
    Define:
        + binary image: image with value 0s and 1s, since it is needed for convolution step
        + proper image: image with values 0s and 255s
                        You can use Helper.binaryToProperImage(img) to 
                        turn binary image into proper image.
"""

import cv2 as cv
import numpy as np

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
    

class VacHvc:
    def __init__(self, secret_img):
        # secret_img must be binary
        self.secret_img = secret_img
        self.BLACK = 0
        self.WHITE = 1

    def makeGaussianKernel(self, kernel_sz=9, sigma=1.5):
        # kernel_sz must be odd
        r = kernel_sz // 2
        cs = np.zeros((kernel_sz, kernel_sz)) + np.arange(kernel_sz) - r
        rs = np.transpose(cs)

        kernel = np.exp(-(cs ** 2 + rs ** 2) / (2 * sigma))
        return kernel / np.sum(kernel)      # not necessary in our use cases though...
        
    def step1(self):
        # generate 2 random pattern: rp1 and rp2
        # s.t. np.bitwise_and(rp1, rp2) can reveal secret
        BLACK, WHITE = self.BLACK, self.WHITE
        self.rp1 = np.zeros(self.secret_img.shape, np.uint8)
        self.rp2 = np.zeros(self.secret_img.shape, np.uint8)
        rand_arr = np.random.rand(*self.secret_img.shape)

        # B -> (B, W) or (W, B)
        # W -> (W, W) or (B, B)
        self.rp1[np.where(np.logical_and(self.secret_img == BLACK, rand_arr < 0.5))] = BLACK
        self.rp2[np.where(np.logical_and(self.secret_img == BLACK, rand_arr < 0.5))] = WHITE

        self.rp1[np.where(np.logical_and(self.secret_img == BLACK, rand_arr > 0.5))] = WHITE
        self.rp2[np.where(np.logical_and(self.secret_img == BLACK, rand_arr > 0.5))] = BLACK

        self.rp1[np.where(np.logical_and(self.secret_img == WHITE, rand_arr < 0.5))] = WHITE
        self.rp2[np.where(np.logical_and(self.secret_img == WHITE, rand_arr < 0.5))] = WHITE

        self.rp1[np.where(np.logical_and(self.secret_img == WHITE, rand_arr > 0.5))] = BLACK
        self.rp2[np.where(np.logical_and(self.secret_img == WHITE, rand_arr > 0.5))] = BLACK

class Test:
    @classmethod
    def Test_Step1(cls):
        vh = VacHvc(Helper.makeSecret())
        with Helper.elapsed_timer() as elapsed:
            vh.step1()
        print(f'Step 1 uses {elapsed():.5f} second')
        cv.imshow('secret', Helper.binaryToProperImage(vh.secret_img))
        cv.imshow('rp1', Helper.binaryToProperImage(vh.rp1))
        cv.imshow('rp2', Helper.binaryToProperImage(vh.rp2))
        cv.imshow('reveal secret', Helper.binaryToProperImage(np.bitwise_and(vh.rp1, vh.rp2)))
        cv.waitKey(0)
        
if __name__ == '__main__':
    Test.Test_Step1()