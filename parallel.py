from functools import partial
import itertools
import main as Main
from pathlib import Path

import cv2 as cv
import numpy as np
import threading

# try to dissect the original image into size 
# (img.shape[0] // DISSECT_CNT[0], img.shape[1] // DISSECT_CNT[1])
DISSECT_CNT = (5, 5)

INPUT_PATH_1 = Path('./example/test_color/img_a.png')
INPUT_PATH_2 = Path('./example/test_color/img_b.png')
OUTPUT_PATH_1 = Path('./img_res_a.png')
OUTPUT_PATH_2 = Path('./img_res_b.png')
SECRET_PATH = Path('./example/test_color/img_sec.png')

DO_COLOR = True

NUM_THREAD = 4

class ThreadResourse:
    def __init__(self, img1_arr, img2_arr, sec_arr):
        # let threads take the task
        # 'next_task' will specify what the next task is,
        # define (-1, -1) to represent all task done and 
        # threads can quit and be joined.

        self.img1_arr = img1_arr
        self.img2_arr = img2_arr
        self.sec_arr = sec_arr

        self.lock = threading.Lock()
        self.next_task = (0, 0)

    def toNextTask(self):
        # modily next_task to the next task...
        num = self.next_task[0] * DISSECT_CNT[1] + self.next_task[1]
        num += 1
        if num == DISSECT_CNT[0] * DISSECT_CNT[1]:
            self.next_task = (-1, -1)
        else:
            self.next_task = (num // DISSECT_CNT[1], num % DISSECT_CNT[1])
        

def threadMain(resourse):
    # take resourse and do VAC-HVC, will rewrite 
    task = (-1, -1)
    while True:
        # get task
        resourse.lock.acquire()
        if resourse.next_task == (-1, -1):
            resourse.lock.release()
            print(f"[{threading.current_thread().name}] Exiting...")
            return
        else:
            task = resourse.next_task
            resourse.toNextTask()
            print(f"[{threading.current_thread().name}] Got task {task}")
        
        # get img
        img1 = resourse.img1_arr[task[0]][task[1]]
        img2 = resourse.img2_arr[task[0]][task[1]]
        sec = resourse.sec_arr[task[0]][task[1]]
        resourse.lock.release()
        
        # VAC-HVC
        vachvc = Main.VacHvcAlgorithm(sec)
        vachvc.prepare()

        if not DO_COLOR:
            # grayscale
            output_imgs = vachvc.execute(img1, img2)
        else:
            # color
            for chnl in range(3):
                img1[:,:,chnl], img2[:,:,chnl] = vachvc.execute(img1[:,:,chnl], img2[:,:,chnl])
            output_imgs = [img1, img2]

        # put back
        resourse.lock.acquire()
        resourse.img1_arr[task[0]][task[1]] = output_imgs[0]
        resourse.img2_arr[task[0]][task[1]] = output_imgs[1]
        resourse.lock.release()

def makeRsCs(img):
    # rs, cs: the height / width of subimages
    shape = img.shape
    rs = [
        shape[0] // DISSECT_CNT[0] + int(i < shape[0] % DISSECT_CNT[0])
        for i in range(DISSECT_CNT[0])
    ]
    cs = [
        shape[1] // DISSECT_CNT[1] + int(i < shape[1] % DISSECT_CNT[1])
        for i in range(DISSECT_CNT[1])
    ]
    return rs, cs
    
def makeImageArray(img, rs, cs):
    # make s.t. img_arr[a][b] = subimage
    # where (a, b) in range(DISSECT_CNT[0]) x range(DISSECT_CNT[0])
    c_rs = [0] + list(itertools.accumulate(rs))
    c_cs = [0] + list(itertools.accumulate(cs))
    ls = [
        [
            img[c_rs[a]:c_rs[a+1], c_cs[b]:c_cs[b+1]]
            for b in range(DISSECT_CNT[1])
        ]
        for a in range(DISSECT_CNT[0])
    ]
    return ls

def mergeImg(img_ls):
    r_ls = [
        cv.hconcat(c_ls)
        for c_ls in img_ls
    ]
    return cv.vconcat(r_ls)

def main():
    mode = cv.IMREAD_COLOR if DO_COLOR else cv.IMREAD_GRAYSCALE
    img1 = cv.imread(str(INPUT_PATH_1), mode)
    img2 = cv.imread(str(INPUT_PATH_2), mode)
    sec = cv.imread(str(SECRET_PATH), cv.IMREAD_GRAYSCALE)
    sec = Main.Helper.properToBinaryImage(sec)
        
    # dissect images into list
    shape = img1.shape
    rs = [
        shape[0] // DISSECT_CNT[0] + int(i < shape[0] % DISSECT_CNT[0])
        for i in range(DISSECT_CNT[0])
    ]
    cs = [
        shape[1] // DISSECT_CNT[1] + int(i < shape[1] % DISSECT_CNT[1])
        for i in range(DISSECT_CNT[1])
    ]

    img1_arr = makeImageArray(img1, rs, cs)
    img2_arr = makeImageArray(img2, rs, cs)
    sec_arr = makeImageArray(sec, rs, cs)

    # do threading
    thread_resource = ThreadResourse(img1_arr, img2_arr, sec_arr)
    thread_ls = []
    with Main.Helper.elapsed_timer() as timer:
        for i in range(NUM_THREAD):
            thread_ls.append(threading.Thread(target=threadMain, args=(thread_resource,)))
        for i in range(NUM_THREAD):
            thread_ls[i].start()
        for i in range(NUM_THREAD):
            thread_ls[i].join()
    print(f'Uses {timer():.6f} seconds')
    
    out1 = mergeImg(img1_arr)
    out2 = mergeImg(img2_arr)
    cv.imwrite(str(OUTPUT_PATH_1), out1)
    cv.imwrite(str(OUTPUT_PATH_1), out2)


def test():
    img = cv.imread('./example/test_color/img_a.png')
    shape = img.shape
    rs = [
        shape[0] // DISSECT_CNT[0] + int(i < shape[0] % DISSECT_CNT[0])
        for i in range(DISSECT_CNT[0])
    ]
    cs = [
        shape[1] // DISSECT_CNT[1] + int(i < shape[1] % DISSECT_CNT[1])
        for i in range(DISSECT_CNT[1])
    ]

    img_arr = makeImageArray(img, rs, cs)
    for v in img_arr:
        for i in v:
            cv.imshow('a', i)
            cv.waitKey(100)
    cv.imshow('a', mergeImg(img_arr))
    cv.waitKey(0)

if __name__ == '__main__':
    main()
