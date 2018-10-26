import os
import cv2
import FocusStack
from multiprocessing import  Process

def stack(folder,image_files):
    focusimages = []
    for img in image_files:
        print ("Reading in file {}".format(img))
        im = cv2.imread("Input/{}/{}".format(folder,img))
        focusimages.append(im)

    merged, depthMap = FocusStack.focus_stack(focusimages)
    cv2.imwrite("{}-allFocus.png".format(folder), merged)
    cv2.imwrite("{}-depth.png".format(folder), depthMap)
    print('allFocus and depth map files are created')

def depthFromDefocus(fold):
    print('Started parallel process on folder-{}'.format(fold))
    image_files = sorted(os.listdir("Input/{}".format(fold)))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)
    stack(fold,image_files)
    return

if __name__ == "__main__":
    imagefolders = sorted(os.listdir("Input"))
    #imagefolders = ['keyboard']
    #p = Pool(min(multiprocessing.cpu_count(),len(imagefolders)))
    #p.map(depthFromDefocus,imagefolders)
    processes = []
    for fold in imagefolders:
        p = Process(target=depthFromDefocus,args=(fold,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
