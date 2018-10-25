import os
import cv2
import FocusStack

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

if __name__ == "__main__":
    imagefolders = sorted(os.listdir("Input"))
    #imagefolders = ['keyboard']
    for fold in imagefolders:
        image_files = sorted(os.listdir("Input/{}".format(fold)))
        for img in image_files:
            if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
                image_files.remove(img)

        stack(fold,image_files)
