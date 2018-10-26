import numpy as np
import cv2
from multiprocessing import Pool, Process, Queue
import multiprocessing
#
#   Align the images so they overlap properly...

def align(imgpair):
    im1, im2 =  imgpair;

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 50;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
    # cv2.imshow("Image 1", im1)
    # cv2.imshow("Image 2", im2)
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)
    #q.put(im2_aligned)
    return im2_aligned



#
#   Compute the gradient map of the image
def LoG(image):
    kernel_size = 5
    blur_size = 5
    blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

#
#   This routine finds the points of best focus in all images and produces a merged result...
#
def focus_stack(unimages):
    print('Aligning focal stack')
    #images = align_images(unimages)
    output_images = []

    #p = Pool()
    images_pair = []
    for img in unimages[1:]:
        images_pair.append((unimages[0],img))

    output_images.append(unimages[0])
    # output_images = output_images + p.map(align,images_pair)

    #processes = []
    #queues = []
    for im in images_pair:
        #q = Queue()
        #p = Process(target=align,args=(im,q,))
        #processes.append(p)
        #queues.append(q)
        #p.start()
        out = align(im)
        #cv2.imshow('image',out)
        #cv2.waitKey(0)
        output_images.append(out)


    #for p in processes:
    #    p.join()
    #for q in queues:
    #    output_images.append(q.get())


    images = np.array(output_images)
    #print(images.shape)
    reference = images[0].copy()

    rows,cols,dims = reference.shape

    print('Computing LoG')
    laps = []
    for i in range(len(images)):
        im = LoG(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))
        laps.append(im)

    laps = np.asarray(laps)

    # Compute the focus mesaure for each lapacian image
    print('Computing sharpness score')
    lapsNP = laps.copy()
    for i in range(len(lapsNP)):
        img = lapsNP[i]
        lapsNP[i] = img*img

    allfocus = np.zeros((rows,cols,3), np.uint8)

    focusMeasure = lapsNP.copy()

    #def measureSharpness(L):
    #    K = 7
    #    kernel = np.ones((K,K),np.uint8)
    #    return cv2.filter2D(L,-1,kernel)
        #for i in range(K,rows-K):
        #    for j in range(K,cols-K):
        #        focusMeasure[k,i,j] = np.sum(L[i-K:i+K+1,j-K:j+K+1])

    K=9
    kernel = np.ones((K,K),np.uint8)
    for k in range(len(lapsNP)):
        focusMeasure[k] = cv2.filter2D(lapsNP[k],-1,kernel)
        #focusMeasure[k] = measureSharpness(lapsNP[k])
        #for i in range(K,rows-K):
        #    for j in range(K,cols-K):
        #        su = 0
        #        focusMeasure[k,i,j] = np.sum(lapsNP[k,i-K:i+K+1,j-K:j+K+1])
        #        for i1 in range(-K,K+1):
        #            for j1 in range(-K,K+1):
        #                su += lapsNP[k,i+i1,j+j1]
        #        focusMeasure[k,i,j] = su

    print('Generating depth map')
    ind = np.argmax(focusMeasure,axis=0)
    #print(ind)
    for i in range(rows):
        for j in range(cols):
            allfocus[i,j] = images[ind[i,j],i,j]

    gaussianStack = laps.copy()
    gaussianStack[0] = cv2.cvtColor(allfocus,cv2.COLOR_BGR2GRAY)
    for i in range(1,len(gaussianStack)):
        gaussianStack[i] = cv2.GaussianBlur(gaussianStack[0],(i*2+1,2*i+1),0)

    im1Gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    imLGray = cv2.cvtColor(images[len(images)-1],cv2.COLOR_BGR2GRAY)

    GAUSSIAN_KERNEL = 11
    differences = gaussianStack.copy()
    for i in range(len(gaussianStack)):
        differences[i] = cv2.GaussianBlur(abs(gaussianStack[i]-im1Gray),(GAUSSIAN_KERNEL,GAUSSIAN_KERNEL),0)

    differences1 = gaussianStack.copy()
    for i in range(len(gaussianStack)):
        differences1[i] = cv2.GaussianBlur(abs(gaussianStack[i]-imLGray),(GAUSSIAN_KERNEL,GAUSSIAN_KERNEL),0)

    depthMap1 = np.argmin(differences,axis=0)
    depthMap2 = np.argmin(differences1,axis=0)


    return allfocus, (depthMap1+np.ones(depthMap2.shape)*np.max(depthMap2) - depthMap2)*255/(2*len(images))
