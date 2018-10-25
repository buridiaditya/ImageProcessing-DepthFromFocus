import numpy as np
import cv2

#
#   Align the images so they overlap properly...

 
def align_images(images):
    output_images = []
    output_images.append(images[0])
    im1 = images[0]
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

    for im2 in images[1:]:
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = images[0].shape
        
        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE
        
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Specify the number of iterations.
        number_of_iterations = 50
        
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
        
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # Show final results
        # cv2.imshow("Image 1", im1)
        # cv2.imshow("Image 2", im2)
        # cv2.imshow("Aligned Image 2", im2_aligned)
        # cv2.waitKey(0)
        output_images.append(im2_aligned)

    return np.array(output_images)

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
    images = align_images(unimages)

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

    imagesNP = np.array(images).copy()
    allfocus = np.zeros((rows,cols,3), np.uint8)    

    focusMeasure = lapsNP.copy()
    K = 5
    for k in range(len(lapsNP)):
        for i in range(2,rows-2):
            for j in range(2,cols-2):
                su = 0
                for i1 in range(-2,3):
                    for j1 in range(-2,3):
                        su += lapsNP[k,i+i1,j+j1]
                focusMeasure[k,i,j] = su

    print('Generating depth map')
    ind = np.argmax(focusMeasure,axis=0)

    for i in range(rows):
        for j in range(cols):
            allfocus[i,j] = imagesNP[ind[i,j],i,j]


    return allfocus, ind*20