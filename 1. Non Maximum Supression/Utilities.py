import cv2
import numpy as np

def get_gradients_xy(img, ksize):
    ## Student Code ~ 2 lines of codes
    #print(img.dtype)
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1,0,ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0,1,ksize=ksize)
    #
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)

    sobelx = np.uint8(sobelx)
    sobely = np.uint8(sobely)

    return sobelx, sobely
    ## Student Code Ends


def rescale(img, min,max):
    ## Student Code
    img = (img-img.min())/float(img.max()-img.min())
    img = min + img * (max-min)
    ## End Student Code
    return img