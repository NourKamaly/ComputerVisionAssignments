import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt


def NonMaximalSuppression(img, radius):
    """
    consider only the max value
    within window of size(radious x radious)
    around each pixel and assume all other value with 0
    """
    suppresedImg = np.zeros(img.shape,dtype=np.uint8)
    height = img.shape[0]
    width = img.shape[1]
    max = -1 
    posX = -1 
    posY = -1
    for row in range (0,height,radius):
        for column in range (0,width, radius):
            filter = img [row:row+radius,column:column+radius]
            for filterRow in range(0,radius):
                for filterColumn in range(0,radius):
                    if filterRow < filter.shape[0] and filterColumn < filter.shape[1]:
                        if filter[filterRow][filterColumn]> max:
                            max = filter[filterRow][filterColumn]
                            posX = filterRow
                            posY = filterColumn
            suppresedImg[row+posX][column+posY] = max
            max=-1
            posX = -1
            posY = -1
    return suppresedImg



"""
1- gradients in both the X and Y directions.
2- smooth the derivative a little using gaussian 
> try on TransA, SimA
> save output as  lab4-1-a-1.png, lab4-1-a-1.png
3- Calculate R:
	3.1 Loop on each pixel:
	3.2 Calculate M for each pixel:
		3.2.1 calculate a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2 
	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
	3.4 Calculate Response at this pixel = det-k*trace^2
	3.5 Display the result, but make sure to re-scale the data in the range 0 to 255 
4- Threshold and Non-Maximal Suppression 

"""
# 1- gradients in both the X and Y directions.
def harris(img, thresh, radius, verbose=True):
    Gx, Gy = utl.get_gradients_xy(img, 5)
    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    # 2- smooth the derivative a little using gaussian
    #Student Code ~ 2 Lines
    Gx = cv2.GaussianBlur(Gx, (5, 5), sigmaX=3,sigmaY=0)
    Gy = cv2.GaussianBlur(Gy, (5, 5), sigmaX=0,sigmaY=3)
    #End Student Code
    
    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04

    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
    # 	3.2 Calculate M for each pixel:
    # 		    M = [[a11, a12],
    #                [a21, a22]]
    #           with a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2
            #Student Code ~ 1 line of code
            M = np.array([[int(Gx[i,j])*int(Gx[i,j]), int(Gx[i,j])*int(Gy[i, j])],
                          [int(Gx[i,j])*int(Gy[i,j]), int(Gy[i,j])*int(Gy[i, j])]])
            #Student Code

    # 	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
            Det_M = np.linalg.det(M)

    # 	3.4 Calculate Response at this pixel = det-k*trace^2
    #   where trace of M is the sum of its diagonals
            #Student Code ~ 1 line of code
            R[i, j] = Det_M - k*(M[0,0]+M[1, 1])**2
            #End Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255

    R = utl.rescale(R, 0, 255)

    plt.imshow(R, cmap="gray")
    plt.show()
    # 5- Threshold and Non-Maximal Suppression
    # Student Code ~ 2 lines of code
    #R[R>thresh] = 255
    #R[R<=thresh] = 0
    #M = 0.01 * R.max()
    #M=-100
    # Threshold for an optimal value, it may vary depending on the image.
    R = NonMaximalSuppression(R, radius)
    R[R > thresh] = 255
    R[R <= thresh] = 0
    # End Student Code
    plt.imshow(R, cmap="gray")
    plt.show()

    return R

img_pairs = [['check.bmp', 'check_rot.bmp']]#,['simA.jpg','simB.jpg'],['transA.jpg','transB.jpg']]
dir = 'input/'
i = 0
radius = [2,3,5,7,9,11,35]
threshold = [190,200,210,220,230,240,250]
for index in range (0,len(radius)):
    for [img1,img2] in img_pairs:
        print ('Image {image1} and image {image2} with radius of {radius} and threshold of {thres}'.format(image1=img1,image2=img2,radius=radius[index],thres=threshold[index]))
        i += 1
        img1 = cv2.imread(dir+img1, 0)
        img2 = cv2.imread(dir+img2, 0)
        r1 = harris(img1,thresh=threshold[index],radius=radius[index])
        r2 = harris(img2,thresh=threshold[index],radius=radius[index]) #Note that threshod may need to be different from picture to another
        plt.figure(i)
        plt.subplot(221), plt.imshow(img1, cmap='gray')
        plt.subplot(222), plt.imshow(img2, cmap='gray')
        plt.subplot(223), plt.imshow(r1, cmap='gray')
        plt.subplot(224), plt.imshow(r2, cmap='gray')
        plt.show()