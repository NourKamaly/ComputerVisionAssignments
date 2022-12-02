#!/usr/bin/env python
# coding: utf-8

# In[83]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics


# In[84]:


directory = "Data"
pairs = [['image1a.jpeg','image1b.jpeg'],['image2a.jpeg','image2b.jpeg'],['image3a.jpeg','image3b.jpeg'],['image4a.jpeg','image4b.jpeg','image4c.png'],
         ['image5a.jpeg','image5b.jpeg'], ['image6a.jpeg','image6b.jpeg'],['image7a.jpeg','image7b.jpeg']]


# In[85]:


def getSIFTKpsAndDecriptors(image):
    SIFTMatcher = cv2.SIFT_create()
    (keypoints, descriptors) = SIFTMatcher.detectAndCompute(image,None)
    #keypoints = np.float32([keypoints.pt for keypoint in keypoints])
    return keypoints, descriptors


# In[86]:


def getMatches(descriptors1, descriptors2):
    bruteForceMatcher = cv2.BFMatcher()
    matches = bruteForceMatcher.knnMatch(descriptors1,descriptors2,2)
    return matches


# In[87]:


def getDavidLoweRatioMatches(matches,davidLoweRatio):
    DLmatches =[]
    for match in matches:
        if(match[0].distance/match[1].distance < davidLoweRatio):
                DLmatches.append(match[0])
    return DLmatches


# In[88]:


def getDistancesArray(matches):
    distanceArray = []
    for match in matches:
        distanceArray.append(match.distance)
    return np.array(distanceArray)


# In[89]:


def getInterQuartileRange(distances):
    firstQuartile = np.percentile(distances,25)
    thirdQuartile = np.percentile(distances,75)
    interQuartileRange = thirdQuartile - firstQuartile
    return interQuartileRange


# In[90]:


def filterMatchesUsingIQR(interQuartileRange, DLmatches):
    IQRMatches = []
    for match in DLmatches:
        if (match.distance < interQuartileRange):
            IQRMatches.append(match)
    return IQRMatches


# In[91]:


def filterMatchesUsingMedian(DLmatches):
    medianMatches = []
    distances = getDistancesArray(DLmatches)
    median = statistics.median(distances)
    for match in DLmatches:
         if (match.distance < median):
                medianMatches.append(match)
    return medianMatches


# In[92]:


def calculateSimilarity(matches, keyPointsLeft,keyPointsRight):
    similarity = len(matches)/min(len(keyPointsLeft),len(keyPointsRight))
    return similarity


# In[93]:


def run(leftPairDirectory,rightPairDirectory, useMedian = False,useIQR = False):
    leftPair = cv2.imread(os.path.join(directory,leftPairDirectory),cv2.IMREAD_GRAYSCALE)
    rightPair = cv2.imread(os.path.join(directory,rightPairDirectory),cv2.IMREAD_GRAYSCALE)
    leftPair = cv2.resize(leftPair,(500,500))
    rightPair = cv2.resize(rightPair,(500,500))
    keyPointsLeft, descriptorsLeft = getSIFTKpsAndDecriptors(leftPair)
    keyPointsRight, descriptorsRight = getSIFTKpsAndDecriptors(rightPair)
    matches = getMatches(descriptorsLeft, descriptorsRight)
    DLmatches = getDavidLoweRatioMatches(matches,0.8)
    if useMedian == True:
        drawnMatches = filterMatchesUsingMedian(DLmatches)
    else: 
        distances = getDistancesArray(DLmatches)
        IQR = getInterQuartileRange(distances)
        drawnMatches = filterMatchesUsingIQR(IQR, DLmatches)
    vis = cv2.drawMatches(leftPair,keyPointsLeft,rightPair,keyPointsRight,drawnMatches,outImg = None,matchesThickness=2)
    cv2.imshow("img",vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    similarity = calculateSimilarity(drawnMatches, keyPointsLeft,keyPointsRight)
    print("The similarity is equal {0}".format(similarity))
    if (similarity > 0.01):
        print("They are similar")
    else :
        print("they are not similar")


# In[94]:


for pair in pairs:
    run(pair[0],pair[1],useMedian= True)


# In[95]:


run(pairs[3][1],pairs[3][2],useMedian= True)


# In[96]:


run(pairs[3][0],pairs[3][2],useMedian = True)

