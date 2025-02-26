import cv2
import numpy as np
import os
import time
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image  = cv2.GaussianBlur(image , (7, 7), 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))


    thershold = cv2.threshold(sobel_combined, 70, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thershold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 1000]
    mask_con = np.zeros(image.shape[:2], np.uint8)

    # target
    for id, con in enumerate(contours):
        if hierarchy[0, id, 3] != -1:
            continue
        cv2.drawContours(mask_con, [con], -1, 3, thickness=cv2.FILLED)

    mask_able = cv2.bitwise_and(thershold, mask_con)
    cv2.imshow('mask_con', mask_con * 80)
    cv2.imshow('thershold', thershold)
    # cv2.imshow('mask_able',mask_able * 80)
    kernel = np.ones((5,5), np.uint8)
    mask_target = cv2.erode(mask_con, kernel, iterations=1)
    mask = np.where(mask_target != 0,1, mask_able)
    # show =  cv2.pyrDown(mask)
    # show1 =  cv2.pyrDown(mask_able)
    # show2 =  cv2.pyrDown(mask_target)
    # cv2.imshow('imge',  show*80)
    # cv2.imshow('imge1',  show1*80)
    # cv2.imshow('img2e',  show2*80)
    return  thershold,mask


# Đọc ảnh
image = cv2.imread(r'img/Image__2024-07-24__16-48-07.bmp')
image = cv2.pyrDown(image)
e,mask_target = preprocess(image)
# # # mask2
# mask = np.where(mask_target != 0, mask_target, 0).astype('uint8')
# cv2.grabCut(image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
# mask2r = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# result2 = image * mask2r[:, :, np.newaxis]
# cv2.imwrite(destination_path + '\ImageProcessing\mask.png', mask2r)
# cv2.imwrite(destination_path + '\ImageProcessing\image2see.png', result2)
# cv2.imshow('i',mask2r*80)
# cv2.imshow('result2',result)
cv2.waitKey(0)
cv2.destroyAllWindows()