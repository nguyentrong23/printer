import cv2
import numpy as np

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image  = cv2.GaussianBlur(image , (5, 5), 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    thershold = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thershold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 1000]
    x_ale, y_ale, w_ale, h_ale = cv2.boundingRect(np.vstack(contours).reshape(-1, 1, 2))
    rect_ale = (x_ale, y_ale, w_ale, h_ale)
    mask_con = np.zeros(image.shape[:2], np.uint8)
    # target
    for id, con in enumerate(contours):
        if hierarchy[0, id, 3] != -1:
            continue
        cv2.drawContours(mask_con, [con], -1, 3, thickness=cv2.FILLED)
        # cv2.imshow('thershold', mask_con * 80)
        # cv2.waitKey(0)
    mask_able = cv2.bitwise_and(thershold, mask_con)
    kernel = np.ones((5,5), np.uint8)
    mask_target = cv2.erode(mask_con, kernel, iterations=1)
    mask = np.where(mask_target != 0,1, mask_able)
    # cv2.imshow('imge',  mask*80)
    # cv2.imshow('thershold', mask_con*80)
    # cv2.imshow('mask_able', mask_able * 80)
    return  thershold, rect_ale,mask


# Đọc ảnh
image = cv2.imread('C:\printer_project\image_process\img\Image__2024-07-24__16-48-07.bmp')
image = cv2.pyrDown(image)
image = cv2.pyrDown(image)
e,rect_ale,mask_target = preprocess(image)
# mask1
mask1 = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(image, mask1,rect_ale, bgdModel, fgdModel,2, cv2.GC_INIT_WITH_RECT)
mask1r = np.where((mask1 ==2) |(mask1 == 0), 0, 1).astype('uint8')
result = image * mask1r[:, :, np.newaxis]

# mask2
mask2 = np.where((mask1 ==2) |(mask1 == 3), 2, 0).astype('uint8')
mask = np.where(mask_target != 0, mask_target, mask2)
cv2.grabCut(image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
mask2r = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result2 = image * mask2r[:, :, np.newaxis]
#
cv2.imshow('i',result2)
cv2.imshow('imge',mask*80)
cv2.waitKey(0)
cv2.destroyAllWindows()