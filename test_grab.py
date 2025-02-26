import cv2
import numpy as np
<<<<<<< HEAD
import os
import argparse
import socket
import psutil
=======

>>>>>>> 34fd6655559e773ee15f8236d7b3c619f13c3d39
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
<<<<<<< HEAD
=======
        # cv2.imshow('thershold', mask_con * 80)
        # cv2.waitKey(0)
>>>>>>> 34fd6655559e773ee15f8236d7b3c619f13c3d39
    mask_able = cv2.bitwise_and(thershold, mask_con)
    kernel = np.ones((5,5), np.uint8)
    mask_target = cv2.erode(mask_con, kernel, iterations=1)
    mask = np.where(mask_target != 0,1, mask_able)
    # cv2.imshow('imge',  mask*80)
<<<<<<< HEAD
    return  thershold, rect_ale,mask
def check_image_exists(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            if cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 0:
                return 1
            else:
                return 4
        else:
            return 3
    else:
        return 0


def grabb():
    check = check_image_exists(image_path)
    if check == 1:
        pass
    else:
        file_path = os.path.join(destination_path, 'test.txt')
        with open(file_path, 'a') as f:
            if check == 0:
                f.write('\n path error')
            elif check == 3:
                f.write('\n image error')
            elif check == 4:
                f.write('\n Bad image')
        return  0
    image = cv2.imread(image_path)
    image = cv2.pyrDown(image)
    e, rect_ale, mask_target = preprocess(image)
    mask1 = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask1, rect_ale, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask1 == 2) | (mask1 == 3), 2, 0).astype('uint8')
    mask = np.where(mask_target != 0, mask_target, mask2)
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask2r = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    return mask2r

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int)
    return parser.parse_args()

def is_pid_running(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except psutil.NoSuchProcess:
        return False

def main():
    args = parse_args()
    print(args.pid)
    # listen event from C# process via socket
    start_server(args.pid)

def start_server(pid,host='localhost', port=65525):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server started on {host}:{port}")
        while True:
            conn, addr = server_socket.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        if is_pid_running(pid):
                            pass
                        else:
                            with open(destination_path + '/test.txt', 'a') as f:
                                f.write(f'\n Killed PID: {pid}')
                            conn.close()
                            server_socket.close()
                            return 0
                    message = data.decode('utf-8')
                    if message == 'trigger':
                        garble = grabb()
                        if garble is not None:
                            cv2.imwrite(destination_path + r'\mask.png',garble)
if __name__ == '__main__':
    destination_path = os.path.expanduser("~\\Documents") + r'\ImageProcessing'
    image_path = destination_path + r'\image.png'
    main()


=======
    # cv2.imshow('thershold', mask_con*80)
    # cv2.imshow('mask_able', mask_able * 80)
    return  thershold, rect_ale,mask


# Đọc ảnh
image = cv2.imread('img/font/art_font2.bmp')
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
>>>>>>> 34fd6655559e773ee15f8236d7b3c619f13c3d39
