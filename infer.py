import torch
import numpy as np
import os
import argparse
import socket
import psutil
import cv2
from model.pspnet import PSPNet
value_scale = 255
mean = np.array([0.485, 0.456, 0.406])
mean = mean*value_scale
mean = mean.reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225])
std = std*value_scale
std = std.reshape(3, 1, 1)

def load_model(model_path, layers=101, classes=2, zoom_factor=8, ignore_label=-1):
    model = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, criterion=None)
    model = torch.nn.DataParallel(model.cuda())
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def infer(model, image_path, size=(473, 473)):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1)).astype(np.float32)
    # apply normalization
    image = image - mean
    image = image / std
    image = np.expand_dims(image, axis=0)
    # HWC to CHW
    image = torch.from_numpy(image).cuda()
    if not isinstance(image, torch.FloatTensor):
        image = image.float()
    result = model(image) # 1, 2, W, H
    # argmax
    result = torch.softmax(result, dim=1)
    result = result.squeeze()
    result = result.detach().cpu().numpy()
    
    force_ground_prob = result[1]
    print(force_ground_prob.shape)
    force_ground_prob[ force_ground_prob >= 0.85] = 1
    force_ground_prob[force_ground_prob < 0.85] = 0
    return force_ground_prob

def posprocess(result:torch.Tensor, image, num_class=2):
    # posprocess the result
    result = np.clip(result*255, 0, 255).astype(np.uint8)
    # mask overlay result in image
    overlay = cv2.resize(result, (image.shape[1], image.shape[0]))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result, overlay

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
                        result, overlay = process ()
                        if result is not None:
                            cv2.imwrite(destination_path + r'\result.png',result)
                            cv2.imwrite(destination_path + r'\overlay.png',overlay)

def process ():
    model = load_model(model_path)
    result = infer(model, image_path, (1025, 769))
    image = cv2.imread(image_path)
    result, overlay = posprocess(result, image)
    return result, overlay

def main():
    args = parse_args()
    print(args.pid)
    start_server(args.pid)


if __name__ == "__main__":
    model_path = r'C:\printer_project\image_process\model\train_epoch_100.pth'
    destination_path = os.path.expanduser("~\\Documents") + r'\ImageProcessing'
    image_path = destination_path + r'\image.png'
    main()
    # cv2.imshow("r",overlay)
    # cv2.waitKey(0)



