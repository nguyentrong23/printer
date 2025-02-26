import torch
import torch.nn as nn
import torch.nn.functional as F
from util import transform
import numpy as np
import os
import cv2
from model.pspnet import PSPNet

value_scale = 255
mean = np.array([0.485, 0.456, 0.406])
mean = mean * value_scale
mean = mean.reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225])
std = std * value_scale
std = std.reshape(3, 1, 1)


def load_model(model_path, layers=101, classes=2, zoom_factor=8, ignore_label=-1):
    model = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, criterion=None)
    model = torch.nn.DataParallel(model.cuda())
    checkpoint = torch.load(model_path)
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
    result = model(image)  # 1, 2, W, H
    # argmax
    result = torch.softmax(result, dim=1)
    result = result.squeeze()
    result = result.detach().cpu().numpy()

    force_ground_prob = result[1]
    print(force_ground_prob.shape)
    force_ground_prob[force_ground_prob >= 0.9] = 1
    force_ground_prob[force_ground_prob < 0.9] = 0
    return force_ground_prob


def posprocess(result: torch.Tensor, image, num_class=2):
    # posprocess the result
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    # mask overlay result in image
    overlay = cv2.resize(result, (image.shape[1], image.shape[0]))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result, overlay


if __name__ == "__main__":
    model_path =  r'C:\printer_project\image_process\model\train_epoch_100.pth'
    image_path = r'C:\printer_project\PSPNet-main\test_image\test.bmp'
    model = load_model(model_path)
    result = infer(model, image_path, (1025, 769))
    image = cv2.imread(image_path)
    result, overlay = posprocess(result, image)
    cv2.imwrite('result.jpg', overlay)
    cv2.imshow("r", result)
    cv2.waitKey(0)

