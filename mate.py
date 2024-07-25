import cv2
import numpy as np


def getcontour(path, thr_area):
    valid_list = []
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area <= thr_area:
            continue
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            mu11 = M['mu11']
            mu20 = M['mu20']
            mu02 = M['mu02']
            angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            valid_list.append({"center": (cx, cy), "con": contour, "area": area, "rotate": -angle})
    return valid_list


def rotate_contour(contour, angle_deg,scale_factor, center, newcenter, show = False):
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    contour = np.array(contour, dtype=np.float32).reshape(-1, 2)
    rotated_contour = cv2.transform(np.array([contour]), M)[0]
    rotated_contour = np.array(rotated_contour, dtype=np.int32).reshape(-1, 1, 2)
    S = np.array([[scale_factor, 0], [0, scale_factor]])
    contour = np.array(rotated_contour, dtype=np.float32).reshape(-1, 2)
    scaled_contour = np.dot(contour - center, S) + center
    scaled_contour = np.array(scaled_contour, dtype=np.int32).reshape(-1, 1, 2)
    if center==newcenter:
        return scaled_contour
    translation_vector = np.array(newcenter, dtype=np.int32) - np.array(center, dtype=np.int32)
    translated_contour = scaled_contour + translation_vector
    return translated_contour


def bitwise_contours(contour1, contour2, mask_shape):
    if contour1 is None or len(contour1) == 0 or contour2 is None or len(contour2) == 0:
        return 0
    mask1 = np.zeros(mask_shape,dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    mask2 = np.zeros(mask_shape,dtype=np.uint8)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)
    result_mask = cv2.bitwise_and(mask1, mask2)
    intersection_points = cv2.countNonZero(result_mask)
    target_points = cv2.countNonZero(mask2)
    percen1 = 1-(intersection_points/(target_points+1))
    #..........................
    real_point = cv2.countNonZero(mask1)
    percen2 = 1-(target_points/(real_point+1))
    loss_percen = abs(percen1) + abs(percen2)
    return intersection_points, loss_percen

def loss_function(params,last_cen, template_contour, target_contour, image_shape):
    angle, scale, tx, ty = params
    center = (tx, ty)
    # print("last_cen, center",last_cen, center)
    transformed_contour = rotate_contour(template_contour, angle, scale, last_cen, center, show = True)
    _,percent = bitwise_contours(transformed_contour, target_contour, image_shape)
    return -percent

def gradient_descent_individual(template_contour, target_contour, image_shape, initial_params, learning_rate,
                                num_iterations, tolerance):
    params = np.array(initial_params, dtype=np.float32)
    velocity = np.zeros_like(params)
    best_params = np.copy(params)
    best_loss = float('inf')
    stagnation_counter = 0
    beta = 0.9
    current_cen = (params[2], params[3])
    for i in range(num_iterations):
        current_loss = loss_function(params,current_cen , template_contour, target_contour, image_shape)
        d_angle = (loss_function([params[0] + 0.1, params[1], params[2], params[3]],current_cen, template_contour, target_contour,
                                 image_shape) - current_loss) / 0.1
        d_scale = (loss_function([params[0], params[1] + 0.001, params[2], params[3]],current_cen, template_contour, target_contour,
                                 image_shape) - current_loss) / 0.001
        d_tx = (loss_function([params[0], params[1], params[2] + 1, params[3]],current_cen, template_contour, target_contour,
                              image_shape) - current_loss) / 1
        d_ty = (loss_function([params[0], params[1], params[2], params[3] + 1],current_cen, template_contour, target_contour,
                              image_shape) - current_loss) / 1

        velocity[0] = beta * velocity[0] + (1 - beta) * d_angle
        velocity[1] = beta * velocity[1] + (1 - beta) * d_scale
        velocity[2] = beta * velocity[2] + (1 - beta) * d_tx
        velocity[3] = beta * velocity[3] + (1 - beta) * d_ty

        current_cen = (params[2], params[3])
        params[0] -= learning_rate * velocity[0]
        params[1] -= learning_rate * velocity[1]
        params[2] -= learning_rate * velocity[2]
        params[3] -= learning_rate * velocity[3]
        print(f"Iteration {i}: current_loss = {current_loss}, params = {params}")
        # Cập nhật giá trị best_loss và best_params nếu current_loss thấp hơn
        if current_loss > best_loss:
            best_loss = current_loss
            best_params = np.copy(params)
            stagnation_counter = 0  # Reset stagnation counter
        else:
            stagnation_counter += 1
        if stagnation_counter > 50:
            print(f"Stopping early at iteration {i} with best loss {best_loss}")
            break
        if abs(current_loss) < 0.003:
            print("Converged", abs(current_loss))
            break
    return best_params

path_temp = "test_location/temp - Copy.png"
path_samp = "test_location/4test.png"
temp_list = getcontour(path_temp, 1000)
wish_list = getcontour(path_samp, 1000)
image = cv2.imread(path_samp)
temp = cv2.imread(path_temp)
result =[]


for id, con in enumerate(wish_list):
    dict ={}
    scale_rate = np.sqrt(con["area"]/(temp_list[0]["area"]+1))
    angle_deg = np.degrees(con["rotate"]) - np.degrees(temp_list[0]["rotate"])
    angle_deg_inv = angle_deg +180
    angle_deg_Square = angle_deg+90
    angle_deg_Square_inv = angle_deg+270

    con1 = rotate_contour(temp_list[0]["con"], angle_deg,scale_rate, temp_list[0]["center"],con["center"] )
    con2 = rotate_contour(temp_list[0]["con"], angle_deg_inv,scale_rate, temp_list[0]["center"],con["center"] )
    con3 = rotate_contour(temp_list[0]["con"], angle_deg_Square,scale_rate, temp_list[0]["center"],con["center"] )
    con4 = rotate_contour(temp_list[0]["con"], angle_deg_Square_inv,scale_rate, temp_list[0]["center"],con["center"] )

    count1,_=bitwise_contours(con1, con["con"], (image.shape[0], image.shape[1]))
    count2,_=bitwise_contours(con2, con["con"], (image.shape[0], image.shape[1]))
    count3,_=bitwise_contours(con3, con["con"], (image.shape[0], image.shape[1]))
    count4,_=bitwise_contours(con4, con["con"], (image.shape[0], image.shape[1]))

    if max(count1,count2,count3,count4) == count1:
        dict["angel"] = con["rotate"]
        dict["con"] = con1
    elif max(count1,count2,count3,count4) == count2:
        dict["angel"] = con["rotate"] + np.pi
        dict["con"] = con2
    elif max(count1,count2,count3,count4) == count3:
        dict["angel"] = con["rotate"] + np.pi/2
        dict["con"] = con3
    elif max(count1,count2,count3,count4) == count4:
        dict["angel"] = con["rotate"] + 3*np.pi/2
        dict["con"] = con4
    dict["cen"] = con['center']
    dict["scale"] = scale_rate


    # debug
    # mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # cv2.drawContours(mask, [ con["con"]], -1, 255, thickness=cv2.FILLED)
    # cv2.imshow(' mask',  mask)
    # ##########################################################

    initial_tx, initial_ty = dict["cen"]
    initial_angel = 0.0
    initial_scale = 1.0
    init_params = [initial_angel, initial_scale, initial_tx, initial_ty]
    learning_rate = 0.001
    num_iterations = 500
    tolerance = 1e-5
    optimized_params = gradient_descent_individual(dict["con"], con["con"], (image.shape[0], image.shape[1]), init_params,learning_rate, num_iterations,tolerance)
    optimized_angle, optimized_scale, x,y = optimized_params
    print(optimized_params)
    # result
    optimized_contour = rotate_contour(dict["con"],optimized_angle, optimized_scale,(x,y),dict["cen"])

    # debug
    # mask1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # cv2.drawContours(mask1, [optimized_contour], -1, 255, thickness=cv2.FILLED)
    # cv2.imshow(' mask1',  mask1)
    # cv2.waitKey(0)
    # ##########################################################


    match_score = cv2.matchShapes(optimized_contour, con["con"], cv2.CONTOURS_MATCH_I1, 0.0)
    print(match_score)
    if match_score <= 0.5:
        # print(f"Shape match score: {match_score}")
        # print(f"center: {con['center']}")
        # print(f"rotate: {np.degrees(dict['angel'])}")
        # print("***********************************************")

        # display2see
        # cv2.drawContours(image, [con["con"]], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.drawContours(image, [optimized_contour], -1, (127, 255, 255), 1, lineType=cv2.LINE_AA)
        # cv2.drawContours(image, [dict["con"]], -1, (127, 0, 255), 1, lineType=cv2.LINE_AA)
        # center = con['center']
        # cv2.circle(image, center, 3, (255, 0, 0), -1)
        # angle = -dict['angel']
        # length = 50
        # end_x = int(center[0] + length * np.cos(angle))
        # end_y = int(center[1] + length * np.sin(angle))
        # end = (end_x, end_y)
        # cv2.arrowedLine(image, center, end, (0, 0, 255), 1,cv2.LINE_AA)
        cv2.imshow('Matched Contours', image)
        cv2.waitKey(0)
cv2.destroyAllWindows()
