import numpy as np
import cv2
import rgb_l515

capture_left = cv2.VideoCapture(0)

num_img = 0

while capture_left.isOpened():

    success_L, img_L = capture_left.read()
    frames = rgb_l515.pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    images_l515 = color_image
    
    key = cv2.waitKey(5)

    if key == ord('q'):
        print("Quit capturing process")
        break

    elif key == ord('s'):
        cv2.imwrite('images/left_images/left_image_' + str(num_img) + '.png', img_L)
        cv2.imwrite('images/right_images/right_image_' + str(num_img) + '.png', images_l515)
        print("you saved " + str(num_img) + "th image")
        num_img += 1

    cv2.imshow('image 1', img_L)
    cv2.imshow('image 2', images_l515)

capture_left.release()

cv2.destroyAllWindows()

