import cv2

capture = cv2.VideoCapture(0)

num_img = 0

while capture.isOpened():

    success, img_mono = capture.read()

    key = cv2.waitKey(5)

    if key == ord('q'):
        print("Quit capturing process")
        break

    elif key == ord('s'):
        cv2.imwrite('images/image_' + str(num_img) + '.png', img_mono)
        print("you saved " + str(num_img) + "th image")
        num_img += 1
    
    cv2.imshow('image_mono', img_mono)

capture.release()
cv2.destroyAllWindows()
    
