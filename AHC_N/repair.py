import cv2

img = cv2.imread('../datasets/test_data_v2/658c27f36a144c1a8079ac1dab68d0a8.jpg')

if img is not None:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
