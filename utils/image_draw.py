import cv2
from utils_2 import draw_frame_division, draw_position_line, draw_position_line_2, draw_depth_cal, draw_box_cal

image = cv2.imread('output_images/depth_map.jpg')
image_2 = cv2.imread('output_images/normal.jpg')
image_3 = cv2.imread('output_images/image_detect.jpg')
image_4 = cv2.imread('output_images/depth.jpg')

shape = (480,640,3)

image = cv2.resize(image, (shape[1] , shape[0]))
image_2 = cv2.resize(image_2, (shape[1] , shape[0]))
image_3 = cv2.resize(image_3, (shape[1] , shape[0]))
image_4 = cv2.resize(image_4, (shape[1] , shape[0]))

image = image/255
image_2 = image_2/255
image_3 = image_3/255
image_4 = image_4/255

#Dummy bounding box
# bounding_box = [[23,121,233,234]]

draw_frame_division(image)
draw_position_line(image_2)
draw_position_line_2(image_3)
draw_box_cal(image_4, [467,374,583,480])

draw_depth_cal(image)

image = cv2.convertScaleAbs(image, alpha=(255.0))
image_2 = cv2.convertScaleAbs(image_2, alpha=(255.0))
image_3 = cv2.convertScaleAbs(image_3, alpha=(255.0))
image_4 = cv2.convertScaleAbs(image_4, alpha=(255.0))

image_4 = cv2.putText(image_4,'Object chair has mean center = 177.78',(150,16+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv2.LINE_AA)

# cv2.imwrite('output_images/normal_position.jpg', image)
# cv2.imwrite('output_images/normal_position_2.jpg', image_2)
# cv2.imwrite('output_images/normal_position_3.jpg', image_3)
cv2.imwrite('output_images/normal_position_4.jpg', image_4)

