import cv2
from nicolai import depth_midas
import matplotlib.pyplot as plt
import numpy as np

image_path = 'images_depth/People/man_1m2.jpg'
image_path_2 = 'images_depth/People/man_1m8.jpg'
image_path_3 = 'images_depth/People/man_2m4.jpg'
image_path_4 = 'images_depth/People/man_3m.jpg'
image_path_5 = 'images_depth/People/man_3m6.jpg'
image_path_8 = 'output_images/image_detect.jpg'

# image_path_6 = 'images_depth/1m2.jpg'
# image_path_7 = 'images_depth/1m4.jpg'


image = cv2.imread(image_path)
image_2 = cv2.imread(image_path_2)
image_3 = cv2.imread(image_path_3)
image_4 = cv2.imread(image_path_4)
image_5 = cv2.imread(image_path_5)
# image_6 = cv2.imread(image_path_6)
# image_7 = cv2.imread(image_path_7)
image_8 = cv2.imread(image_path_8)

depth_map, original_depth_map = depth_midas(image)
depth_map_2, original_depth_map_2 = depth_midas(image_2)
depth_map_3, original_depth_map_3 = depth_midas(image_3)
depth_map_4, original_depth_map_4 = depth_midas(image_4)
depth_map_5, original_depth_map_5 = depth_midas(image_5)
# depth_map_6, original_depth_map_6 = depth_midas(image_6)
# depth_map_7, original_depth_map_7 = depth_midas(image_7)

og_point = (593,229)
# (x,y)

point = (255,210)
point_2 = (294, 232)
point_3 = (298, 236)
point_4 = (310, 246)
point_5 = (322, 239)
# point_6 = (345, 260)
# point_7 = (340, 255)

k = 3

#shape is (y,x)

print("m4")

print(f"Point mean value (7x7): {np.mean(original_depth_map[point[1]-k:point[1]+k, point[0]-k:point[0]+k])}")
print(f"Point value: {original_depth_map[point[1]][point[0]]}")

depth_map[point[1]][point[0]] = 255

print(f"Max pixel value: {np.max(original_depth_map)}")
print(f"Calibrate point value: {original_depth_map[og_point[1]][og_point[0]]}")
print(f"Calibrate point value (7x7): {np.mean(original_depth_map[og_point[1]-k:og_point[1]+k, og_point[0]-k:og_point[0]+k])}")
print("--------------------")

print("m6")
print(f"Point mean value (7x7): {np.mean(original_depth_map_2[point_2[1]-k:point_2[1]+k, point_2[0]-k:point_2[0]+k])}")
print(f"Point value: {original_depth_map_2[point_2[1]][point_2[0]]}")

depth_map_2[point_2[1]][point_2[0]] = 255

print(f"Max pixel value: {np.max(original_depth_map_2)}")
print(f"Calibrate point value: {original_depth_map_2[og_point[1]][og_point[0]]}")
print(f"Calibrate point value (7x7): {np.mean(original_depth_map_2[og_point[1]-k:og_point[1]+k, og_point[0]-k:og_point[0]+k])}")
print("--------------------")

print("m8")
print(f"Point mean value (7x7): {np.mean(original_depth_map_3[point_3[1]-k:point_3[1]+k, point_3[0]-k:point_3[0]+k])}")
print(f"Point value: {original_depth_map_3[point_3[1]][point_3[0]]}")

depth_map_3[point_3[1]][point_3[0]] = 255

print(f"Max pixel value: {np.max(original_depth_map_3)}")
print(f"Calibrate point value: {original_depth_map_3[og_point[1]][og_point[0]]}")
print(f"Calibrate point value (7x7): {np.mean(original_depth_map_3[og_point[1]-k:og_point[1]+k, og_point[0]-k:og_point[0]+k])}")
print("--------------------")

print("1m")
print(f"Point mean value (7x7): {np.mean(original_depth_map_4[point_4[1]-k:point_4[1]+k, point_4[0]-k:point_4[0]+k])}")
print(f"Point value: {original_depth_map_4[point_4[1]][point_4[0]]}")

depth_map_4[point_4[1]][point_4[0]] = 255

print(f"Max pixel value: {np.max(original_depth_map_4)}")
print(f"Calibrate point value: {original_depth_map_4[og_point[1]][og_point[0]]}")
print(f"Calibrate point value (7x7): {np.mean(original_depth_map_4[og_point[1]-k:og_point[1]+k, og_point[0]-k:og_point[0]+k])}")
print("--------------------")

print("1m2")
print(f"Point mean value (7x7): {np.mean(original_depth_map_5[point_5[1]-k:point_5[1]+k, point_5[0]-k:point_5[0]+k])}")
print(f"Point value: {original_depth_map_5[point_5[1]][point_5[0]]}")

depth_map_5[point_5[1]][point_5[0]] = 255

print(f"Max pixel value: {np.max(original_depth_map_5)}")
print(f"Calibrate point value: {original_depth_map_5[og_point[1]][og_point[0]]}")
print(f"Calibrate point value (7x7): {np.mean(original_depth_map_5[og_point[1]-k:og_point[1]+k, og_point[0]-k:og_point[0]+k])}")
print("--------------------")

# print(f"Point mean value (7x7): {np.mean(original_depth_map_6[point_6[1]-k:point_6[1]+k, point_6[0]-k:point_6[0]+k])}")
# print(f"Max pixel value: {np.max(original_depth_map_6)}")
# print(f"Point value: {original_depth_map_6[point_6[1]][point_6[0]]}")
# print("--------------------")

# print(f"Point mean value (7x7): {np.mean(original_depth_map_7[point_7[1]-k:point_7[1]+k, point_7[0]-k:point_7[0]+k])}")
# print(f"Max pixel value: {np.max(original_depth_map_7)}")
# print(f"Point value: {original_depth_map_7[point_7[1]][point_7[0]]}")
# print("--------------------")

plt.figure()
plt.imshow(depth_map)
plt.show()

plt.figure()
plt.imshow(depth_map_2)
plt.show()

plt.figure()
plt.imshow(depth_map_3)
plt.show()

plt.figure()
plt.imshow(depth_map_4)
plt.show()

plt.figure()
plt.imshow(depth_map_5)
plt.show()

# plt.figure()
# plt.imshow(depth_map_6)
# plt.show()

# plt.figure()
# plt.imshow(depth_map_7)
# plt.show()

plt.figure()
plt.imshow(image_8)
plt.show()