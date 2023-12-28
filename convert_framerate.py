# import modules 
import cv2

# open file 
cap = cv2.VideoCapture('/content/video.mp4') 

# get FPS of input video 
fps = cap.get(cv2.CAP_PROP_FPS) 

# define output video and it's FPS 
output_file = 'output.mp4'
output_fps = fps * 2

# define VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_file, fourcc, output_fps, 
					(int(cap.get(3)), int(cap.get(4)))) 

# read and write frams for output video 
while cap.isOpened(): 
	ret, frame = cap.read() 
	if not ret: 
		break

	out.write(frame) 

# release resources 
cap.release() 
out.release() 
cv2.destroyAllWindows() 

# download output video on local machine 
files.download(output_file) 
