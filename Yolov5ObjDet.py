import cv2
import torch
import os

# Load model
# Set device
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Check if CUDA is available
if torch.cuda.is_available():
    model.cuda()  # Assign the model to run on GPU
else:
    model.cpu()  # Assign the model to run on CPU
    print("CUDA is not available. Running on CPU instead.")

# preparing folder




# Check if the folder already exists
if not os.path.exists('./Saved Images'):
# If it doesn't exist, create the folder
    os.makedirs('./Saved Images')

frame_num = 0
# loading camera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame_num += 1
    results = model(frame)
    print(type(results))
    res_ten = results.xyxy[0]  # im predictions (tensor) this is a 6 element tensor
    res_pan = results.pandas().xyxyn[0]  # im predictions (tensor) this is a 6 element tensor

    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2

    # Choose text color (BGR format)
    text_color = (0, 255, 0)  # Green color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    for i, res in enumerate(res_ten):
        res_arr = res.cpu().numpy()
        print(res_arr)
        # print(type(res_arr[0]))

        start_point = (int(res_arr[0]), int(res_arr[1]))
        end_point = (int(res_arr[2]), int(res_arr[3]))

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Define text and position
        obj_name = res_pan['name'][i]
        text_position = (int(res_arr[0]), int(res_arr[1]-20))  # (x, y) coordinates
        # Write text on the image
        cv2.putText(frame, obj_name+' '+f"{res_arr[4]:.2f}", text_position, font, font_scale, text_color, font_thickness)

    cv2.imshow("preview", frame)
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press and get the key value
    if key == ord('s') or key == ord('S'):  # If 'S' is pressed
        cv2.imwrite('./Saved Images/frame'+f"{frame_num}"+'.jpg', frame)  # Save the frame as 'saved_frame.jpg'
        print("Frame saved.")
    elif key == 27:  # If 'ESC' is pressed
        break  # Exit the loop

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
