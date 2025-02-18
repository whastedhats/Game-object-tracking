import cv2
import numpy as np
from mss import mss
from time import time

from datetime import datetime

# Initialize screen capture 
sct = mss()
monitor_width = 800
monitor_height = 600
lt = sct.monitors[0]['width']-monitor_width
monitor = {"top": 0, "left": lt, "width": monitor_width, "height": monitor_height} # Define capture area

# 紀錄水平位置
L_line = -1
track_width = 56



def log( content , filename = 'log.txt'):
    with open(filename, 'a') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{current_time} : {content}\n")
        
def process_frame_color_mask( frame ):
    # Convert frames to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定義顏色範圍 (例如：藍色)
    lower_bound = np.array([23, 250, 250])   # HSV 下限
    upper_bound = np.array([27, 255, 255])  # HSV 上限

    # Define a mask using the lower and upper bounds of the color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # x , y = 0 , 599
    # h,s,v= hsv[y,x]
    # print(f" h={h} , s={s} , v={v}")
    # Apply the mask to the original frame
    color_mask = cv2.bitwise_and(frame, frame, mask=mask)
    
 
    horizontal_projection = np.divide(np.sum(color_mask, axis=1),255)  # Sum along rows
    vertical_projection = np.divide(np.sum(color_mask, axis=0),255)    # Sum along columns
    # print( horizontal_projection.shape)#height:460
    # print( vertical_projection.shape)#width:620
    # log_record= ''
    # w = monitor.get('width')
    # wd=int(w/10)
    # for i in range(0,w,wd):
    #     total = 0
    #     for j in range(0,wd):
    #         total += vertical_projection[i+j]
    #     log_record += "{:6.0f} ".format(total)
    # log(log_record)
    
    
    start_y = -1 
    end_y = -1
    start_x = -1
    end_x = -1

    hy = np.where(horizontal_projection > 10)[0]
    if len(hy) > 0:
        start_y = hy[0]
        end_y = hy[-1]
    
    hx = np.where(vertical_projection > 10)[0]
    if len(hx)>0 :
        start_x = hx[0] 
        end_x = hx[-1]

    print(start_x , end_x , start_y , end_y)    
    return color_mask , start_x , end_x , start_y , end_y


# Define a function to calculate frame differences
def process_frame_diff2(curr_frame):

    # Convert frames to HSV
    hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    # 定義顏色範圍 (例如：藍色)
    lower_bound = np.array([0, 0, 80])   # HSV 下限
    upper_bound = np.array([255, 255, 255])  # HSV 上限

    # Define a mask using the lower and upper bounds of the color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # x , y = 0 , 599
    # h,s,v= hsv[y,x]
    # print(f" h={h} , s={s} , v={v}")
    # Apply the mask to the original frame
    color_mask = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)


    # Convert frames to grayscale    
    curr_gray = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    
    
    
    # Apply a binary threshold to highlight motion
    _, thresh = cv2.threshold(curr_gray, 50, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(curr_gray, 255, 255, cv2.THRESH_BINARY)

    # 依照軌道位置切割
    split_indices = [L_line, L_line+track_width, L_line+track_width*2, L_line+track_width*3, L_line+track_width*4]
    sub_arrays = np.hsplit(thresh, split_indices)

    # 顯示結果
    # for i, sub_array in enumerate(sub_arrays):
    #     print(f"Sub-array {i}:\n{sub_array.shape}\n")

    # 四個軌道，取陣列 index 1,2,3,4
    horizontal_projection1 = np.divide(np.sum(sub_arrays[1], axis=1),255) 
    horizontal_projection2 = np.divide(np.sum(sub_arrays[2], axis=1),255)
    horizontal_projection3 = np.divide(np.sum(sub_arrays[3], axis=1),255)
    horizontal_projection4 = np.divide(np.sum(sub_arrays[4], axis=1),255)



    # horizontal_projection = np.divide(np.sum(thresh, axis=1),255)  # Sum along rows
    # vertical_projection = np.divide(np.sum(thresh, axis=0),255)    # Sum along columns
    # print( horizontal_projection.shape)#height:460
    # print( vertical_projection.shape)#width:620
    # log_record= ''
    # w = monitor.get('width')
    # wd=int(w/10)
    # for i in range(0,w,wd):
    #     total = 0
    #     for j in range(0,wd):
    #         total += vertical_projection[i+j]
    #     log_record += "{:6.0f} ".format(total)
    # log(log_record)
    
    
    start_y = -1 
    end_y = -1
    start_x = -1
    end_x = -1

    hy = np.where(horizontal_projection1 > 10)[0]
    if len(hy) > 0:
        start_y = hy[0]
        end_y = hy[-1]
    

    block_sy=-1
    block_ey=-1
    block_list =[]
    for i in range(len(hy)):
        if i==0:
            block_sy = hy[i]
            block_ey = hy[i]
        else:
            if hy[i] - hy[i-1] > 1 :
                block_ey = hy[i-1]
                block_list.append((block_sy,block_ey))
                # new block
                block_sy = hy[i]
            if  i == len(hy)-1:
                block_ey = hy[i]
                block_list.append((block_sy,block_ey))
    # log(f"hp:{horizontal_projection1}")
    # log(f"hy:{hy}")
    # log(f"bl:-----------")
    # for i in range(len(block_list)):
    #     log(f"bl:{block_list[i]}")
    # log(f"bl:end-----------")

    start_x = L_line
    end_x = L_line+track_width
    start_x = L_line
    end_x = L_line+track_width
    
    # Dilate the thresholded image to fill in gaps
    # dilated = cv2.dilate(thresh, None, iterations=2)
    # print(start_x , end_x , start_y , end_y)
    return thresh2 , start_x , end_x , start_y , end_y , block_list


# Define a function to calculate frame differences
def process_frame_diff(prev_frame, curr_frame, threshold=25):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between frames
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Apply a binary threshold to highlight motion
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(diff, 500, 255, cv2.THRESH_BINARY)

    # 依照軌道位置切割
    split_indices = [L_line, L_line+track_width, L_line+track_width*2, L_line+track_width*3, L_line+track_width*4]
    sub_arrays = np.hsplit(thresh, split_indices)

    # 顯示結果
    # for i, sub_array in enumerate(sub_arrays):
    #     print(f"Sub-array {i}:\n{sub_array.shape}\n")

    # 四個軌道，取陣列 index 1,2,3,4
    horizontal_projection1 = np.divide(np.sum(sub_arrays[1], axis=1),255) 
    horizontal_projection2 = np.divide(np.sum(sub_arrays[2], axis=1),255)
    horizontal_projection3 = np.divide(np.sum(sub_arrays[3], axis=1),255)
    horizontal_projection4 = np.divide(np.sum(sub_arrays[4], axis=1),255)



    # horizontal_projection = np.divide(np.sum(thresh, axis=1),255)  # Sum along rows
    # vertical_projection = np.divide(np.sum(thresh, axis=0),255)    # Sum along columns
    # print( horizontal_projection.shape)#height:460
    # print( vertical_projection.shape)#width:620
    # log_record= ''
    # w = monitor.get('width')
    # wd=int(w/10)
    # for i in range(0,w,wd):
    #     total = 0
    #     for j in range(0,wd):
    #         total += vertical_projection[i+j]
    #     log_record += "{:6.0f} ".format(total)
    # log(log_record)
    
    
    start_y = -1 
    end_y = -1
    start_x = -1
    end_x = -1

    hy = np.where(horizontal_projection1 > 10)[0]
    if len(hy) > 0:
        start_y = hy[0]
        end_y = hy[-1]
    

    block_sy=-1
    block_ey=-1
    block_list =[]
    for i in range(len(hy)):
        if i==0:
            block_sy = hy[i]
            block_ey = hy[i]
        else:
            if hy[i] - hy[i-1] > 1 :
                block_ey = hy[i-1]
                block_list.append((block_sy,block_ey))
                # new block
                block_sy = hy[i]
            if  i == len(hy)-1:
                block_ey = hy[i]
                block_list.append((block_sy,block_ey))
    # log(f"hp:{horizontal_projection1}")
    # log(f"hy:{hy}")
    # log(f"bl:-----------")
    # for i in range(len(block_list)):
    #     log(f"bl:{block_list[i]}")
    # log(f"bl:end-----------")

    start_x = L_line
    end_x = L_line+track_width
    start_x = L_line
    end_x = L_line+track_width
    
    # Dilate the thresholded image to fill in gaps
    # dilated = cv2.dilate(thresh, None, iterations=2)
    # print(start_x , end_x , start_y , end_y)
    return thresh2 , start_x , end_x , start_y , end_y , block_list

# Initialize variables
prev_frame = None
fps = 0
last_time = time()

try:
    while True:
        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        curr_frame  = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # Remove alpha channel
        # Process for motion tracking
        # if prev_frame is not None:
        #     motion_mask , x_s , x_e , y_s , y_e= process_frame_diff(prev_frame, curr_frame)            

        #     if x_s>-1 and x_e>-1 and  y_s>-1 and y_e>-1:
        #         cv2.rectangle(motion_mask, (x_s, y_s), (x_e, y_e), (255, 255, 255), 2)

        #     # Display motion tracking
        #     cv2.imshow("Motion Mask", motion_mask)
        #     cv2.imshow("Current Frame", curr_frame)

        # # Update previous frame
        # prev_frame = curr_frame

        if L_line == -1:
            color_mask, x_s , x_e , y_s , y_e = process_frame_color_mask(curr_frame)
            if x_s>-1 and x_e>-1 and y_s>-1 and y_e>-1:
                s = x_s-80
                L_line = s
                wd = 56
                cv2.rectangle(curr_frame, (s, y_s), (s+wd, y_e), (255, 255, 255), 2)
                cv2.rectangle(curr_frame, (s+wd, y_s), (s+wd*2, y_e), (255, 255, 255), 2)
                cv2.rectangle(curr_frame, (s+wd*2, y_s), (s+wd*3, y_e), (255, 255, 255), 2)
                cv2.rectangle(curr_frame, (s+wd*3, y_s), (s+wd*4, y_e), (255, 255, 255), 2)
                cv2.rectangle(curr_frame, (460, 520), (570,520), (255, 255, 255), 2) # 底線位置520
                print(f"s:{s}")
                ##位置: 422,423
            cv2.imshow("Color Mask", color_mask)    
            cv2.imshow("Current Frame", curr_frame)
        else:
            if prev_frame is not None:
                motion_mask , x_s , x_e , y_s , y_e , roi = process_frame_diff2( curr_frame)            

                # if x_s>-1 and x_e>-1 and  y_s>-1 and y_e>-1:
                #     cv2.rectangle(motion_mask, (x_s, y_s), (x_e, y_e), (255, 255, 255), 2)

                if roi is not None:
                    for i in range(len(roi)):
                        cv2.rectangle(motion_mask, (L_line, roi[i][0]), (L_line+track_width, roi[i][1]), (255, 255, 255), 2)


                # Display motion tracking
                cv2.imshow("Motion Mask", motion_mask)
                cv2.imshow("Current Frame", curr_frame)

            # Update previous frame
            prev_frame = curr_frame
        
        
        # Display FPS
        fps = 1 / (time() - last_time)
        last_time = time()
        print(f"FPS: {fps:.2f}")

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program stopped.")

# Clean up
cv2.destroyAllWindows()
