import cv2
from pyzbar.pyzbar import decode
from tkinter import PhotoImage,ttk
import tkinter as tk
import threading
import numpy as np
from PIL import Image,ImageTk
import math
from itertools import combinations
from pyscreenshot import grab

cap = cv2.VideoCapture(0)

clicked_points = []     #畫過的點座標
object_height = 11      #相機高度
low = 30
high = 120
threshold = 150
edge = 2
shape = (400,300)
button_case = 0
point = [(),(),()]
photo_list_ruler = []
boolen = True
dirictionresult = 0
lengthresult = 0
angleresult = 0
def calculate_distance(point1, point2):
    # 計算像素距離
    pixel_distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # 計算物理距離
    physical_distance = (pixel_distance / shape[0]) * object_height

    return physical_distance
def remove_duplicate_point(corners, threshold=20):
    if len(corners)==3:
        return -1
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            distance = np.linalg.norm(corners[i] - corners[j])
            if distance < threshold:
                return i  # 返回需要刪除的點的位置索引

    return -1

#使順序正常
def sort_remaining_points(remaining_points):
    sorted_points = np.zeros((4, 2), dtype=int)
    distances = []

    # 計算任意兩點間的距離，並記錄最短距離的兩條邊的共同點
    for i in range(len(remaining_points)):
        for j in range(i + 1, len(remaining_points)):
            dist = np.linalg.norm(remaining_points[i] - remaining_points[j])
            distances.append((i, j, dist))


    distances.sort(key=lambda x: x[2])


    sorted_points[0] = remaining_points[distances[0][0]]
    sorted_points[1] = remaining_points[distances[0][1]]
    sorted_points[3] = remaining_points[distances[1][1]]


    sorted_points[2] = [-1,-1]

    return sorted_points
    
def angle():
    global point,boolen
    def on_mouse_click(event):
        global canvas,point,boolen
        if event.x > shape[0]:
            prev_x = shape[0]
        elif event.x <0:
            prev_x = 0
        else:
            prev_x = event.x
        if event.y > shape[1]:
            prev_y = shape[1]
        elif event.y <0:
            prev_y = 0
        else:
            prev_y = event.y
        if len(point[2]) ==2 or len(point[1]) == 0:
            point = [(prev_x,prev_y),(),()]
        else:
            boolen = True
            point[2] = (prev_x,prev_y)
       
    def on_mouse_drag(event):
        global canvas,point,boolen
        
        if event.x > shape[0]:
            dragev_x = shape[0]
        elif event.x <0:
            dragev_x = 0
        else:
            dragev_x = event.x
        if event.y > shape[1]:
            dragev_y = shape[1]
        elif event.y < 0:
            dragev_y = 0
        else:
            dragev_y = event.y 
        if len(point[2]) == 0:
            point[1] = (dragev_x,dragev_y)
        else :point[2] = (dragev_x,dragev_y)
        boolen = True
        
    def on_mouse_release(event):
        global canvas,point,boolen
        if event.x > shape[0]:
            dragev_x = shape[0]
        elif event.x <0:
            dragev_x = 0
        else:
            dragev_x = event.x
        if event.y > shape[1]:
            dragev_y = shape[1]
        elif event.y <0:
            dragev_y = 0
        else:
            dragev_y = event.y 
        if len(point[2]) == 0:
            point[1] = (dragev_x,dragev_y)
        else :point[2] = (dragev_x,dragev_y)
    #    boolen = True
    ret, frame = cap.read() 
    frame = cv2.resize(frame,shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=arr)
    canvas.delete("all")
    canvas.bind("<Button-1>", on_mouse_click)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)
    #tk.image = photo
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    if len(point[1]) == 2:
        canvas.create_line(point[0][0],point[0][1],point[1][0],point[1][1],width=3,fill='red')
        if len(point[2]) == 2:
            canvas.create_line(point[2][0],point[2][1],point[1][0],point[1][1],width=3,fill='red')
            canvas.create_text(point[1][0], point[1][1], text=f'{calculate_angle(np.array([point[1][0] - point[0][0], point[1][1] - point[0][1]]),np.array([point[1][0] - point[2][0], point[1][1] - point[2][1]])):.3f}', fill="red")
            '''
            if boolen:
                boolen = False
                text_label.configure(state='normal')
                text_label.insert(1.0,f'{calculate_angle(np.array([point[1][0] - point[0][0], point[1][1] - point[0][1]]),np.array([point[1][0] - point[2][0], point[1][1] - point[2][1]])):.3f}°\n')
                text_label.configure(state='disabled')
            '''
    if button_case ==3:
        angle()
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def ruler():
    global point,boolen,photo
    def on_mouse_click(event):
        global canvas,point,boolen
        if event.x > shape[0]:
            prev_x = shape[0]
        elif event.x <0:
            prev_x = 0
        else:
            prev_x = event.x
        if event.y > shape[1]:
            prev_y = shape[1]
        elif event.y <0:
            prev_y = 0
        else:
            prev_y = event.y
        point[0] = (prev_x,prev_y)
        point[1] = ()

    def on_mouse_drag(event):
        global canvas,point,boolen
        
        if event.x > shape[0]:
            dragev_x = shape[0]
        elif event.x <0:
            dragev_x = 0
        else:
            dragev_x = event.x
        if event.y > shape[1]:
            dragev_y = shape[1]
        elif event.y < 0:
            dragev_y = 0
        else:
            dragev_y = event.y 
        point[1] = (dragev_x,dragev_y)
        boolen = True
    def on_mouse_release(event):
        global canvas,point,boolen
        if event.x > shape[0]:
            dragev_x = shape[0]
        elif event.x <0:
            dragev_x = 0
        else:
            dragev_x = event.x
        if event.y > shape[1]:
            dragev_y = shape[1]
        elif event.y <0:
            dragev_y = 0
        else:
            dragev_y = event.y 
        point[1] = (dragev_x,dragev_y)
        boolen = True
    
    if len(point[1]) == 2:
        canvas.create_line(point[0][0],point[0][1],point[1][0],point[1][1],width=3,fill='red')
        canvas.create_text(point[1][0], point[1][1]+10, text=f'{calculate_distance(point[0],point[1]):.3f}cm', fill="red")
        '''
        if boolen:
            boolen = False
            text_label.configure(state='normal')
            text_label.insert(1.0,f'{calculate_distance(point[0],point[1]):.3f}cm\n')
            text_label.configure(state='disabled')
        '''    
    ret, frame = cap.read() 
    frame = cv2.resize(frame,shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=arr)
    canvas.delete("all")
    canvas.bind("<Button-1>", on_mouse_click)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)
    #tk.image = photo
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    if button_case ==1:
        ruler()

def draw_dashed_line(image, start_point, end_point, color, thickness, dash_length):
    x1, y1 = start_point
    x2, y2 = end_point

    # 計算線段的總長度
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # 計算單位向量
    unit_vector = ((x2 - x1) / line_length, (y2 - y1) / line_length)

    # 計算虛線的數量
    num_dashes = int(line_length / dash_length)

    # 畫出每一段虛線
    for i in range(num_dashes):
        start = (int(x1 + i * unit_vector[0] * dash_length), int(y1 + i * unit_vector[1] * dash_length))
        end = (int(x1 + (i + 0.5) * unit_vector[0] * dash_length), int(y1 + (i + 0.5) * unit_vector[1] * dash_length))
        cv2.line(image, start, end, color, thickness)
           
def qrcode():    
    ret, frame = cap.read() 
    #frame = cv2.imread("4.jpg")
    
    # 讀取一張影像
    #image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    # 將image的內容複製到frame
    #np.copyto(frame, image)
    # 解碼 QR 碼
    decoded_objects = decode(frame)

    # 顯示解碼結果
    for obj in decoded_objects:
        # 取得 QR 碼的位置
        points = obj.polygon
        if points is not None and len(points) == 4:
            points = np.array(points, dtype=np.int32)  # 將 Python 列表轉換為 NumPy 陣列
            hull = cv2.convexHull(points)
            cv2.polylines(frame, [hull], True, (0, 255, 0), 2)

            # 輸出角點座標
            #print(f'QR Code Points: {points}')
            qr_code_data = obj.data.decode('utf-8')
            # 取得 QR 碼的文本
            #print(f'QR Code Data: {qr_code_data}')
            '''
            if qr_code_data!= text_label.get('1.0', '1.end-1c'):
                print(qr_code_data)
                print(text_label.get('1.0','1.end'))
                text_label.configure(state='normal')
                text_label.insert('1.0',f'{qr_code_data}\n')
                text_label.configure(state='disabled')
            '''
    frame = cv2.resize(frame,shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=arr)
    kernel = np.ones((1, 1), np.uint8)
    #v_label.config(image = photo)
    #v_label.image = photo
    #canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    #tk.image = photo
    if button_case == 2:
        qrcode()
       

def videoloop():
    # 擷取影像
    ret, frame = cap.read() 
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return
    #image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    # 將image的內容複製到frame
    #np.copyto(frame, image)
    # 彩色轉灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 影像去雜訊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 邊緣偵測
    if edge == 1:
        canny = cv2.Canny(gray, low, high)
    if edge == 2:
        canny = cv2.Canny(blurred, low, high)
    kernel= np.ones((3,3),np.uint8)
    canny = cv2.dilate(canny,kernel,iterations=1)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold, None, 10, 0)

    # 設置閥值
    angle_threshold = 5
    coord_threshold = 10
    length_threshold = 10


    drawn_points = set()

    if lines is not None:
        xmax=0
        ymax=xmax
        xmin=255
        ymin=xmin
        
        for i in range(0, len(lines)):
            #l = lines[i][0]
            #image2 = frame
            #cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            
            '''
            if(l[0]>xmax):
                xmax=l[0]
            if(l[2]>xmax):
                xmax=l[2]
            
            if(l[1]>ymax):
                ymax=l[1]
            if(l[3]>ymax):
                ymax=l[3]

            if(l[0]<xmin):
                xmin=l[0]
            if(l[2]<xmin):
                xmin=l[2]
            
            if(l[1]<ymin):
                ymin=l[1]
            if(l[3]<ymin):
                ymin=l[3]
            '''
            #取四個角

            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]

                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                vector1 = np.array([x2 - x1, y2 - y1])
                vector2 = np.array([x4 - x3, y4 - y3])

                angle = calculate_angle(vector1, vector2)

                length1 = np.linalg.norm(vector1)
                length2 = np.linalg.norm(vector2)

                endpoints_match = (
                        (abs(x1 - x3) < coord_threshold and abs(y1 - y3) < coord_threshold) or
                        (abs(x1 - x4) < coord_threshold and abs(y1 - y4) < coord_threshold) or
                        (abs(x2 - x3) < coord_threshold and abs(y2 - y3) < coord_threshold) or
                        (abs(x2 - x4) < coord_threshold and abs(y2 - y4) < coord_threshold)
                )

                if endpoints_match and abs(
                        angle - 90) < angle_threshold and abs(length1 - length2) < length_threshold:

                    point = (x1, y1)
                    if point not in drawn_points:

                        drawn_points.add(point)

                    point = (x2, y2)
                    if point not in drawn_points:

                        drawn_points.add(point)

                    point = (x3, y3)
                    if point not in drawn_points:

                        drawn_points.add(point)

                    point = (x4, y4)
                    if point not in drawn_points:

                        drawn_points.add(point)

        #畫箭頭並印出其與斜率無窮大的邊的夾角

        all_rectangles = []

        point_combinations = list(combinations(drawn_points, 4))

        for combo in point_combinations:
            points = np.array(list(combo), dtype=np.int32)

            rect = cv2.convexHull(points)

            all_rectangles.append(rect)

        max_area = 0
        max_rectangle = None

        for rect in all_rectangles:
            area = cv2.contourArea(rect)
            if area > max_area:
                max_area = area
                max_rectangle = rect
        if max_rectangle is not None:
            max_rectangle_corners = max_rectangle.reshape(-1, 2)
            if len(max_rectangle_corners)>=3 and len(max_rectangle_corners)<=4:
                index_to_remove = remove_duplicate_point(max_rectangle_corners)
                if index_to_remove != -1:
                    remaining_points = np.delete(max_rectangle_corners, index_to_remove, axis=0)
                    remaining_points = remaining_points[remaining_points[:, 0].argsort()]
                    sorted_points = sort_remaining_points(remaining_points)
                    vector_sum = sorted_points[1] - sorted_points[0] + sorted_points[3] - sorted_points[0]
                    sorted_points[2] = sorted_points[0] + vector_sum
                    max_rectangle_corners = sorted_points
                for i in range(len(max_rectangle_corners)):
                        i1=(i+1)%len(max_rectangle_corners)
                        #算出實際距離
                        point_distance=calculate_distance(max_rectangle_corners[i], max_rectangle_corners[i1])
                        cv2.line(frame, tuple(map(int, max_rectangle_corners[i])),tuple(map(int, max_rectangle_corners[i1])), (0, 255, 0), 2)
                        distance_text_position = (
                            int((max_rectangle_corners[i][0] + max_rectangle_corners[i1][0]) // 2) ,
                            int((max_rectangle_corners[i][1] + max_rectangle_corners[i1][1]) // 2)
                        )
                        #cv2.putText(frame, f"{point_distance:.2f}", distance_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    
                for corner in max_rectangle_corners:
                    cv2.circle(frame, tuple(corner), 5, (255, 0, 0), -1)
                slopes = []
                for i in range(len(max_rectangle_corners)):
                    x1, y1 = max_rectangle_corners[i]
                    x2, y2 = max_rectangle_corners[(i + 1) % len(max_rectangle_corners)]
                    slope = (y2 - y1) / (x2 - x1 + 1e-10)
                    slopes.append(slope)
                slope_threshold = 0.5
                flag = False  # 添加了flag來確保只繪製一次箭頭
                for i in range(len(max_rectangle_corners)):
                    current_slope = slopes[i]

                    # 在 0 和 +∞ 之間的邊（不包括 0 和負無窮）
                    if 0 <= current_slope != np.inf:
                        x1, y1 = max_rectangle_corners[i]
                        x2, y2 = max_rectangle_corners[(i + 1) % len(max_rectangle_corners)]

                        # 起點為其中一邊的中點
                        start_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                        x3, y3 = max_rectangle_corners[(i + 2) % len(max_rectangle_corners)]
                        x4, y4 = max_rectangle_corners[(i + 3) % len(max_rectangle_corners)]
                        end_point=((x3 + x4) // 2, (y3 + y4) // 2)
                        '''
                        # 找到斜率相同但不同邊的中點，作為箭頭的終點
                        end_point = None
                        
                        for j in range(len(max_rectangle_corners)):
                            if i != j and abs(slopes[i] - slopes[j]) < slope_threshold:  # 添加了阈值比较
                                x3, y3 = max_rectangle_corners[j]
                                x4, y4 = max_rectangle_corners[(j + 1) % len(max_rectangle_corners)]

                                # 確保從 x 較小的中點畫箭頭到 x 較大的中點
                                if x3 < x4:
                                    end_point = ((x3 + x4) // 2, (y3 + y4) // 2)
                                else:
                                    end_point = ((x4 + x3) // 2, (y4 + y3) // 2)
                                break
                        '''
                        if end_point is not None and flag == False:
                            # 延長箭頭線段
                            extension_factor = 0.25  # 延長因子
                            extended_start_point = (
                                int(start_point[0] - (end_point[0] - start_point[0]) * extension_factor),
                                int(start_point[1] - (end_point[1] - start_point[1]) * extension_factor)
                            )
                            extended_end_point = (
                                int(end_point[0] + (end_point[0] - start_point[0]) * extension_factor),
                                int(end_point[1] + (end_point[1] - start_point[1]) * extension_factor)
                            )
                            if extended_start_point[1]<extended_end_point[1]:
                                temp_point=extended_start_point
                                extended_start_point=extended_end_point
                                extended_end_point=temp_point
                            # 在中點處畫上延長後的箭頭
                            flag = True
                            cv2.arrowedLine(frame, extended_start_point, extended_end_point, (0, 0, 255), 2)
                            angle = np.degrees(np.arctan(abs((end_point[1] - start_point[1]) / (end_point[0] - start_point[0] + 1e-10))))

                            # 在箭頭附近繪制夹角度數
                            text_position = (
                                int((start_point[0] + extended_end_point[0]) // 2)-15,
                                int((start_point[1] + extended_end_point[1]) // 2)-35
                            )
                            cv2.putText(frame, f"{90-angle:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            degree_position = (
                                text_position[0]+50,text_position[1]-15
                            )
                            cv2.circle(frame, degree_position, 3, (0, 0, 0),2)
                            #畫虛線
                            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                            max_y=frame.argmax(axis=0).max()
                            draw_dashed_line(frame, (mid_point[0], 0), (mid_point[0], max_y), (0, 0, 0), 2, 10)
            else:
                print("not 4")

        #cv2.imshow("lines",image2)
        #cv2.imshow("Rectangles", frame)
        
        frame = cv2.resize(frame,(400,400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=arr)
        kernel = np.ones((1, 1), np.uint8)
        #canvas.delete("all")
        #tk.image = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        if button_case ==0:
            videoloop()
        
def prevSnapshot():
    global lengthresult
    if lengthresult>0 :
        lengthresult = lengthresult-1
        canvasphoto,pt = photo_list_ruler[lengthresult]
        lengthlist.delete("all")
        lengthlist.create_image(0, 0, anchor=tk.NW, image=canvasphoto)
        lengthlist.create_line(pt[0][0],pt[0][1],pt[1][0],pt[1][1],width=3,fill='red')
        lengthlist.create_text(pt[1][0], pt[1][1]+10, text=f'{calculate_distance(pt[0],pt[1]):.3f}cm', fill="red")
    return

def nextSnapshot():
    global lengthresult
    if lengthresult+1< len(photo_list_ruler):
        lengthresult = lengthresult+1
        canvasphoto,pt = photo_list_ruler[lengthresult]
        lengthlist.delete("all")
        lengthlist.create_image(0, 0, anchor=tk.NW, image=canvasphoto)
        lengthlist.create_line(pt[0][0],pt[0][1],pt[1][0],pt[1][1],width=3,fill='red')
        lengthlist.create_text(pt[1][0], pt[1][1]+10, text=f'{calculate_distance(pt[0],pt[1]):.3f}cm', fill="red")
    return
        
def TakeSnapshot():
    global button_case
    #if button_case ==0:
        
    if button_case == 1:
        global lengthlist,photo_list_ruler,lengthresult,point
        '''
        canvasphoto = ImageTk.getimage(photo)
        canvasphoto = canvasphoto.resize((300,210))
        canvasphoto = ImageTk.PhotoImage(canvasphoto)
        '''
        '''
        canvasphoto = grab(bbox=(0,0,300,400))
        canvasphoto = ImageTk.PhotoImage(canvasphoto)
        '''
        photo_list_ruler.append((photo,point))
        lengthresult = len(photo_list_ruler)-1
        lengthlist.create_image(0, 0, anchor=tk.NW, image=photo)
        lengthlist.create_line(point[0][0],point[0][1],point[1][0],point[1][1],width=3,fill='red')
        lengthlist.create_text(point[1][0], point[1][1]+10, text=f'{calculate_distance(point[0],point[1]):.3f}cm', fill="red")
    #elif button_case == 2:
        '''
    elif button_case == 3:
        global anglelist,photo_list_angle,angleresult,point
        photo_list_angle.append((photo,point))
        angleresult = len(photo_list_angle)-1
        anglelist.create_image(0, 0, anchor=tk.NW, image=photo)
        anglelist.create_line(point[2][0],point[2][1],point[1][0],point[1][1],width=3,fill='red')
        anglelist.create_text(point[1][0], point[1][1], text=f'{calculate_angle(np.array([point[1][0] - point[0][0], point[1][1] - point[0][1]]),np.array([point[1][0] - point[2][0], point[1][1] - point[2][1]])):.3f}', fill="red")
    '''
    return

    
    
width , height = 640,480

#按鈕功能
def button_command(button_number):
    global button_case,point
    canvas.delete('all')
    if button_case == 1 or button_case == 3:
        canvas.unbind("<Button-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")
        point = [(),(),()]
    for but in button:
        but.config(relief =tk.RAISED)
    if button_number == image[4]:
        button_case = 4
        button[4].config(relief = tk.SUNKEN)
        cap.release()
        window.destroy()
    elif button_number == image[1]:
        button_case = 1
        button[1].config(relief = tk.SUNKEN)
        thread_ruler = threading.Thread(target= ruler)
        thread_ruler.start() 
    elif button_number == image[2]:
        button_case = 2
        button[2].config(relief = tk.SUNKEN)
        thread_qrcode = threading.Thread(target=qrcode)
        thread_qrcode.start()
    elif button_number == image[3]:
        button_case = 3
        button[3].config(relief = tk.SUNKEN)
        thread_angle = threading.Thread(target=angle)
        thread_angle.start()
    elif button_number == image[0]:
        button_case = 0
        button[0].config(relief = tk.SUNKEN)
        thread_crop = threading.Thread(target=videoloop)
        thread_crop.start()
        


window = tk.Tk()
window.minsize(width=600, height=420)  
#window.attributes("-zoomed", True)

#按鈕
button_path = [r'./photo/crop.png'
            , r'./photo/ruler.png'
            , r'./photo/qrcode.png'
            , r'./photo/angle.png'
            , r'./photo/exit.png'
            ]
image = []
for path in button_path:
    btnicon = PhotoImage(file=path)
    btnicon = ImageTk.getimage(btnicon)
    btnicon = btnicon.resize((60,60))
    btnicon = ImageTk.PhotoImage(btnicon)
    image.append(btnicon)
button_frame = tk.Frame(window)
button_frame.pack(side=tk.TOP, pady=5, anchor='nw')
button = []
for i in image:
    button.append(tk.Button(button_frame, image=i,width=80,height=80, command=lambda btn=i: button_command(btn)))
for i in range(len(button)):    
    button[i].pack( side = 'left',padx=15,anchor= 'n')
button[0].config(relief = tk.SUNKEN)
video_frame = tk.Frame(window,width=shape[0],height=shape[1])
video_frame.pack(side=tk.LEFT, padx=1, pady=5,anchor='nw')
video_frame.pack_propagate(False)

canvas = tk.Canvas(video_frame)
canvas.pack(fill=tk.BOTH)
#NFUlogo
icon = Image.open("./photo/nfu.png")
img = ImageTk.PhotoImage(icon)
logo = tk.Label(image=img)
logo.pack()
#SnapshotButton
btnframe = tk.Frame(window,width=10,height= 230)
btnframe.pack(side=tk.LEFT)
lastbutton = tk.Button(btnframe, text="last",width=3,height=2,command=prevSnapshot)
lastbutton.grid(row=0,column=0)
pic_button = tk.Button(btnframe, text=">",width=3,height=3,command=TakeSnapshot)
pic_button.grid(row=1,column=0)
nextbutton = tk.Button(btnframe, text="next",width=3,height=2,command=nextSnapshot)
nextbutton.grid(row=2,column=0)
#video
vl_crop = threading.Thread(target=videoloop)
vl_crop.start()
#info
info_label = ttk.Notebook(window,width=350,height=210)
tab1 = ttk.Frame(info_label)
tab2 = ttk.Frame(info_label)
tab3 = ttk.Frame(info_label)
info_label.add(tab1,text='長度')
info_label.add(tab2,text='QRcode')
info_label.add(tab3,text='角度')
info_label.pack_propagate(False)
info_label.pack(side=tk.RIGHT,padx=3,pady=3,anchor = 'ne')
#lengthresult
lengthlist = tk.Canvas(tab1)
lengthlist.pack(anchor='center')
anglelist = tk.Canvas(tab1)
anglelist.pack(anchor='center')

'''#textlabel
text_label = tk.Text(tab1,state='disabled')
y_scrollbar = tk.Scrollbar(tab1, command=text_label.yview)
y_scrollbar.pack(side=tk.RIGHT,fill = tk.Y)
text_label.config(yscrollcommand = y_scrollbar.set)
text_label.pack()
'''


window.resizable(width=False, height=False)
window.mainloop()
