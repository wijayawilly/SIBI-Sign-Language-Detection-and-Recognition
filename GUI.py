from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename
import keyboard
import os
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#region Functions
def video_cap():

    #region Variables
    global predicted_class
    global auto
    global prev_frame_time
    global del_all
    #endregion

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame,1)

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
  
    # puting the FPS count on the frame
    cv2.putText(frame, fps, (580, 35), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    #Creating ROI
    roi = frame[top_ROI:btm_ROI, right_ROI:left_ROI]

    
    if segment_on:
        skin_segmented_ycbcr = skin_color_segmentation_ycbcr(roi,frame,is_off) #Skin color Segmentation
        prediction_box = np.reshape(skin_segmented_ycbcr,(1,skin_segmented_ycbcr.shape[0],skin_segmented_ycbcr.shape[1],3))
        dialog_var.set("[INFO] Background Removal is Currently ON !")
        switch_var.set("YCbCr Segmentation ON")
    else:
        frame[150:350,10:210] = roi
        roi = cv2.resize(roi, (100, 100))
        prediction_box = np.reshape(roi,(1,roi.shape[0],roi.shape[1],3))
        dialog_var.set("[INFO] Background Removal is Currently OFF !")
        switch_var.set("YCbCr Segmentation OFF")
    
    if model:
        # Prediction
        prediction_box = prediction_box / 255.0
        pred = model.predict(prediction_box)
        pred_class = np.argmax(pred)

        if pred_class == 0:
            pred_class = "A"
            if predicted_class != "A":
                predicted_class="A"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 1:
            pred_class = "B"
            if predicted_class != "B":
                predicted_class="B"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 2:
            pred_class = "C"
            if predicted_class != "C":
                predicted_class="C"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 3:
            pred_class = "D"
            if predicted_class != "D":
                predicted_class="D"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 4:
            pred_class = "Del"
            if predicted_class != "Del":
                predicted_class="Del"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 5:
            pred_class = "E"
            if predicted_class != "E":
                predicted_class="E"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 6:
            pred_class = "F"
            if predicted_class != "F":
                predicted_class="F"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 7:
            pred_class = "G"
            if predicted_class != "G":
                predicted_class="G"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 8:
            pred_class = "H"
            if predicted_class != "H":
                predicted_class="H"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 9:
            pred_class = "I"
            if predicted_class != "I":
                predicted_class="I"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 10:
            pred_class = "J"
            if predicted_class != "J":
                predicted_class="J"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 11:
            pred_class = "K"
            if predicted_class != "K":
                predicted_class="K"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 12:
            pred_class = "L"
            if predicted_class != "L":
                predicted_class="L"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 13:
            pred_class = "M"
            if predicted_class != "M":
                predicted_class="M"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 14:
            pred_class = "N"
            if predicted_class != "N":
                predicted_class="N"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 15:
            pred_class = "Nothing"
            if predicted_class != "Nothing":
                predicted_class="Nothing"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 16:
            pred_class = "O"
            if predicted_class != "O":
                predicted_class="O"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 17:
            pred_class = "P"
            if predicted_class != "P":
                predicted_class="P"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 18:
            pred_class = "Q"
            if predicted_class != "Q":
                predicted_class="Q"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 19:
            pred_class = "R"
            if predicted_class != "R":
                predicted_class="R"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 20:
            pred_class = "S"
            if predicted_class != "S":
                predicted_class="S"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 21:
            pred_class = "Space"
            if predicted_class != "Space":
                predicted_class="Space"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 22:
            pred_class = "T"
            if predicted_class != "T":
                predicted_class="T"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 23:
            pred_class = "U"
            if predicted_class != "U":
                predicted_class="U"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 24:
            pred_class = "V"
            if predicted_class != "V":
                predicted_class="V"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 25:
            pred_class = "W"
            if predicted_class != "W":
                predicted_class="W"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 26:
            pred_class = "X"
            if predicted_class != "X":
                predicted_class="X"
                auto = 0
            else:
                auto = auto + 1

        elif pred_class == 27:
            pred_class = "Y"
            if predicted_class != "Y":
                predicted_class="Y"
                auto = 0
            else:
                auto = auto + 1
        
        elif pred_class == 28:
            pred_class = "Z"
            if predicted_class != "Z":
                predicted_class="Z"
                auto = 0
            else:
                auto = auto + 1
        

        cv2.putText(frame, pred_class, (10, 60), cv2.FONT_ITALIC, 2, (51,255,51), 5)
        cv2.rectangle(frame, (left_ROI, top_ROI), (right_ROI,btm_ROI), (255,128,0), 3) #Visual Rectangle for ROI
        cv2.putText(frame, "Put Hands Here!" ,(10, 130), cv2.FONT_ITALIC, 0.8, (51,255,51), 2)
    else:
        dialog_var.set("[INFO] Please Load Weight First !")
    
    if auto >= 15:
        if predicted_class != "Nothing" and predicted_class != "Del" and predicted_class != "Space":
            prediction.append(predicted_class)
            if len(prediction) <=0:
                pred_text.insert(END, prediction)
            else:
                for i in prediction:
                    pred_text.delete("1.0", "end")
                    pred_text.insert(END, "".join(prediction))
            
        elif predicted_class == "Space":
            prediction.append(" ")
            pred_text.delete("1.0", "end")
            pred_text.insert(END, "".join(prediction))

        elif predicted_class == "Del":
            if len(prediction) == 0:
                pred_text.insert(END, prediction)
            else:
                prediction.pop(len(prediction) - 1)
                pred_text.delete("1.0", "end")
                pred_text.insert(END, "".join(prediction))
        auto = 0
    if keyboard.is_pressed('r'):
        prediction.clear()
        pred_text.delete("1.0", "end")



    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    l_cap.imgtk = imgtk
    l_cap.configure(image=imgtk)
    l_cap.after(1, video_cap)

def skin_color_segmentation_ycbcr(roi_frame, main_frame,is_on):

    luma_val = luma_slider.get()
    cr_val = cr_slider.get()
    cb_val = cb_slider.get()

    ycbcr = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2YCrCb)
    blured_ycbcr = cv2.GaussianBlur(ycbcr,(9,9), 0)

    skin_low = np.array([luma_val, cr_val, cb_val], dtype="uint8")
    skin_high = np.array([255, 173, 127], dtype="uint8")

    #Allowing only Skin color
    skin_mask = cv2.inRange(blured_ycbcr, skin_low, skin_high)

    #Finding Contour Based on Skin Color
    contour,_= cv2.findContours(skin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    

    #Creating black mask for black BG 
    mask = np.zeros(skin_mask.shape)

    #check if there are contour or not
    if len(contour) > 0 :
        print(len(contour))
        max_cnt_area = max(contour, key=cv2.contourArea)
        # cv2.drawContours(roi_frame, max_cnt_area, -1, (255,255,255), 1)
        # print("Contour Shape : ", len(cnt_area))
        # print("Contour MAX Shape : ", len(max_cnt_area))
        # skin = cv2.bitwise_and(roi, roi, mask=skin_mask)
        cv2.fillPoly(mask, [max_cnt_area], (255,255,255))

    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.erode(mask, None, iterations=3)
    mask_stack = np.dstack([mask]*3) #Creating 3 Color Channel

    #Converting Into float matrices
    mask_stack  = mask_stack.astype('float32') / 255.0 
    roi         = roi_frame.astype('float32') / 255.0

    #Blending
    masked = (mask_stack * roi) + ((1-mask_stack) * (0.0,0.0,0.0))
    masked = (masked * 255).astype('uint8')
    main_frame[150:350,10:210] =  masked

    #Rescaling
    masked_ycbcr = cv2.resize(masked, (100, 100))

    return masked_ycbcr

def models_load():
    global path
    global model
    dialog_var.set("Weight Loading...")
    path = askopenfilename(initialdir='\\', filetypes=[('Model file','*.h5')])
    model = load_model(path)
    model_path = path
    size = round(os.stat(path).st_size * (9.5367431640625* (10 ** -7)))
    model_name = os.path.basename(path)
    model_info_var.set("Model Name : " + model_name + "\n\n" + "Model Path : " + model_path + "\n\n" + "Model Size : " + str(size) + " MB")
    dialog_var.set("Weight Loaded !")
    flag = 0

def ycbcr_switch():
    global is_off
    global segment_on

    if is_off:
        segment_on = True
        is_off = False
    else:
        segment_on = False
        is_off=True

#endregion


#region Global Variables
prediction = []
bright_val = 0
is_off = True
segment_on = False
predicted_class=""
model = ""
auto=0
del_all=0
#endregion




root = Tk()
root.geometry('1150x1080')
root.title("Sign Language Detection and Recognition")

#region Variable
pred_show = StringVar()
dialog_var = StringVar()

switch_var = StringVar()
switch_var.set("YCbCr Segmentation OFF")

model_info_var = StringVar()
model_info_var.set("Model Name : None" + "\nModel Path : None" + "\nModel Size : None")



#Creating ROI frame for capturing hand
top_ROI=150
btm_ROI=350
right_ROI=10
left_ROI=210

#endregion

#region Frames
#Main Frame
main_frame = Frame(root)
main_frame.place(x=0,y=0)

#Top Frame
top_frame = Frame(root)
top_frame.place(x=80,y=0)

# Mid Frame
mid_frame = Frame(root)
mid_frame.place(x=400,y=520)

#Left Low Bottom Frame
left_low_bottom_frame = Frame(root)
left_low_bottom_frame.place(x=80,y=520)

#Bottom Frame
bottom_frame = Frame(root)
bottom_frame.place(x=400,y=620)

#Low bottom Frame
low_bottom_frame = Frame(root)
low_bottom_frame.place(x=580,y=700)



#endregion

#region Labels, Button, slider, keyPressed

#region Top Frames
#Creating Label for the Camera Capture
l_cap = Label(top_frame, bg='black', bd=5)
l_cap.pack(side=RIGHT, padx=(50,0))

#Label For slider
ycrcb_labelframe = LabelFrame(top_frame, text="YCbCr Value Adjustment",font =("Times", 15), bg="Gray", width=500)
ycrcb_labelframe.pack(side=LEFT, padx=(0,50))

l_luma = Label(ycrcb_labelframe, text="Luma Value", font=('Times',10))
l_luma.grid(row=0, column=0, pady=(0,450))

l_cr = Label(ycrcb_labelframe, text="Cr Value", font=('Times',10))
l_cr.grid(row=0, column=1, pady=(0,450), padx=(10,0))

l_cb = Label(ycrcb_labelframe, text="Cb Value", font=('Times',10))
l_cb.grid(row=0, column=2, pady=(0,450),  padx=(10,0))

#Y,Cr,Cb Value Slider
luma_slider = Scale(ycrcb_labelframe, from_=0, to=255, length=400, resolution=1, orient=VERTICAL)
luma_slider.grid(row=0, column=0, pady=(10,0))

cr_slider = Scale(ycrcb_labelframe, from_=0, to=255, length=400, resolution=1, orient=VERTICAL)
cr_slider.grid(row=0, column=1, pady=(10,0))

cb_slider = Scale(ycrcb_labelframe, from_=0, to=255, length=400, resolution=1, orient=VERTICAL)
cb_slider.grid(row=0, column=2, pady=(10,0))

#Label for Model Info
l_model_info_title = LabelFrame(left_low_bottom_frame, text="Model Info", font =("Times", 15))
l_model_info_title.pack()

l_model_info = Label(l_model_info_title, textvariable=model_info_var, font=("Times", 15), justify=LEFT, bg="grey", wraplength=300)
l_model_info.pack()

#endregion

#region bottom Frames
#Label for Prediction Title
l_pred_title = LabelFrame(mid_frame, text="Predicted Alfabet", font =("Times", 15), labelanchor='n')
l_pred_title.pack()

#Textbox for Prediction Words
pred_text = Text(l_pred_title,font=('Times', 16), height=5, width=58)
pred_text.pack()

# Button Loading CNN Models
model_load = Button(bottom_frame, width=20, height=3, text="Load Model", command=lambda: models_load(), bg="grey")
model_load.pack(side=LEFT, padx=(0,275))

# Button for segmentation switch
ycbcr_btn = Button(bottom_frame, width=30, height=3 ,textvariable=switch_var, command=lambda: ycbcr_switch(), bg="grey")
ycbcr_btn.pack(side=RIGHT)


#Notifications
l_notif = LabelFrame(low_bottom_frame, text="Notification Box", bg="blue", width=400, height=10)
l_notif.pack()

l_notif_box = Label(l_notif, font=("Courier", 15), textvariable=dialog_var, fg="red", bg="lightcyan", wraplength=300)
l_notif_box.pack()

#endregion

#endregion

    
cap = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

video_cap()
root.mainloop()





