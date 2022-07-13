"""
建立人臉資料
判斷
"""
import cv2
import os
import glob
import numpy as np

total = 10
pictPath = r'C:\opencv\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pictPath)      # 建立辨識檔案物件
if not os.path.exists("Face_id"):                    # 如果不存在Face_id資料夾
    os.mkdir("Face_id")                              # 就建立Face_id
name = input("請輸入英文名字 : ")
if os.path.exists("Face_id\\" + name):
    print("此名字的人臉資料已經存在")
else:
    os.mkdir("Face_id\\" + name)
    cap = cv2.VideoCapture(0)                       # 開啟攝影機
    num = 1                                         # 影像編號
    while(cap.isOpened()):                          # 攝影機有開啟就執行迴圈   
        ret, img = cap.read()                       # 讀取影像
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
                minNeighbors = 3, minSize=(20,20))
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住人臉
        cv2.imshow("Photo", img)                    # 顯示影像在OpenCV視窗
        key = cv2.waitKey(500)
        if ret == True:                             # 讀取影像如果成功
            imageCrop = img[y:y+h,x:x+w]                      # 裁切
            imageResize = cv2.resize(imageCrop,(160,160))     # 重製大小
            faceName = "Face_id\\" + name + "\\" + name + str(num) + ".jpg"
            cv2.imwrite(faceName, imageResize)      # 儲存人臉影像           
            if num >= total:                        # 拍指定人臉數後才終止               
                if num == total:
                    print(f"拍攝第 {num} 次人臉成功")
                break
            print(f"拍攝第 {num} 次人臉成功")
            num += 1
    cap.release()                                   # 關閉攝影機
    cv2.destroyAllWindows()
    
nameList = []                                       # 姓名
faces_db = []                                       # 儲存所有人臉
labels = []                                         # 建立人臉標籤
index = 0                                           # 編號索引
dirs = os.listdir('Face_id')                         # 取得所有資料夾及檔案
for d in dirs:                                      # d是所有人臉的資料夾
    if os.path.isdir('Face_id\\' + d):               # 獲得資料夾
        faces = glob.glob('Face_id\\' + d + '\\*.jpg')  # 資料夾中所有人臉
        for face in faces:                          # 讀取人臉
            img = cv2.imread(face, cv2.IMREAD_GRAYSCALE)
            faces_db.append(img)                    # 人臉存入串列
            labels.append(index)                    # 建立數值標籤
        nameList.append(d)                          # 將英文名字加入串列
        index += 1
print(f"標籤名稱 = {nameList}")
print(f"標籤序號 ={labels}")
# 儲存人名串列，可在未來辨識人臉時使用
f = open('Face_id\\employee.txt', 'w')
f.write(','.join(nameList))
f.close()

print('建立人臉辨識資料庫')
model = cv2.face.LBPHFaceRecognizer_create()        # 建立LBPH人臉辨識物件
model.train(faces_db, np.array(labels))             # 訓練LBPH人臉辨識
model.save('Face_id\\deepmind.yml')                  # 儲存LBPH訓練數據
print('人臉辨識資料庫完成')
pictPath = r'C:\opencv\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pictPath)      # 建立辨識物件

model = cv2.face.LBPHFaceRecognizer_create()
model.read('Face_id\\deepmind.yml')                  # 讀取已訓練模型
f = open('Face_id\\employee.txt', 'r')               # 開啟姓名標籤
names = f.readline().split(',')                     # 將姓名存於串列

cap = cv2.VideoCapture(0)
while(cap.isOpened()):                              # 如果開啟攝影機成功
    ret, img = cap.read()                           # 讀取影像
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
                minNeighbors = 3, minSize=(20,20))
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 藍色框住人臉
    cv2.imshow("Face", img)                         # 顯示影像
    k = cv2.waitKey(500)                            # 0.2秒讀鍵盤一次
    if ret == True:       
        if k == ord("n") or k == ord("N"):          # 按 n 或 N 鍵
            imageCrop = img[y:y+h,x:x+w]                    # 裁切
            imageResize = cv2.resize(imageCrop,(160,160))   # 重製大小
            cv2.imwrite("Face_id\\face.jpg", imageResize)    # 將測試人臉存檔
            break
cap.release()                                       # 關閉攝影機
cv2.destroyAllWindows()
# 讀取員工人臉
gray = cv2.imread("Face_id\\face.jpg", cv2.IMREAD_GRAYSCALE)
val = model.predict(gray)
if val[1] < 100:                                     #人臉辨識成功
    print(f"您好 {names[val[0]]} ")
    print(f"匹配值是 {val[1]:6.2f}")
else:
    print("對不起無資料")