from utils import *
from matplotlib import pyplot as plt
import os
import cv2

max_val = 8
max_pt = -1
max_kp = 0

def read_img(file_name):
img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)
return img

def resize_img(image, dim):
res = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
return res

print("Welcome To COMPUTER VISION FRAMEWORK FOR INDIAN CURRENCY RECOGNITION")
orb = cv2.ORB_create()
videoCaptureObject = cv2.VideoCapture(0)
result = True

while (result):
    ret, frame = videoCaptureObject.read()
    cv2.imwrite("D:/Currency/NewPicture.jpg", frame)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()

test_img = read_img('D:/Currency/NewPicture.jpg')
scale_percent = 40
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)
original = resize_img(test_img, dim)
print(original)

(kp1, des1) = orb.detectAndCompute(test_img, None)
training_set = ['D:/Currency/5.jpg', 'D:/Currency/10.jpg',
'D:/Currency/10_coin.jpg', 'D:/Currency/20.jpg',
'D:/Currency/50.jpg', 'D:/Currency/100.jpg',
'D:/Currency/200.jpg', 'D:/Currency/500.jpg',
'D:/Currency/2000.jpg']

for i in range(0, len(training_set)):
    # train image
    train_img = cv2.imread(training_set[i])
    (kp2, des2) = orb.detectAndCompute(train_img, None)
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])
            
    if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2
        
    print(i, ' ', training_set[i], ' ', len(good))

    
if max_val != 8:
    # print(training_set[max_pt])
    print('good matches ', max_val)

    train_img = cv2.imread(training_set[max_pt])
    img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
    note = str(training_set[max_pt])[66:-4]
    print('\nDetected denomination: Rs. ', note)
    (plt.imshow(img3), plt.show())

else:
    print('No Matches')





