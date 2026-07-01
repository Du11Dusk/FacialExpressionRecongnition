import cv2

# 加载分类器
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# 读取图片
img = cv2.imread("images/happy.png")

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print("检测到人脸数量：", len(faces))

# 画框
for (x, y, w, h) in faces:

    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )

cv2.imshow("Face Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()