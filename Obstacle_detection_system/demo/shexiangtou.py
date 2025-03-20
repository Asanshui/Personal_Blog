import cv2 as cv

# 读取视频信息
cap = cv.VideoCapture("http://admin:admin@192.168.1.6:8081/")  # @前为账号密码，@后为ip地址
face_xml = cv.CascadeClassifier("haarcascade_frontalface_default.xml")  # 导入XML文件

while True:
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法连接到摄像头，尝试重新连接...")
        cap = cv.VideoCapture("http://admin:admin@192.168.1.6:8081/")  # 尝试重新连接
        continue

    # 按下 'q' 键退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv.destroyAllWindows()