import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
 'model': 'cfg/yolo.cfg',
 'load': 'bin/yolov2.weights',
 'threshold': 0.3
}
tfnet = TFNet(options)

# read the color image and covert to RGB
img = cv2.imread('darkflow/sample_img/sample_dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# use YOLO to predict the image
result = tfnet.return_predict(img)
print(result)

# pull out some info from the results
for i in range(0, len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']
# add the box and label and display it
    img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    plt.imshow(img)
plt.show()


