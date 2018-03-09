import cv2
import pandas as pd
import numpy as np
import train_net as nn
from matplotlib import pyplot as plt

print "Beginning Training Network"
#classifier = nn.train()
print "Finished Training Network"

classifier = nn.get_classifier("./data/")
cv2.namedWindow("preview")
cv2.moveWindow("preview", 0,0)
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

counter = 0
while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    if (counter == 100):
    	print "Getting Prediction"

        frame_y, frame_x, channels = frame.shape
        center_x, center_y = frame_x/2, frame_y/2
        resized_frame = frame[(center_y - frame_y/4):(center_y + frame_y/4), (center_x - frame_x/4) : (center_x + frame_x/4)]

        #resized_frame = frame[0:1080, 0:720]
    	rf = cv2.resize(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY), dsize=(28, 28))

    	resized_image = rf.reshape((1, 784)).flatten()

    	d = [-1]
    	for i in resized_image:
    		d.append(i)

    	df = pd.DataFrame.from_records([d])
    	features = df.loc[0,1:784]
    	features = features / 255



    	tt, te = nn.parse_labels_and_features(df)

        rand_example = np.random.choice(te.index)
        _, ax = plt.subplots()
        ax.matshow(te.loc[rand_example].values.reshape(28, 28))
        #ax.matshow(rf)
        ax.set_title("Label: 1")
        ax.grid(False)
        plt.show()

    	print nn.predict_class(tt, te, classifier)
    	counter = 0
    	print counter

    counter = counter + 1

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()

cv2.destroyWindow("preview")