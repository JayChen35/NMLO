from tensorflow import keras
from dataparser import get_testing_data
import numpy as np

model = keras.models.load_model('checkpoint')
testing_imgs, testing_ids, class_labels = get_testing_data()

print(testing_imgs.shape)
pred = model.predict(testing_imgs)
print(pred)
for i in range(10):
    print(pred[i])
print(pred.shape)

# output_np = np.zeros((len(pred), 2), dtype=object)
file = open("submission4.csv", "w+")
file.write("id,label\n")
for i in range(len(pred)):
    idx = np.argmax(pred[i])
    out = str(testing_ids[i]) + "," + class_labels[idx] + "\n"
    file.write(out)
    # output_np[i][0] = str(testing_ids[0])
    # output_np[i][1] = class_labels[idx]

file.close()
# print(output_np)
# np.savetxt("submission.csv", output_np, delimiter=",")
