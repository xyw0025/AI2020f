import os
from matplotlib import pyplot as plt
import logging
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data path = "./data"


pic_size = 128
batch_size = 128
epochs=90

# for file names
timestr = time.strftime("%Y%m%d_%H%M")

if not os.path.exists(timestr):
    os.makedirs(timestr)


from tensorflow.keras import Sequential, callbacks, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, InputLayer
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# #初始化model
model = Sequential()

model.add(InputLayer(input_shape=(128, 128, 3)))
model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2))) #64,64,64

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2))) #32,32,128
#
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(11, activation='softmax'))

model.summary()


from tensorflow.keras import optimizers, metrics

opt = optimizers.Adam(learning_rate=0.0003)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
)


validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


training_total = 9866 
validate_total = 3430


train_generator = train_datagen.flow_from_directory(
        './data/training',
        target_size=(pic_size, pic_size),
        batch_size=batch_size,
        class_mode='categorical', shuffle=True)

validate_generator = validate_datagen.flow_from_directory(
        './data/validation',
        target_size=(pic_size, pic_size),
        batch_size=batch_size,
        class_mode='categorical')
class_list = list(range(11))
for k,v in train_generator.class_indices.items():
    class_list[int(v)] = k
with open('class_list.txt', 'w') as f:
    f.write(','.join(class_list))
    
mc1 = callbacks.ModelCheckpoint(timestr + '/first_weights_{epoch:08d}.h5', save_weights_only=False, period=5)
learning_rate_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=5, factor=0.8, min_lr=0.00001)
mc2 = callbacks.ModelCheckpoint(timestr + '/sec_weights_{epoch:08d}.h5', save_weights_only=False, period=5)




history = model.fit(
      train_generator,
      steps_per_epoch=round(training_total/batch_size),
      epochs=round(epochs*0.5),
      validation_data=validate_generator,
      validation_steps=round(validate_total/batch_size),
      callbacks=[mc1, learning_rate_decay]
)
train_generator = train_datagen.flow_from_directory(
        './origin(1)/training',
        target_size=(pic_size, pic_size),
        batch_size=batch_size,
        class_mode='categorical', shuffle=True)

validate_generator = validate_datagen.flow_from_directory(
        './origin(1)/validation',
        target_size=(pic_size, pic_size),
        batch_size=batch_size,
        class_mode='categorical')

opt = optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# learing rate reset as 0.0003


sec_history = model.fit(
      train_generator,
      steps_per_epoch=round(training_total/batch_size),
      epochs=round(epochs*0.5),
      validation_data=validate_generator,
      validation_steps=round(validate_total/batch_size),
      callbacks=[mc2, learning_rate_decay]
)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#### !!!! val_acc -> val_accuracy to avoid error
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.savefig('./plots/accuracy_'+ timestr + '.png')
plt.show()

plt.plot(sec_history.history['accuracy'])
plt.plot(sec_history.history['val_accuracy'])
#### !!!! val_acc -> val_accuracy to avoid error
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.savefig('./plots/accuracy_'+ timestr + '.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.savefig('./plots/loss_'+ timestr + '.png')
plt.show()




model.save('./weights/weights_'+timestr+'.h5')



import numpy as np
from tensorflow.keras.preprocessing import image


predictions = {}

for i in os.listdir('origin/testing'):
    img = image.load_img(os.path.join('origin/testing', i), target_size=(pic_size, pic_size))
    input_arr = image.img_to_array(img) / 255
    input_arr = np.array([input_arr]) 
    predictions[i.split('.')[0]] = model.predict_classes(input_arr)[0]
    
with open('./predicts/predict_' + timestr + '.csv', 'w') as f:
    f.write('Id,Category\n')

    for i in sorted(predictions.keys(), key=lambda x: int(x)):
        f.write('%04d,%s\n' % (int(i), predictions[i]))
      
