import keras.src.saving.legacy.hdf5_format
import keras.src.saving.legacy.hdf5_format
import numpy as np
from functions import doFinal

label_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer',
               'siren', 'street_music', 'human speech']
# Tentative classification of good and bad categories
classification = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0])

model = keras.src.saving.legacy.hdf5_format.load_model_from_hdf5('FinalModel.hdf5')
print('Model loaded')
doFinal(model)







# Construct model

from keras.layers import Conv2D, GlobalAveragePooling2D, AvgPool2D

# model = Sequential()
# model.add(Conv2D(filters=24, kernel_size=2, input_shape=(num_rows, num_columns, num_channels),
#                  kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=48, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=96, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=194, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(GlobalAveragePooling2D())
#
# model.add(Dense(num_labels, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = model.evaluate(x_test, y_test, verbose=1)
# accuracy = 100 * score[1]
#
# print("Pre-training accuracy: %.4f%%" % accuracy)
# from keras.callbacks import ModelCheckpoint
# from datetime import datetime
#
# num_epochs = 72
# num_batch_size = 256
#
# checkpointer = ModelCheckpoint(filepath='Best2DCNN1.hdf5',
#                                verbose=1, save_best_only=True)
# start = datetime.now()
#
# model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
#           callbacks=[checkpointer], verbose=1)
#
#
# duration = datetime.now() - start
# print("Training completed in time: ", duration)
