"""Explore learning curves for classification of handwritten digits"""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensoflow_hub as tf

digits=load_digits()

def display_digits():
    """Read in the 8x8 pictures of numbers and display 10 of them"""
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model(tensorflow_model=False):
    if tensorflow_model==False:
        """Train a model on pictures of digits.

        Read in 8x8 pictures of numbers and evaluate the accuracy of the model
        when different percentages of the data are used as training data. This function
        plots the average accuracy of the model as a function of the percent of data
        used to train it.
        """
        data = load_digits()
        num_trials = 10
        train_percentages = range(5, 95, 5)
        test_accuracies = numpy.zeros(len(train_percentages))

        # train models with training percentages between 5 and 90 (see
        # train_percentages) and evaluate the resultant accuracy for each.
        # You should repeat each training percentage num_trials times to smooth out
        # variability.
        # For consistency with the previous example use
        # model = LogisticRegression(C=10**-10) for your learner

        # TODO: your code here

        fig = plt.figure()
        plt.plot(train_percentages, test_accuracies)
        plt.xlabel('Percentage of Data Used for Training')
        plt.ylabel('Accuracy on Test Set')
        plt.show()
    if tensorflow_model==True:
        
        class model ():
            def __init__ (self):
                base_model=ResNet50V2(include_top=False)
                base_model.trainable=False

                data_augmentetion=tf.keras.models.Sequential([
                    tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
                    tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
                    tf.keras.layers.experimental.preprocessing.RandomWidth(0.2)
                ])
                data=tf.keras.preprocessing.image_dataset_from_directory(digits,
                                                                            batch_size=32,
                                                                            image_size=(224, 224),
                                                                            label_mode='categorical')


                inputs=tf.keras.layers.Input(shape=(224, 224, 3))

                augmented_layer=data_augmentetion(inputs)

                x=base_model(augmented_layer, training=False)

                pool_layer_1=tf.keras.layers.GlobalMaxPooling2D()(x)
                pool_layer_2=tf.keras.layers.GlobalAveragePooling2D()(tf.expand_dims(tf.expand_dims(pool_layer_1, axis=0), axis=0))

                outputs=tf.keras.layers.Dense(13, activation='softmax')(pool_layer_2)

                model=tf.keras.Model(inputs, outputs)

                model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics='accuracy')
                
                #This function helps in tracking the models performance and also the loss curves

                history=model.fit(data, epochs=1, steps_per_epoch=len(data), callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200)),
                                                                                                                tf.keras.callbacks.ModelCheckpoint('checkpoin2.ckpt',
                                                                                                                    save_weights_only=True,
                                                                                                                    save_freq='epoch',
                                                                                                                    monitor='loss')]

                                                                                                                    verbose=0)

                #The performance of the model resides over here
                pd.DataFrame(history.history).plot()
                plt.figure(figsize=(10,10))
                lrs=1e-4*10**(np.arange(0, 1)/ 200)
                plt.semilogx(lrs, history.history['loss'])
        model()        
                
                
                
                
                
                
                
                
                
                
                
if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    display_digits()
    # train_model()
