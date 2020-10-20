import os
from matplotlib import pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3


class TransferredModel:
    def __init__(self, trained_weights_path, input_size):
        """:param
            trained_weights_path - path where pretrained model is saved
            input_size - the input shape of the images
        """
        self.trained_weights_loc = trained_weights_path
        self.input_size = input_size

    # create inception v3 model and initialize with pretrained weights
    def create_model(self):
        # create inception v3 model and load pretrained weights
        v3_model = InceptionV3(input_shape=self.input_size, include_top=False, weights=None)
        v3_model.load_weights(self.trained_weights_loc)

        # freeze all the trained models
        for layer in v3_model.layers:
            layer.trainable = False

        # just in case you want model summary
        # model.summary()

        # get output layer to flatten it for dense layer
        # it is possible to use different layers but here just lets try mixed10
        last_layer = v3_model.get_layer('mixed4')
        last_layer_output = last_layer.output
        flatten = layers.Flatten()(last_layer_output)

        # add some dense and dropout layers
        X = layers.Dense(1024, activation='relu')(flatten)
        X = layers.Dropout(0.5)(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        # create the model and return
        model = Model(v3_model.input, X)
        return model

    # compile model
    def compiled_model(self, opt):
        """:param
            opt - optimizer to be used during training
        """
        model_compiled = self.create_model()
        model_compiled.compile(opt, loss='binary_crossentropy', metrics=['acc'])
        return model_compiled


if __name__ == '__main__':
    # load the dataset
    # dataset used is from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
    dataset_loc = 'data/cats_and_dogs_filtered'
    train_dir = os.path.join(dataset_loc, 'train')
    valid_dir = os.path.join(dataset_loc, 'validation')

    # prepare data generator for train and validation
    train_data_gen = ImageDataGenerator(rescale=1.0/255.,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True
                                        )
    valid_data_gen = ImageDataGenerator(rescale=1.0/255.)

    # flow the data from the directory using ImageDataGenerator
    train_gen = train_data_gen.flow_from_directory(train_dir,
                                                   batch_size=64,
                                                   class_mode='binary',
                                                   target_size=(150,150)
                                                   )
    valid_gen = train_data_gen.flow_from_directory(valid_dir,
                                                   batch_size=64,
                                                   class_mode='binary',
                                                   target_size=(150,150)
                                                   )

    # input size for 150x150 rgb image
    input_size = (150, 150, 3)
    optimizer = SGD(0.00001)

    # download inception_v3 pretrained model from the link below and put in pretrained
    # https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    trained_model_location = 'data/pretrained/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model_transfer = TransferredModel(trained_model_location, input_size)
    compiled_model = model_transfer.compiled_model(optimizer)

    # finally train the pretrain model with the new data
    history = compiled_model.fit(train_gen, validation_data=valid_gen, epochs=10)

    # show accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # now plot them here
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()


