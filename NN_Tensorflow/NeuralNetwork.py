import tensorflow as tf

if __name__ == "__main__":
    # Load the mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape the array to the dimensions so that the keras API works
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # normalise the photos
    x_train = tf.keras.utils.normalize( x_train, axis=1 )
    x_test  = tf.keras.utils.normalize( x_test , axis=1 )

    # Creating a Sequential Model and adding the layers
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Flatten() )
    model.add( tf.keras.layers.Dense( 196, activation=tf.nn.relu ) )
    model.add( tf.keras.layers.Dense( 49, activation=tf.nn.relu ) )
    model.add( tf.keras.layers.Dense( 10, activation=tf.nn.softmax ) )

    # Compile the model for the soon to do evaluation
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    # Fit the model with the training data
    model.fit(x=x_train,y=y_train, epochs=5)

    # Print the data of the model
    model.summary()  

    result = model.evaluate(x_test, y_test)
    print( "The total loss: {}".format(result[0]) )
    print( "The total accuracy: {}".format(result[1]))