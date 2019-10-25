import tensorflow as tf
import dlib

# optional
# tf.device('GPU:0')

#  download dataset from cloud
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(y_train.shape,y_test.shape)
def create_model(lr, dropout):
  """
    Create model respect to dropout and learnin rate, use learning to train
  """
  _lr = lr/10000
  _dropout = dropout/100
  tf.set_random_seed(1)
  print(f"Current model learning_rate:{_lr}, dropout {_dropout}" )
  model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28,1)),
    tf.keras.layers.Conv2D(4,(5,5), activation='relu', use_bias=False,kernel_initializer='random_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(8,(5,5), activation='relu', use_bias=False,kernel_initializer='random_normal'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(_dropout),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='random_uniform')
  ])
  adam = tf.keras.optimizers.Adam(learning_rate=_lr)
  model.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def evaluate(learning_rate ,dropout):
  model = create_model(learning_rate, dropout)
  model.fit(x_train, y_train, epochs=4,verbose=0)
  result = model.evaluate(x_test,  y_test, verbose=2)
  print("Model test result: ", result)
  # minimize the test loss
  return result[0]


# Here the algortihm will minimize "evaluate" methods (which returns test loss). And algortihm will search learning rate and drop out in given ranges.
# (1-10000) / 10000 -> learning rate
# 0.0 - 0.5 -> dropuout rate
x,y = dlib.find_min_global(evaluate, 
                           [1,0],  
                           [10000,50],   
                           15)    
    
print("Params, learning_rate: ", x[0]/1000, "dropout: " , x[1]/100,  " Test results: loss: ", y)