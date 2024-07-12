"""
 - Blockchain for Federated Learning -
      Federated Learning Script
"""

import tensorflow as tf
import numpy as np
import pickle

def reset():
    tf.keras.backend.clear_session()
import tensorflow as tf
import numpy as np
import pickle

class NNWorker:
    def __init__(self, X=None, Y=None, tX=None, tY=None, size=0, id="nn0", steps=10):
        ''' 
        Function to initialize Data and Network parameters
        '''
        self.id = id
        self.train_x = X
        self.train_y = Y
        self.test_x = tX
        self.test_y = tY
        self.size = size
        self.learning_rate = 0.1
        self.num_steps = steps
        self.n_hidden_1 = 256
        self.n_hidden_2 = 256
        self.num_input = 784
        self.num_classes = 10

    def build(self, base):
        ''' 
        Function to initialize/build network based on updated values received 
        from blockchain
        '''
        self.model = self._create_model()
        self._set_weights(base)

    def build_base(self):
        ''' 
        Function to initialize/build network with random initialization
        '''
        self.model = self._create_model()

    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_hidden_1, activation='relu', input_shape=(self.num_input,)),
            tf.keras.layers.Dense(self.n_hidden_2, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      
                      metrics=['accuracy'])
        return model

    def _set_weights(self, base):
        self.model.layers[0].set_weights([base['w1'], base['b1']])
        self.model.layers[1].set_weights([base['w2'], base['b2']])
        self.model.layers[2].set_weights([base['wo'], base['bo']])
    def close():
        print("closed")
        
    def train(self):
        
        ''' 
        Function to train the data, optimize and calculate loss and accuracy per batch
        '''
        if len(self.train_y.shape) > 1 and self.train_y.shape[1] > 1:  # Check if one-hot encoded
            self.train_y = np.argmax(self.train_y, axis=1)

        if len(self.test_y.shape) > 1 and self.test_y.shape[1] > 1:  # Check if one-hot encoded
            self.test_y = np.argmax(self.test_y, axis=1)

        print(self.train_x.shape)
        print(self.train_y.shape)
        self.model.fit(self.train_x, self.train_y, epochs=self.num_steps, verbose=1)
        print("Optimization Finished!")
        

    def centralized_accuracy(self):
        ''' 
        Function to train the data and calculate centralized accuracy based on 
        evaluating the updated model performance on test data 
        '''
        cntz_acc = dict()
        cntz_acc['epoch'] = []
        cntz_acc['accuracy'] = []

        self.build_base()
        for step in range(1, self.num_steps + 1):
            self.model.fit(self.train_x, self.train_y, epochs=1, verbose=1)
            acc = self.evaluate()
            cntz_acc['epoch'].append(step)
            cntz_acc['accuracy'].append(acc)
            print("epoch", step, "accuracy", acc)
        return cntz_acc

    def evaluate(self):
        '''
        Function to calculate accuracy on test data
        '''
        if len(self.test_y.shape) > 1 and self.test_y.shape[1] > 1:  # Check if one-hot encoded
            self.test_y = np.argmax(self.test_y, axis=1)
            print("00000000")
        _, accuracy = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        return accuracy

    def get_model(self):
        '''
        Function to get the model's trainable parameter values
        '''
        weights = {
            'w1': self.model.layers[0].get_weights()[0],
            'b1': self.model.layers[0].get_weights()[1],
            'w2': self.model.layers[1].get_weights()[0],
            'b2': self.model.layers[1].get_weights()[1],
            'wo': self.model.layers[2].get_weights()[0],
            'bo': self.model.layers[2].get_weights()[1]
        }
        weights["size"] = self.size
        return weights

# Example usage
# nn_worker = NNWorker(X=train_X, Y=train_Y, tX=test_X, tY=test_Y, size=size, id="nn0", steps=10)
# nn_worker.build_base()
# nn_worker.train()
# accuracy = nn_worker.evaluate()
# model_params = nn_worker.get_model()
