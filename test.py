import numpy as np 
#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 20000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, middleLayerNeurons, outputLayerNeurons = 2,2,4,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
middle_weights = np.random.uniform(size=(hiddenLayerNeurons,middleLayerNeurons))
middle_bias =np.random.uniform(size=(1,middleLayerNeurons))
output_weights = np.random.uniform(size=(middleLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

print("Initial hidden weights: ",end='')
print(*hidden_weights)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)


#Training algorithm
for _ in range(epochs):
	#Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
        
    middle_layer_activation = np.dot(hidden_layer_output,middle_weights)
    middle_layer_activation += middle_bias
    middle_layer_output = sigmoid(middle_layer_activation)

    output_layer_activation = np.dot(middle_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    #Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_middle_layer = d_predicted_output.dot(output_weights.T)
    d_middle_layer = error_middle_layer * sigmoid_derivative(middle_layer_output)

    error_hidden_layer = d_middle_layer.dot(middle_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #Updating Weights and Biases
    output_weights += middle_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr

    middle_weights += hidden_layer_output.T.dot(d_middle_layer) * lr
    middle_bias += np.sum(d_middle_layer, axis=0, keepdims=True)

    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

print("Final hidden weights: ",end='')
print(*hidden_weights)
print("Final hidden bias: ",end='')
print(*hidden_bias)
print("Final output weights: ",end='')
print(*output_weights)
print("Final output bias: ",end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)