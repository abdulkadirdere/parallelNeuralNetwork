#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>

#define learning_rate 0.1
#define epochs 10
#define actual 100

int randomNumberGeneration(int upperBound, int lowerBound) {
    // creates a random integer within the bounds
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

double *createData(double *array, int num_element) {
    for (int i = 0; i < num_element; i++) {
        array[i] = randomNumberGeneration(9, 0);
    }
    return array;
}

double *createArray(double num_element) {
    double *array = (double *)malloc(num_element * sizeof(double *));
    // create synthetic data for matrix
    array = createData(array, num_element);
    return array;
}

void printArray(double *array, int width) {
    for (int i = 0; i < width; i++) {
        printf("%3.3f ", array[i]);
    }
    printf("\n");
}

double *createWeights(int num_element) {
    // allocate memory
    double *array = (double *)malloc(num_element * sizeof(double *));

    // generate initial weights
    for (int i = 0; i < num_element; i++) {
        double weight = rand() / ((double) RAND_MAX);
        array[i] = weight;
    }
    return array;
}

double sigmoid(double  node){
    return 1/(1+exp(-1*node));
}

// a sequential version of matrix multiplication
double *matrix_multiply_seq(double *a, double *b, double *ab, int row, int col){
	for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            ab[j]=0.0;
			for(int k=0; k<row; k++){
                ab[j] += a[k] * b[k*col+j];
            }
            ab[j] = sigmoid(ab[j]);
        }
    }
    return ab;
}

double error(double prediction){
    return (0.5 * pow((prediction - actual),2));
}

double *backprop_output(double *output_weights, double *hidden_nodes, double predicted_value, double actual_value, int num_hidden_nodes){
    double delta = predicted_value - actual_value;
    for (int i=0; i< num_hidden_nodes; i++){
        output_weights[i] = output_weights[i] - learning_rate * (hidden_nodes[i]*delta);
    }
    return output_weights;
}

double *backprop_hidden(double *input_weights, double *hidden_weights, double *input_nodes, double predicted_value, double actual_value, int row, int col){
    double delta = predicted_value - actual_value;
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            input_weights[i*col+j] = input_weights[i*col+j] - learning_rate * (input_nodes[i] * delta * hidden_weights[j]);
        }
    }
    return input_weights;
}

double neural_net_seq(){
    // initialisation
    int num_input_nodes = 512; // number of input layer nodes
    int num_hidden_nodes = 1024; // number of hidden layer nodes
    int num_output_nodes = 1; // number of output nodes
    int num_hidden_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    // generate input nodes
    double *h_input_nodes = createArray(num_input_nodes);

    // allocate memory for hidden_nodes and output nodes
    double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
    double *h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

    // generate initial weights for hidden and output layer
    double *h_hidden_weights= createWeights(num_hidden_weights);
    double *h_output_weights= createWeights(num_output_weights);

    for (int epoch=0; epoch<epochs; epoch++){
        // matrix multiplication hidden layer
        h_hidden_nodes = matrix_multiply_seq(h_input_nodes, h_hidden_weights, h_hidden_nodes, num_input_nodes, num_hidden_nodes);

        h_output_nodes = matrix_multiply_seq(h_hidden_nodes, h_output_weights, h_output_nodes, num_hidden_nodes, num_output_nodes);
        double predicted = h_output_weights[0];
        
        // weights must be updated
        h_output_weights = backprop_output(h_output_weights, h_hidden_nodes, predicted, actual, num_hidden_nodes);

        h_hidden_weights = backprop_hidden(h_hidden_weights, h_output_weights, h_input_nodes, predicted, actual, num_input_nodes, num_hidden_nodes);
        // printArray(h_hidden_weights, num_hidden_weights);

        // calculate the error
        double error_value = error(predicted);
        printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
        if (error_value < 0.5){
            // printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
            break;
        }
    }
    return 0;
}

int main() {
    //-------------- Serial Neural Network --------------//
    // CUDA timing of event
    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    cudaEventRecord(serial_start);
    neural_net_seq();
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);

    printf("Serial Neural Network Time: %3.6f ms \n", serial_time);
    return 0;
}