#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "inc/helper_functions.h" // includes cuda.h and cuda_runtime_api.h
#include "inc/helper_cuda.h" // helper functions for CUDA error check

#define learning_rate 0.1
#define epochs 10
#define actual 100
#define BLOCK_SIZE 32

__constant__ double c_learning_rate;

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

double neural_net_seq(int num_input_nodes, int num_hidden_nodes, int num_output_nodes, int num_hidden_weights, int num_output_weights){
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
        // printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
        if (error_value < 1){
            printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
            break;
        }
    }
    //-------------- Free Memory --------------//
    free(h_input_nodes);
    free(h_hidden_weights);
    free(h_hidden_nodes);
    free(h_output_weights);
    free(h_output_nodes);

    return 0;
}


__global__ void matrix_multiply_shared(double *input_A, double *input_B, double *output_AB, int height, int width){

    int tdi = blockIdx.x * blockDim.x + threadIdx.x;

    if (tdi < width){
        double sum=0;
        for(int k=0; k<height; k++){
            sum = sum + (input_A[k] * input_B[k*width+tdi]);
            // printf("AB: %f  %f   %f    %d   %d \n", sum, input_A[k], input_B[k*width+tdi], k, k*width+tdi);
        }
        output_AB[tdi] = 1/(1+exp(-1*sum));
        // output_AB[tdi] = sum;
    }
    
}

__global__ void backprop_output_kernel(double *output_weights, double *hidden_nodes, double predicted_value, double actual_value, int num_hidden_nodes){
    int tdi = blockIdx.x * blockDim.x + threadIdx.x;

    double delta;
    if (tdi == 0){
        delta = predicted_value - actual_value;
    }
    
    if (tdi < num_hidden_nodes){
        output_weights[tdi] = output_weights[tdi] - c_learning_rate * (hidden_nodes[tdi]*delta);
        // printf("output: %f  %f   %f  %f \n", output_weights[tdi], c_learning_rate, (hidden_nodes[tdi]*delta), delta);
    }
}


__global__ void backprop_hidden_kernel(double *layer1_weight, double *layer2_weight, double *input_nodes, double predicted_value, double actual_value, int input_size, int hidden_size){
    int tdi = blockIdx.x * blockDim.x + threadIdx.x;

    double delta;
    if (tdi == 0){
        delta = predicted_value - actual_value;
    }

    if (tdi < hidden_size){
        for(int j=0; j<input_size; j++){
            layer1_weight[tdi*input_size+j] = layer1_weight[tdi*input_size+j] - learning_rate * (input_nodes[j] * delta * layer2_weight[tdi]);
        }
    }
}


double neural_net_cuda(int num_input_nodes, int num_hidden_nodes, int num_output_nodes, int num_hidden_weights, int num_output_weights){
    // initialisation
    double learn_rate = learning_rate;
    cudaMemcpyToSymbol(c_learning_rate, &learn_rate, sizeof(double));

    // generate input nodes
    double *h_input_nodes = createArray(num_input_nodes);
    // printArray(h_input_nodes, num_input_nodes);

    // allocate memory for hidden_nodes and output nodes
    double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
    double *h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

    // generate initial weights for hidden and output layer
    double *h_hidden_weights= createWeights(num_hidden_weights);
    double *h_output_weights= createWeights(num_output_weights);
    // printArray(h_output_weights, num_output_weights);

    // allocate memory in device for input, hidden and output nodes
    double *d_input_nodes=0, *d_hidden_nodes=0, *d_output_nodes=0;
    checkCudaErrors(cudaMalloc((void**)&d_input_nodes, sizeof(double) * num_input_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_hidden_nodes, sizeof(double) * num_hidden_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_output_nodes, sizeof(double) * num_output_nodes));

    // allocate memory in device for hidden_nodes and output nodes
    double *d_hidden_weights=0, *d_output_weights=0;
    checkCudaErrors(cudaMalloc((void**)&d_hidden_weights, sizeof(double) * num_hidden_weights));
    checkCudaErrors(cudaMalloc((void**)&d_output_weights, sizeof(double) * num_output_weights));

    // copy memory from host to device
    checkCudaErrors(cudaMemcpy(d_input_nodes, h_input_nodes, sizeof(double) * num_input_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hidden_nodes, h_hidden_nodes, sizeof(double) * num_hidden_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output_nodes, h_output_nodes, sizeof(double) * num_output_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hidden_weights, h_hidden_weights, sizeof(double) * num_hidden_weights, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output_weights, h_output_weights, sizeof(double) * num_output_weights, cudaMemcpyHostToDevice));
    
    double threads = 1024;
    int block;

    for (int epoch=0; epoch<epochs; epoch++){
        // forward pass from input to hidden layer
        block = (num_hidden_nodes / 1024) + 1;
        dim3 dimBlock(threads,1);
        dim3 dimGrid(block,1);
        matrix_multiply_shared<<<dimGrid, dimBlock>>>(d_input_nodes, d_hidden_weights, d_hidden_nodes, num_input_nodes, num_hidden_nodes);

        // forward pass from hidden to output layer
        block = (num_output_nodes / 1024)+1;
        dim3 dimBlock_output(threads,1);
        dim3 dimGrid_output(block, 1);
        matrix_multiply_shared<<<dimGrid_output,dimBlock_output>>>(d_hidden_nodes, d_output_weights, d_output_nodes, num_hidden_nodes, num_output_nodes);

        checkCudaErrors(cudaMemcpy(h_output_nodes, d_output_nodes, sizeof(double) * num_output_nodes, cudaMemcpyDeviceToHost));
        double predicted = h_output_weights[0];

        // backpropagation from output to hidden layer
        block = (num_hidden_nodes / 1024)+1;
        dim3 dimBlock_backprop(threads,1);
        dim3 dimGrid_backprop(block, 1);

        backprop_output_kernel<<<dimGrid_backprop, dimBlock_backprop>>>(d_output_weights, d_hidden_nodes, predicted, actual, num_hidden_nodes);
        checkCudaErrors(cudaMemcpy(h_output_weights, d_output_weights, sizeof(double) * num_output_weights, cudaMemcpyDeviceToHost));

        // backpropagation from hidden to input layer
        block = (num_hidden_nodes / 1024)+1;
        dim3 dimBlock_backprop_input(threads,1);
        dim3 dimGrid_backprop_input(block, 1);

        backprop_hidden_kernel<<<dimGrid_backprop_input, dimBlock_backprop_input>>>(d_hidden_weights, d_output_weights, d_input_nodes, predicted, actual, num_input_nodes, num_hidden_nodes);
        checkCudaErrors(cudaMemcpy(h_hidden_weights, d_hidden_weights, sizeof(double) * num_hidden_weights, cudaMemcpyDeviceToHost));
        // printArray(h_hidden_weights, num_hidden_weights);


        // calculate the error
        double error_value = error(predicted); 
        // printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
        if (error_value < 1){
            printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
            break;
        }

    }
    //-------------- CUDA Free Memory --------------//
    checkCudaErrors(cudaFree(d_input_nodes));
    checkCudaErrors(cudaFree(d_hidden_weights));
    checkCudaErrors(cudaFree(d_hidden_nodes));
    checkCudaErrors(cudaFree(d_output_weights));
    checkCudaErrors(cudaFree(d_output_nodes));

    free(h_input_nodes);
    free(h_hidden_weights);
    free(h_hidden_nodes);
    free(h_output_weights);
    free(h_output_nodes);
    return 0;
}


int main() {
    // initialisation
    int num_input_nodes = 512; // number of input layer nodes
    int num_hidden_nodes = 1024; // number of hidden layer nodes
    int num_output_nodes = 1; // number of output nodes
    int num_hidden_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    //-------------- Serial Neural Network --------------//
    // CUDA timing of event
    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    cudaEventRecord(serial_start);
    neural_net_seq(num_input_nodes, num_hidden_nodes, num_output_nodes, num_hidden_weights, num_output_weights);
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);

    //-------------- CUDA Neural Network --------------//
    // CUDA timing of event
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    cudaEventRecord(cuda_start);
    neural_net_cuda(num_input_nodes, num_hidden_nodes, num_output_nodes, num_hidden_weights, num_output_weights);
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);

    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);

    //-------------- CUDA Performance Metrics --------------//

    // std::cout << "Input Nodes: " << num_input_nodes << " Hidden Nodes:" << num_hidden_nodes << " Output Nodes:" << num_output_nodes << std::endl;

    printf("Serial Neural Network Time: %3.6f ms \n", serial_time);
    printf("Cuda Neural Network Time: %3.6f ms \n", cuda_time);
    return 0;
}