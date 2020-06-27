#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "inc/helper_cuda.h"
#include "inc/helper_functions.h"

#define learning_rate 0.1
#define epochs 1
#define actual 100


int randomNumberGeneration(int upperBound, int lowerBound) {
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

void printArray(double *array, int width) {
    for (int i = 0; i < width; i++) {
        printf("%3.3f ", array[i]);
    }
    printf("\n");
}

__global__ void matrix_multiply_shared(double *input_A, double *input_B, double *output_AB, int width){
    printf("MM on Device: %f \n", input_A[0]);
    printf("MM on Device: %f \n", input_A[1]);
    printf("MM on Device: %f \n", input_A[2]);
    printf("MM on Device: %f \n", input_A[3]);
    printf("MM on Device: %f \n", input_A[4]);
}


double neural_net_cuda(){
    // initialisation
    int num_input_nodes = 5; // number of input layer nodes
    int num_hidden_nodes = 5; // number of hidden layer nodes
    int num_output_nodes = 1; // number of output nodes
    int num_hidden_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    // generate input nodes
    double *h_input_nodes = createArray(num_input_nodes);
    printArray(h_input_nodes, num_input_nodes);

    // allocate memory for hidden_nodes and output nodes
    double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
    double *h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

    // generate initial weights for hidden and output layer
    double *h_hidden_weights= createWeights(num_hidden_weights);
    double *h_output_weights= createWeights(num_output_weights);

    // allocate memory in device for input, hidden and output nodes
    double *d_input_nodes, *d_hidden_nodes, *d_output_nodes;
    checkCudaErrors(cudaMalloc((void**)&d_input_nodes, sizeof(double) * num_input_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_hidden_nodes, sizeof(double) * num_hidden_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_output_nodes, sizeof(double) * num_output_nodes));

    // allocate memory in device for hidden_nodes and output nodes
    double *d_hidden_weights, *d_output_weights;
    checkCudaErrors(cudaMalloc((void**)&d_hidden_weights, sizeof(double) * num_hidden_weights));
    checkCudaErrors(cudaMalloc((void**)&d_output_weights, sizeof(double) * num_output_weights));

    // copy memory from host to device
    checkCudaErrors(cudaMemcpy(d_input_nodes, h_input_nodes, sizeof(double) * num_input_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hidden_nodes, h_hidden_nodes, sizeof(double) * num_hidden_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output_nodes, h_output_nodes, sizeof(double) * num_output_nodes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hidden_weights, h_hidden_weights, sizeof(double) * num_hidden_weights, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_output_weights, h_output_weights, sizeof(double) * num_output_weights, cudaMemcpyHostToDevice));
    
    for (int epoch=0; epoch<epochs; epoch++){

        dim3 dimGrid(1,1);
        dim3 dimBlock(1,1);
        matrix_multiply_shared<<<dimGrid, dimBlock>>>(d_input_nodes, d_hidden_weights, d_hidden_nodes, num_input_nodes);

        // double error_value = error(predicted);
        // if (error_value < 0.5){
        //     printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
        //     break;
        // }
    }

    // double *test = (double *)malloc(num_input_nodes * sizeof(double *));
    // checkCudaErrors(cudaMemcpy(test, d_input_nodes, sizeof(double) * num_input_nodes, cudaMemcpyDeviceToHost));
    // printArray(test, num_input_nodes);

    return 0;
}


int main(void){

    //Get the device properties
    int devID = findCudaDevice(0, 0);
    cudaGetDeviceProperties(0, 0);

    // Timing using Cuda Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    neural_net_cuda();

    cudaEventRecord(stop,0);
    cudaEventSynchronize( stop );
    float elapseTime;
    cudaEventElapsedTime(&elapseTime, start, stop);
    float throughput = (1*1*1*1/((elapseTime*1000)*(10^9)));

    printf( "GPU Shared Mem Throughput: %3.6f ms\n", throughput);
    printf( "GPU Shared Mem Time elpased: %3.6f ms\n", elapseTime );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    cudaDeviceReset();
}
