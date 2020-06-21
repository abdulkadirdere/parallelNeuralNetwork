#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>



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
            ab[i*col+j]=0.0;
			for(int k=0; k<row; k++){
                ab[i*col+j] += a[i*row+k] * b[k*col+j];
            }
            ab[i*col+j] = sigmoid(ab[i*col+j]);
        }
    }
    return ab;
}

double error(double *prediction, double actual){
    return (0.5 * pow((prediction[0] - actual),2));
}

double neural_net_seq(){
    // initialisation
    int num_input_nodes = 2; // number of input layer nodes
    int num_hidden_nodes = 3; // number of hidden layer nodes
    int num_output_nodes = 1;
    int num_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    // generate input nodes
    double *h_input_nodes = createArray(num_input_nodes);

    // allocate memory for hidden_nodes and output nodes
    double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
    double *h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

    // generate initial weights for hidden and output layer
    double *h_hidden_weights= createWeights(num_weights);
    double *h_output_weights= createWeights(num_output_weights);

    for (int epoch=0; epoch<10; epoch++){
        // matrix multiplication hidden layer
        h_hidden_nodes = matrix_multiply_seq(h_input_nodes, h_hidden_weights, h_hidden_nodes, num_input_nodes, num_hidden_nodes);

        h_output_nodes = matrix_multiply_seq(h_hidden_nodes, h_output_weights, h_output_nodes, num_hidden_nodes, num_output_nodes);
        // printArray(h_output_nodes, num_output_nodes);

        // calculate the error
        double error_value = error(h_output_nodes, 1);
        printf("Epoch:%d - Error:%3.4f \n", epoch, error_value);

    }
    return 0;
}

int main() {
    neural_net_seq();
    return 0;
}