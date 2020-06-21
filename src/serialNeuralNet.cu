#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>



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

// a sequential version of matrix multiplication
double *matrix_multiply_seq(double *a, double *b, double *ab, int width){
	for(int i=0; i<width; i++)
		for(int j=0; j<width; j++){
			ab[i*width+j]=0.0;
			for(int k=0; k<width; k++){
				ab[i*width+j] += a[i*width+k] * b[k*width+j];
			}
        }
    return ab;
}


int main() {
    // initialisation
    int num_input_nodes = 3; // number of input layer nodes
    int num_hidden_nodes = 3; // number of hidden layer nodes
    int num_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    
    double *h_input_nodes = createArray(num_input_nodes);
    printArray(h_input_nodes, num_input_nodes);

    // generate initial weights
    double *h_weights= createWeights(num_weights);
    printArray(h_weights, num_weights);

    // h_hidden_nodes
    double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));

    // matrix multiplication 
    h_hidden_nodes = matrix_multiply_seq(h_input_nodes, h_weights, h_hidden_nodes, num_input_nodes);
    printArray(h_hidden_nodes, num_hidden_nodes);

    return 0;
}