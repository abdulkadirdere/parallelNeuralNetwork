#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define learning_rate 0.1
#define epochs 1
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

int main(int argc, char **argv){

    int process_rank, comm_size;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &process_rank);

    int num_input_nodes = 4; // number of input layer nodes
    int num_hidden_nodes = 4; // number of hidden layer nodes
    int num_output_nodes = 1; // number of output nodes

    int num_hidden_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    double *h_input_nodes;
    // allocate memory for hidden_nodes and output nodes
    double *h_hidden_nodes;
    double *h_output_nodes;
    // // generate initial weights for hidden and output layer
    double *h_hidden_weights;
    double *h_output_weights;

    if (process_rank == 0){
        // // generate input nodes
        h_input_nodes = createArray(num_input_nodes);

        // allocate memory for hidden_nodes and output nodes
        h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
        h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

        // // generate initial weights for hidden and output layer
        h_hidden_weights= createWeights(num_hidden_weights);
        h_output_weights= createWeights(num_output_weights);

        printf("\n");
        printArray(h_hidden_weights, num_hidden_weights);
        printArray(h_input_nodes, num_input_nodes);

    }
    MPI_Barrier(comm);
    int rows_per_process = (num_hidden_nodes/comm_size) + 1;

    double *local_row = malloc((rows_per_process*num_hidden_nodes) * sizeof(double));
    // double *local_input_node = malloc(num_input_nodes * sizeof(double));

    MPI_Scatter(h_hidden_weights, 2, MPI_DOUBLE, local_row, 2, MPI_DOUBLE, 0, comm);
    // MPI_Allgather(h_input_nodes, num_input_nodes, MPI_DOUBLE, local_input_node, num_input_nodes, MPI_DOUBLE, comm);
    // MPI_Bcast(&j, 1, MPI_INT, 0, comm);

    printf("\nProcess Number: %d\n", process_rank);
    printf("%f", local_row[0]);

    MPI_Finalize(); 

   return 0;
 }