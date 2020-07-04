#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "mpi.h"

void printMatrix(char prompt[], double local_A[], int col, int columns_per_process, int row, int rank, MPI_Comm comm) {
    double *matrix;
    if (rank == 0) {
        matrix = malloc(row * col * sizeof(double));
        printf("%s Matrix: \n", prompt);

        MPI_Gather(local_A, columns_per_process * row, MPI_DOUBLE, matrix, columns_per_process * row, MPI_DOUBLE, 0, comm);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%f ", matrix[i * col + j]);
            }
            printf("\n");
        }
        printf("\n");
        free(matrix);
    } else {
        MPI_Gather(local_A, columns_per_process * row, MPI_DOUBLE, matrix, columns_per_process * row, MPI_DOUBLE, 0, comm);
    }
}

void printArray(char prompt[], double local_array[], int col, int rank, MPI_Comm comm) {
    if (rank == 3) {
        printf("%s Vector: \n", prompt);
        for (int i = 0; i < col; i++) {
            printf("%f ", local_array[i]);
        }
        printf("\n");
    }
}

void allocateArrays(double **local_hidden_weights, double **input_nodes, double **local_hidden_nodes, double **local_output_nodes, int num_input_nodes, int columns_per_process, int num_output_nodes, MPI_Comm comm) {
    *local_hidden_weights = malloc(columns_per_process * num_input_nodes * sizeof(double));
    *input_nodes = malloc(num_input_nodes * sizeof(double));
    *local_hidden_nodes = malloc(columns_per_process * sizeof(double));
    *local_output_nodes = malloc(num_output_nodes * sizeof(double));
}

void createMatrix(char prompt[], double local_a[], int col, int columns_per_process, int row, int rank, MPI_Comm comm) {
    double *A;
    int upperBound = 9;
    int lowerBound = 1;

    if (rank == 0) {
        A = malloc(row * col * sizeof(double));
        srand(200);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                A[i * col + j] = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
            }
        }
        MPI_Scatter(A, columns_per_process * row, MPI_DOUBLE, local_a, columns_per_process * row, MPI_DOUBLE, 0, comm);
        // free(A);
    } else {
        MPI_Scatter(A, columns_per_process * row, MPI_DOUBLE, local_a, columns_per_process * row, MPI_DOUBLE, 0, comm);
    }
}

void createVector(char prompt[], double local_b[], int col, int rank, MPI_Comm comm) {
    // double *B;
    int upperBound = 9;
    int lowerBound = 1;

    if (rank == 0) {
        srand(200);
        for (int i = 0; i < col; i++) {
            local_b[i] = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
        }
    }
    MPI_Bcast(local_b, col, MPI_DOUBLE, 0, comm);
}

void matrix_multiply_mpi(double local_a[], double local_b[], double local_ab[], int row_per_processor, int col, int columns_per_process, MPI_Comm comm) {
    double *B;
    B = malloc(col * sizeof(double));

    MPI_Allgather(local_b, columns_per_process, MPI_DOUBLE, B, columns_per_process, MPI_DOUBLE, comm);

    for (int i = 0; i < row_per_processor; i++) {
        local_ab[i] = 0.0;
        for (int j = 0; j < col; j++) {
            local_ab[i] += local_a[i * col + j] * B[j];
        }
        // local_ab[i] = 1/(1+exp(-1*local_ab[i])); // sigmoid function
    }
    free(B);
}

double *gatherArray(double local_array[], int col, int columns_per_process, int rank, MPI_Comm comm) {
    double *array = NULL;

    if (rank == 0) {
        array = malloc(col * sizeof(double));

        MPI_Gather(local_array, columns_per_process, MPI_DOUBLE, array, columns_per_process, MPI_DOUBLE, 0, comm);
        for (int i = 0; i < col; i++) {
            array[i*col] = array[i];
        }
        // free(array);
    } else {
        MPI_Gather(local_array, columns_per_process, MPI_DOUBLE, array, columns_per_process, MPI_DOUBLE, 0, comm);
    }
    return  array;
}

void printResult(double *array, int width) {
    for (int i = 0; i < width; i++) {
        printf("%3.3f ", array[i]);
    }
    printf("\n");
}

void main(int argc, char *argv[]) {
    int num_input_nodes = 6;                                      // number of input layer nodes - columns
    int num_hidden_nodes = 8;                                     // number of hidden layer nodes - rows
    int num_output_nodes = 1;                                     // number of output nodes
    int num_hidden_weights = num_input_nodes * num_hidden_nodes;  // num of weights = num of input nodes x num of hidden nodes
    int num_output_weights = num_hidden_nodes * num_output_nodes;

    double *local_hidden_weights;
    double *local_output_weights;

    double *local_hidden_nodes;
    double *local_output_nodes;

    double *input_nodes;
    double *hidden_nodes;
    double *output_nodes;

    int row_l1, rows_per_process_l1, col_l1, columns_per_process_l1, row_l2, rows_per_process_l2, col_l2, columns_per_process_l2;
    int rank, comm_size;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    columns_per_process_l1 = num_hidden_nodes / comm_size;

    row_l2 = num_hidden_nodes;
    col_l2 = num_output_nodes;
    rows_per_process_l2 = row_l2 / comm_size;
    columns_per_process_l2 = num_output_nodes;

    allocateArrays(&local_hidden_weights, &input_nodes, &local_hidden_nodes, &local_output_nodes, num_input_nodes, columns_per_process_l1, num_output_nodes, comm);

    // create layer 1 weights data
    createMatrix("Layer 1 Weights", local_hidden_weights, num_hidden_nodes, columns_per_process_l1, num_input_nodes, rank, comm);
    printMatrix("Layer 1 Weights", local_hidden_weights, num_hidden_nodes, columns_per_process_l1, num_input_nodes, rank, comm);
    MPI_Barrier(comm);

    if (rank==1){
        printf("weights: %f \n", local_hidden_weights[3]);
    }

    // create input nodes data
    createVector("Input Node", input_nodes, num_input_nodes, rank, comm);
    printArray("Input Node", input_nodes, num_input_nodes, rank, comm);

    // multiply input nodes with layer 1 weights to find hidden node values
    matrix_multiply_mpi(local_hidden_weights, local_input_nodes, local_hidden_nodes, rows_per_process_l1, col_l1, columns_per_process_l1, comm);
    printArray("Matrix Multiplication",local_hidden_nodes, col_l1, columns_per_process_l1, rank, comm);


    // hidden_nodes = gatherArray(local_hidden_nodes, row, rows_per_process, rank, comm);
    // if (rank == 0){
    //     printf("Hidden Nodes \n");
    //     printResult(hidden_nodes, num_hidden_nodes);
    // }

    // createVector("Layer 2 Weights", local_output_weights, col, columns_per_process, rank, comm);
    // printArray("Layer 2 Weights", local_output_weights, col, columns_per_process, rank, comm);
    
    // matrix_multiply_mpi(local_output_weights, local_hidden_nodes, local_output_nodes, num_hidden_nodes/comm_size, num_output_nodes, num_output_nodes/comm_size, comm);
    // // printf("local_hidden_nodes: %f \n", local_output_nodes[0]);
    // printArray("Matrix Multiplication 2",local_output_nodes, row, rows_per_process, rank, comm);


    MPI_Finalize();
}