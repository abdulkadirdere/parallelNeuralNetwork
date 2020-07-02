#define N 4
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"


void printMatrix(double local_A[], int row, int row_per_processor, int col,int rank,MPI_Comm  comm) {
        double *matrix;
        if (rank == 0) {
                matrix = malloc(row * col * sizeof(double));
                printf("Matrix: \n");

                MPI_Gather(local_A, row_per_processor*col, MPI_DOUBLE, matrix, row_per_processor*col, MPI_DOUBLE, 0, comm);
                for (int i = 0; i < row; i++) {
                        for (int j = 0; j < col; j++){
                                printf("%f ", matrix[i*col+j]);
                        }
                        printf("\n");
                }
        printf("\n");
        free(matrix);
        } else {
                MPI_Gather(local_A, row_per_processor*col, MPI_DOUBLE, matrix, row_per_processor*col, MPI_DOUBLE, 0, comm);
        }
}

void printArray(double local_array[], int col, int columns_per_process,int rank, MPI_Comm  comm) {
   double* array;

   if (rank == 0) {
      array = malloc(col * sizeof(double));

      MPI_Gather(local_array, columns_per_process, MPI_DOUBLE, array, columns_per_process, MPI_DOUBLE, 0, comm);
      printf("Array:\n");
      for (int i = 0; i < col; i++){
              printf("%f ", array[i]);
      }
      printf("\n");
      free(array);
   }  else {
      MPI_Gather(local_array, columns_per_process, MPI_DOUBLE, array, columns_per_process, MPI_DOUBLE, 0, comm);
   }
}

void allocateArrays(double **local_a, double **local_b, double **local_ab, int row_per_processor, int col, int columns_per_process, MPI_Comm comm){
        *local_a = malloc(row_per_processor*col*sizeof(double));
        *local_b = malloc(columns_per_process*sizeof(double));
        *local_ab = malloc(row_per_processor*sizeof(double));
}

void createMatrix(char prompt[], double local_a[], int row, int row_per_processor, int col, int rank, MPI_Comm comm){
        double *A;
        int upperBound = 9;
        int lowerBound = 1;

        if (rank == 0) {
		A = malloc(row * col * sizeof(double));
		srand(200);
		for (int i = 0; i < row; i++){
                        for (int j = 0; j < col; j++){
                                A[i*col+j] = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
                        }
                }
		MPI_Scatter(A, row_per_processor * col, MPI_DOUBLE, local_a, row_per_processor * col, MPI_DOUBLE, 0, comm);
		free(A);
	} else {
		MPI_Scatter(A, row_per_processor * col, MPI_DOUBLE, local_a, row_per_processor * col, MPI_DOUBLE, 0, comm);
	}
}

void createVector(char prompt[], double local_b[], int col, int columns_per_process, int rank, MPI_Comm comm){
        double *B;
        int upperBound = 9;
        int lowerBound = 1;

        if (rank == 0) {
		B = malloc(col * sizeof(double));
		srand(200);
		for (int i = 0; i < col; i++){
                        B[i] = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
                }
                MPI_Scatter(B, columns_per_process, MPI_DOUBLE, local_b, columns_per_process, MPI_DOUBLE, 0, comm);
		free(B);
	} else {
		MPI_Scatter(B, columns_per_process, MPI_DOUBLE, local_b, columns_per_process, MPI_DOUBLE, 0, comm);
	}

}

void matrix_multiply_mpi(double local_a[], double local_b[], double local_ab[], int row_per_processor, int col, int columns_per_process, MPI_Comm  comm) {
   double* B;
   B = malloc(col * sizeof(double));
   
   MPI_Allgather(local_b, columns_per_process, MPI_DOUBLE, B, columns_per_process, MPI_DOUBLE, comm);

   for (int i = 0; i < row_per_processor; i++) {
      local_ab[i] = 0.0;
      for (int j = 0; j < col; j++){
              local_ab[i] += local_a[i*col+j]*B[j];
        }
        // local_ab[i] = 1/(1+exp(-1*local_ab[i])); // sigmoid function
   }
   free(B);
} 


void main(int argc, char *argv[]){
        int num_input_nodes = 8; // number of input layer nodes - columns
        int num_hidden_nodes = 8; // number of hidden layer nodes - rows
        int num_output_nodes = 1; // number of output nodes
        int num_hidden_weights = num_input_nodes * num_hidden_nodes; // num of weights = num of input nodes x num of hidden nodes
        int num_output_weights = num_hidden_nodes * num_output_nodes;

        double *local_hidden_weights;
        double *local_input_nodes;
        double *local_hidden_nodes;
        int row, rows_per_process, col, columns_per_process;
        int rank, comm_size;
        MPI_Comm comm;

        MPI_Init(NULL, NULL);
        comm = MPI_COMM_WORLD;
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &rank);

        row = num_input_nodes;
        col = num_hidden_nodes;
        rows_per_process = row/comm_size;
        columns_per_process = col/comm_size;

        // Get_dims(&row, &rows_per_process, &col, &columns_per_process, rank, comm_size, comm);

        allocateArrays(&local_hidden_weights, &local_input_nodes, &local_hidden_nodes, rows_per_process, col, columns_per_process, comm);

        createMatrix("A", local_hidden_weights, row, rows_per_process, col, rank, comm);
        printMatrix(local_hidden_weights, row, rows_per_process, col, rank, comm);
        createVector("B", local_input_nodes, col, columns_per_process, rank, comm);
        printArray(local_input_nodes, col, columns_per_process, rank, comm);
        
        matrix_multiply_mpi(local_hidden_weights, local_input_nodes, local_hidden_nodes, rows_per_process, col, columns_per_process, comm);
        printArray(local_hidden_nodes, row, rows_per_process, rank, comm);

        MPI_Finalize();
}