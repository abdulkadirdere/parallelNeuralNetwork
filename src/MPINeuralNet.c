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

// double neural_net_mpi(int num_input_nodes, int num_hidden_nodes, int num_output_nodes, int num_hidden_weights, int num_output_weights){
//     // generate input nodes
//     double *h_input_nodes = createArray(num_input_nodes);

//     // allocate memory for hidden_nodes and output nodes
//     double *h_hidden_nodes = (double *)malloc(num_hidden_nodes * sizeof(double *));
//     double *h_output_nodes = (double *)malloc(num_output_nodes * sizeof(double *));

//     // generate initial weights for hidden and output layer
//     double *h_hidden_weights= createWeights(num_hidden_weights);
//     double *h_output_weights= createWeights(num_output_weights);

//     //-------------- MPI Neural Network --------------//
//     // MPI timing of event


//     for (int epoch=0; epoch<epochs; epoch++){
//         // matrix multiplication hidden layer
//         h_hidden_nodes = matrix_multiply_seq(h_input_nodes, h_hidden_weights, h_hidden_nodes, num_input_nodes, num_hidden_nodes);

//         h_output_nodes = matrix_multiply_seq(h_hidden_nodes, h_output_weights, h_output_nodes, num_hidden_nodes, num_output_nodes);
//         double predicted = h_output_weights[0];
        
//         // weights must be updated
//         h_output_weights = backprop_output(h_output_weights, h_hidden_nodes, predicted, actual, num_hidden_nodes);

//         h_hidden_weights = backprop_hidden(h_hidden_weights, h_output_weights, h_input_nodes, predicted, actual, num_input_nodes, num_hidden_nodes);
//         // printArray(h_hidden_weights, num_hidden_weights);

//         // calculate the error
//         double error_value = error(predicted);
//         // printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
//         if (error_value < 1){
//             printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
//             break;
//         }
//     }


//     //-------------- Free Memory --------------//

//     return 0;
// }


// a mpi version of matrix multiplication
double *matrix_multiply_mpi(double *a, double *b, double *ab, int row, int col){
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

// void Check_for_error(
//       int       local_ok   /* in */, 
//       char      fname[]    /* in */,
//       char      message[]  /* in */, 
//       MPI_Comm  comm       /* in */) {
//    int ok;

//    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
//    if (ok == 0) {
//       int my_rank;
//       MPI_Comm_rank(comm, &my_rank);
//       if (my_rank == 0) {
//          fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, 
//                message);
//          fflush(stderr);
//       }
//       MPI_Finalize();
//       exit(-1);
//    }
// }  /* Check_for_error */

/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print a matrix distributed by block rows to stdout
 * In args:   title:    name of matrix
 *            local_A:  calling process' part of matrix
 *            m:        global number of rows
 *            local_m:  local number of rows (m/comm_sz)
 *            n:        global (and local) number of cols
 *            my_rank:  calling process' rank in comm
 *            comm:     communicator containing all processes
 * Errors:    if malloc of local storage on process 0 fails, all
 *            processes quit.            
 * Notes:
 * 1.  comm should be MPI_COMM_WORLD because of call to Check_for_errors
 * 2.  local_m should be the same on all the processes
 */
void Print_matrix(
      char      title[]    /* in */,
      double    local_A[]  /* in */, 
      int       m          /* in */, 
      int       local_m    /* in */, 
      int       n          /* in */,
      int       my_rank    /* in */,
      MPI_Comm  comm       /* in */) {
   double* A = NULL;
   int i, j = 1;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));
      MPI_Gather(local_A, local_m*n, MPI_DOUBLE, A, local_m*n, MPI_DOUBLE, 0, comm);
      printf("\nThe matrix %s\n", title);
      for (i = 0; i < m; i++) {
         for (j = 0; j < n; j++)
            printf("%f ", A[i*n+j]);
         printf("\n");
      }
      printf("\n");
      free(A);
   } else {
        MPI_Gather(local_A, local_m*n, MPI_DOUBLE, A, local_m*n, MPI_DOUBLE, 0, comm);
   }
}  /* Print_matrix */


int main(int argc, char **argv){ 
    int my_rank, comm_sz;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank == 0){
            // initialisation
        int num_input_nodes = 4; // number of input layer nodes
        int num_hidden_nodes = 4; // number of hidden layer nodes
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
        printArray(h_hidden_weights, num_hidden_weights);

        // // matrix multiplication hidden layer
        // h_hidden_nodes = matrix_multiply_mpi(h_input_nodes, h_hidden_weights, h_hidden_nodes, num_input_nodes, num_hidden_nodes);

        double *local_A = malloc(num_input_nodes * sizeof(double));
        for (int i=0; i < num_input_nodes; i++){
            for (int j=0; j< num_hidden_nodes; j++){
                // h_hidden_weights[i*num_hidden_nodes+j] = 0.0;
                MPI_Scatter(h_hidden_weights, num_input_nodes, MPI_DOUBLE, local_A, num_hidden_nodes, MPI_DOUBLE, 0, comm);
            }
        }
        // Print_matrix("Layer 1 Weights", local_A, num_input_nodes, num_hidden_nodes, 1, my_rank, comm);
        printf("%f \n", local_A[0]);
    }
    // } else {
    //     MPI_Scatter(h_hidden_weights, num_input_nodes, MPI_DOUBLE, local_A, num_hidden_nodes, MPI_DOUBLE, 0, comm);
    // }
    


    MPI_Finalize(); 

  //-------------- MPI Neural Network --------------//
  // MPI timing of event


//   for (int epoch=0; epoch<epochs; epoch++){
//       // matrix multiplication hidden layer
//       h_hidden_nodes = matrix_multiply_seq(h_input_nodes, h_hidden_weights, h_hidden_nodes, num_input_nodes, num_hidden_nodes);

//       h_output_nodes = matrix_multiply_seq(h_hidden_nodes, h_output_weights, h_output_nodes, num_hidden_nodes, num_output_nodes);
//       double predicted = h_output_weights[0];
      
//       // weights must be updated
//       h_output_weights = backprop_output(h_output_weights, h_hidden_nodes, predicted, actual, num_hidden_nodes);

//       h_hidden_weights = backprop_hidden(h_hidden_weights, h_output_weights, h_input_nodes, predicted, actual, num_input_nodes, num_hidden_nodes);
//       // printArray(h_hidden_weights, num_hidden_weights);

//       // calculate the error
//       double error_value = error(predicted);
//       // printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
//       if (error_value < 1){
//           printf("Epoch:%d - Error:%3.4f  - Predicted:%3.4f \n", epoch, error_value, predicted);
//           break;
//       }
//   }



  // neural_net_mpi(num_input_nodes, num_hidden_nodes, num_output_nodes, num_hidden_weights, num_output_weights);

  //  int num_proces, myrank;

  //  MPI_Init(&argc, &argv);

  //  MPI_Comm_size(MPI_COMM_WORLD, &num_proces);
  //  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  //  printf("From process %d out of %d, Hello World!\n", myrank, num_proces);

  //  MPI_Finalize(); 
   return 0;
 }