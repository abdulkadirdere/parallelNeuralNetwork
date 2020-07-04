/* File:     mpi_mat_vect_mult.c
 *
 * Purpose:  Implement parallel matrix-vector multiplication using
 *           one-dimensional arrays to store the vectors and the
 *           matrix.  Vectors use block distributions and the
 *           matrix is distributed by block rows.
 *
 * Compile:  mpicc -g -Wall -o mpi_mat_vect_mult mpi_mat_vect_mult.c
 * Run:      mpiexec -n <number of processes> ./mpi_mat_vect_mult
 *
 * Input:    Dimensions of the matrix (m = number of rows, n= number of columns)
 *           m x n matrix A
 *           n-dimensional vector x
 * Output:   Product vector y = Ax
 *
 * Errors:   If an error is detected (m or n negative, m or n not evenly
 *           divisible by the number of processes, malloc fails), the
 *           program prints a message and all processes quit.
 *
 * Notes:     
 *    1. Number of processes should evenly divide both m and n
 *    2. Define DEBUG for verbose output
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEBUG 1

void Get_dims(int* m_p, int* local_m_p, int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_arrays(double** local_A_pp, double** local_x_pp, double** local_y_pp, int local_m, int n, int local_n, MPI_Comm comm);
void Random_matrix(char prompt[], double local_A[], int m, int local_m, int n, int my_rank, MPI_Comm comm);
void Random_vector(char prompt[], double local_vec[], int n, int local_n, int my_rank, MPI_Comm comm);
void Print_matrix(char title[], double local_A[], int m, int local_m, int n, int my_rank, MPI_Comm comm);
void Print_vector(char title[], double local_vec[], int n,int local_n, int my_rank, MPI_Comm comm);
void Mat_vect_mult(double local_A[], double local_x[], double local_y[], int local_m, int n, int local_n, MPI_Comm comm);
/*-------------------------------------------------------------------*/
int main(void) {
   double* local_A;
   double* local_x;
   double* local_y;
   int m, local_m, n, local_n;
   int my_rank, comm_sz;
   MPI_Comm comm;

   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);
   Allocate_arrays(&local_A, &local_x, &local_y, local_m, n, local_n, comm);
   Random_matrix("A", local_A, m, local_m, n, my_rank, comm);
#  ifdef DEBUG
   Print_matrix("A", local_A, m, local_m, n, my_rank, comm);
#  endif
   Random_vector("x", local_x, n, local_n, my_rank, comm);
#  ifdef DEBUG
   Print_vector("x", local_x, n, local_n, my_rank, comm);
#  endif

   Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, comm);
#  ifdef DEBUG
   Print_vector("y", local_y, m, local_m, my_rank, comm);
//    printf("m: %d   - local_m: %d  ---- local_n: %d \n", m,local_m, local_n);
#  endif

   free(local_A);
   free(local_x);
   free(local_y);
   MPI_Finalize();
   return 0;
}  /* main */


/*-------------------------------------------------------------------
 * Function:  Get_dims
 * Purpose:   Get the dimensions of the matrix and the vectors from
 *            stdin.
 * In args:   my_rank:   calling processes rank in comm
 *            comm_sz:   number of processes in comm
 *            comm:      communicator containing all processes calling
 *                       Get_dims
 * Out args:  m_p:       global number of rows of A and components in y
 *            local_m_p: local number of rows of A and components of y
 *            n_p:       global number of cols of A and components of x
 *            local_n_p: local number of components of x
 *
 * Errors:    if either m or n isn't positive or if m or n isn't evenly
 *            divisible by comm_sz, the program prints an error message
 *            and quits.
 * Note:
 *    All processes in comm should call Get_dims
 */
void Get_dims(
      int*      rows        /* out */, 
      int*      rows_per_process  /* out */,
      int*      columns        /* out */,
      int*      columns_per_process  /* out */,
      int       my_rank    /* in  */,
      int       comm_sz    /* in  */,
      MPI_Comm  comm       /* in  */) {

   if (my_rank == 0) {
      printf("Enter the number of rows\n");
      scanf("%d", rows);
      printf("Enter the number of columns\n");
      scanf("%d", columns);
   }
   MPI_Bcast(rows, 1, MPI_INT, 0, comm);
   MPI_Bcast(columns, 1, MPI_INT, 0, comm);

   *rows_per_process = *rows/comm_sz;
   *columns_per_process = *columns/comm_sz;
}  /* Get_dims */

/*-------------------------------------------------------------------
 * Function:   Allocate_arrays
 * Purpose:    Allocate storage for local parts of A, x, and y
 * In args:    local_m:    local number of rows of A and components of y
 *             n:          global and local number of cols of A and global
 *                         number of components of x
 *             local_n:    local number of components of x
 *             comm:       communicator containing all calling processes
 * Out args:   local_A_pp: local storage for matrix (m/comm_sz rows, n cols)
 *             local_x_pp: local storage for x (n/comm_sz components)
 *             local_y_pp: local_storage for y (m/comm_sz components) 
 *
 * Errors:     if a malloc fails, the program prints a message and all
 *             processes quit
 * Note:
 *    Communicator should be MPI_COMM_WORLD because of call to 
 * Check_for_errors
 */
void Allocate_arrays(
      double**  local_A_pp  /* out */, 
      double**  local_x_pp  /* out */, 
      double**  local_y_pp  /* out */, 
      int       local_m     /* in  */, 
      int       n           /* in  */,   
      int       local_n     /* in  */, 
      MPI_Comm  comm        /* in  */) {

   *local_A_pp = malloc(local_m*n*sizeof(double));
   *local_x_pp = malloc(local_n*sizeof(double));
   *local_y_pp = malloc(local_m*sizeof(double));

}  /* Allocate_arrays */

/*-------------------------------------------------------------------
 * Function:  Random_matrix
 * Purpose:   Generate random elements for the matrix and distribute 
 *            among the processes using a block row distribution
 * In args:   m:       global number of rows of A
 *            local_m: local number of rows of A
 *            n:       global and local number of cols of A
 *            my_rank: process rank in communicator comm
 *            comm:    communicator containing processes calling
 *                     Random_matrix
 * Out args:  local_A: the local matrix
 * Errors:    if malloc of temporary storage fails on process 0, the 
 *            program prints a message and all processes quit
 * Note:
 * 1. Communicator should be MPI_COMM_WORLD because of call to 
 *    Check_for_errors
 * 2. local_m and n should be the same on each process
*/
void Random_matrix(
			char prompt[],		/* in  */ 
			double local_A[], /* out */
			int m, 						/* in */
			int local_m, 			/* in */
			int n, 						/* in */
			int my_rank, 			/* in */
			MPI_Comm comm			/* in */) {
	double *A = NULL;
	int i, j;
	if (my_rank == 0) {
		A = malloc(m*n*sizeof(double));

#		ifdef DEBUG
		printf("Generating the matrix %s\n", prompt);
#		endif
		srand(2018);
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				A[i*n+j] = (double)rand( ) / RAND_MAX;
		MPI_Scatter(A, local_m*n, MPI_DOUBLE, local_A, local_m*n, MPI_DOUBLE, 0, comm);
		free(A);
	} else {
		MPI_Scatter(A, local_m*n, MPI_DOUBLE, local_A, local_m*n, MPI_DOUBLE, 0, comm);
	}
} /* Random_matrix */

/*-------------------------------------------------------------------
 * Function:  Random_vector
 * Purpose:   Generate random elements for the vector and distribute 
 *            among the processes using a block distribution
 * In args:   n:       global order of vector
 *            local_n: local order of vector (n/comm_sz)
 *            my_rank: process rank in communicator comm
 *            comm:    communicator containing processes calling
 *                     Random_matrix
 * Out args:  local_vec: the local vector
 * Errors:    if malloc of temporary storage fails on process 0, the 
 *            program prints a message and all processes quit
 * Note:
 * 1. Communicator should be MPI_COMM_WORLD because of call to 
 *    Check_for_errors
 * 2. local_n should be the same on each process
*/
void Random_vector(
			char prompt[],			/* in  */ 
			double local_vec[],	/* out */
			int n, 							/* in */
			int local_n, 				/* in */
			int my_rank, 				/* in */
			MPI_Comm comm				/* in */) {
	double *vec = NULL;
	int i;
	if (my_rank == 0) {
		vec = malloc(n*sizeof(double));

#		ifdef DEBUG
		printf("Generating the vector %s\n", prompt);
#		endif
		srand(2018);
		for (i = 0; i < n; i++)
			vec[i] = (double)rand( ) / RAND_MAX;
		MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0, comm);
		free(vec);
	} else {
		MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0, comm);
	}
} /* Random_vector */

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
   int i, j, local_ok = 1;

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

/*-------------------------------------------------------------------
 * Function:  Print_vector
 * Purpose:   Print a vector with a block distribution
 * In args:   title:      name of vector
 *            local_vec:  calling process' part of vector
 *            n:          global number of components
 *            local_n:    local number of components (n/comm_sz)
 *            my_rank:    calling process' rank in comm
 *            comm:       communicator containing all processes
 * Errors:    if malloc of local storage on process 0 fails, all
 *            processes quit.            
 * Notes:
 * 1.  comm should be MPI_COMM_WORLD because of call to Check_for_errors
 * 2.  local_n should be the same on all the processes
 */
void Print_vector(
      char      title[]     /* in */, 
      double    local_vec[] /* in */, 
      int       n           /* in */,
      int       local_n     /* in */,
      int       my_rank     /* in */,
      MPI_Comm  comm        /* in */) {
   double* vec = NULL;
   int i;

   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));

      MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
      printf("\nThe vector %s\n", title);
      for (i = 0; i < n; i++)
         printf("%f ", vec[i]);
      printf("\n");
      free(vec);
   }  else {
      MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
   }
}  /* Print_vector */

/*-------------------------------------------------------------------
 * Function:  Mat_vect_mult
 * Purpose:   Multiply a matrix A by a vector x.  The matrix is distributed
 *            by block rows and the vectors are distributed by blocks
 * In args:   local_A:  calling process' rows of matrix A
 *            local_x:  calling process' components of vector x
 *            local_m:  calling process' number of rows 
 *            n:        global (and local) number of columns
 *            local_n:  calling process' number of components of x
 *            comm:     communicator containing all calling processes
 * Errors:    if malloc of local storage on any process fails, all
 *            processes quit.            
 * Notes:
 * 1.  comm should be MPI_COMM_WORLD because of call to Check_for_errors
 * 2.  local_m and local_n should be the same on all the processes
 */
void Mat_vect_mult(
      double    local_A[]  /* in  */, 
      double    local_x[]  /* in  */, 
      double    local_y[]  /* out */,
      int       local_m    /* in  */, 
      int       n          /* in  */,
      int       local_n    /* in  */,
      MPI_Comm  comm       /* in  */) {
   double* x;
   int local_i, j;

   x = malloc(n*sizeof(double));
   
   MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);

   for (local_i = 0; local_i < local_m; local_i++) {
      local_y[local_i] = 0.0;
      for (j = 0; j < n; j++)
         local_y[local_i] += local_A[local_i*n+j]*x[j];
   }
   free(x);
}  /* Mat_vect_mult */
