/**
 *   \file my_it_mat_vect_mult.c
 *   \brief Multiplica iterativamente un matriz nxn 
 *          por un vector de n posiciones
 *
 *   \author Danny Múnera
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* función para generar <size> cantidad de datos aleatorios */
void gen_data(double * array, int size);
/* función para multiplicar iterativamente un matriz 
 * <m x n> por un vector de tam <n> */
void mat_vect_mult(double* A, double* x, double* y, int n, int it);
/* función para imprimir un vector llamado <name> de tamaño <m>*/
void print_vector(char* name, double*  y, int m);

void Mat_vect_mult_par(
  double    local_A[]  /* in  */, 
  double    local_x[]  /* in  */, 
  double    local_y[]  /* out */,
  int       local_m    /* in  */, 
  int       n          /* in  */,
  int       local_n    /* in  */,
  int iters,
  MPI_Comm  comm       /* in  */);

void Print_vector_par(
  char      title[]     /* in */, 
  double    local_vec[] /* in */, 
  int       n           /* in */,
  int       local_n     /* in */,
  int       my_rank     /* in */,
  MPI_Comm  comm        /* in */);

void scatter_matrix(
  double* A,
  double local_A[],
  int n,
  int local_n,
  int my_rank,
  MPI_Comm comm);

void scatter_vector(
  double* x,
  double local_x[],
  int n,
  int local_n,
  int my_rank,
  MPI_Comm comm);

void get_input_dims(
  int* n_p,
  int* local_n_p,
  int* iters_p,
  long* seed_p,
  int my_rank,
  int comm_sz,
  MPI_Comm comm);

int main()
{
  double* A = NULL, * local_A;
  double* x = NULL, * local_x;
  double* y = NULL, * local_y;
  int n, iters;
  long seed;
  int my_rank, comm_size;
  int local_n;
  MPI_Comm comm;

  MPI_Init(NULL, NULL);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &my_rank);

  // Obtener las dimensiones
  get_input_dims(&n, &local_n, &iters, &seed, my_rank, comm_size, comm);

  // la matriz A tendrá una representación unidimensional
  A = malloc(sizeof(double) * n * n);
  x = malloc(sizeof(double) * n);
  y = malloc(sizeof(double) * n);

  local_A = malloc(sizeof(double) * n * n);
  local_x = malloc(sizeof(double) * n);
  local_y = malloc(sizeof(double) * n);

  //generar valores para las matrices
  gen_data(A, n*n);
  gen_data(x, n);

  scatter_matrix(A, local_A, n, local_n, my_rank, comm);
  scatter_vector(x, local_x, n, local_n, my_rank, comm);

  Mat_vect_mult_par(local_A, local_x, local_y, local_n, n, local_n, iters, comm);

  print_vector("y", local_y, n);
  Print_vector_par("Y par", local_y, n, local_n, my_rank, comm);
  MPI_Finalize();
  return 0;
}

void gen_data(double * array, int size){
  int i;
  for (i = 0; i < size; i++)
    array[i] = (double) rand() / (double) RAND_MAX;
}

void mat_vect_mult(double* A, double* x, double* y, int n, int it){
  int h, i, j;
  for(h = 0; h < it; h++){
    for(i = 0; i < n; i++){
      y[i] = 0.0;
      for(j = 0; j < n; j++)
	y[i] += A[i*n+j] * x[j];
    }
    // x <= y
    for(i = 0; i < n; i++)
      x[i] = y[i];
  }
}

void print_vector(char* name, double*  y, int m) {
   int i;
   printf("\nVector %s\n", name);
   for (i = 0; i < m; i++)
      printf("%f ", y[i]);
   printf("\n");
}

void Mat_vect_mult_par(
      double    local_A[]  /* in  */, 
      double    local_x[]  /* in  */, 
      double    local_y[]  /* out */,
      int       local_m    /* in  */, 
      int       n          /* in  */,
      int       local_n    /* in  */,
      int       iters,
      MPI_Comm  comm       /* in  */) {
   double* x;
   int local_i, j, h;
   int local_ok = 1;

   x = malloc(n*sizeof(double));
   MPI_Allgather(local_x, local_n, MPI_DOUBLE,
         x, local_n, MPI_DOUBLE, comm);

  
  for (h = 0; h < iters; h++) {
    for (local_i = 0; local_i < local_m; local_i++) {
      local_y[local_i] = 0.0;
      for (j = 0; j < n; j++) {
         local_y[local_i] += local_A[local_i*n+j]*x[j];
      }
    }

    MPI_Barrier(comm);

    for(local_i = 0; local_i < n; local_i++)
      local_A[local_i] = local_y[local_i];
  }

   
   free(x);
}


void get_input_dims(
  int* n_p,
  int* local_n_p,
  int* iters_p,
  long* seed_p,
  int my_rank,
  int comm_sz,
  MPI_Comm comm
  ){
    if (my_rank == 0) {
      printf("Ingrese la dimensión n:\n");
      scanf("%d", n_p);
      printf("Ingrese el número de iteraciones:\n");
      scanf("%d", iters_p);
      printf("Ingrese semilla para el generador de números aleatorios:\n");
      scanf("%ld", seed_p);
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(iters_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(seed_p, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    srand(*seed_p);

    *local_n_p = *n_p/comm_sz;
  }

  void scatter_matrix(
    double* A,
    double local_A[],
    int n,
    int local_n,
    int my_rank,
    MPI_Comm comm
  ) {
    int i, j;

    if (my_rank == 0) {
      MPI_Scatter(A, local_n*n, MPI_DOUBLE, 
            local_A, local_n*n, MPI_DOUBLE, 0, comm);
    } else {
      MPI_Scatter(A, local_n*n, MPI_DOUBLE, 
            local_A, local_n*n, MPI_DOUBLE, 0, comm);
    }
  }

void scatter_vector(
  double* x,
  double local_x[],
  int n,
  int local_n,
  int my_rank,
  MPI_Comm comm
) {
  double* vec = NULL;
  int i;

  if (my_rank == 0) {
    MPI_Scatter(x, local_n, MPI_DOUBLE,
            local_x, local_n, MPI_DOUBLE, 0, comm);
  } else {
    MPI_Scatter(x, local_n, MPI_DOUBLE,
            local_x, local_n, MPI_DOUBLE, 0, comm);
  }

}

void Print_vector_par(
      char      title[]     /* in */, 
      double    local_vec[] /* in */, 
      int       n           /* in */,
      int       local_n     /* in */,
      int       my_rank     /* in */,
      MPI_Comm  comm        /* in */) {
   double* vec = NULL;
   int i, local_ok = 1;

   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      if (vec == NULL) local_ok = 0;
      
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
            vec, local_n, MPI_DOUBLE, 0, comm);
      printf("\nThe vector %s\n", title);
      for (i = 0; i < n; i++)
         printf("%f ", vec[i]);
      printf("\n");
      free(vec);
   }  else {
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
            vec, local_n, MPI_DOUBLE, 0, comm);
   }
} 