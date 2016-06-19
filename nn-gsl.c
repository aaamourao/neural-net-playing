#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

typedef struct {
  // number of inputs into the layer
  int num_inputs;

  // number of neurons in the layer (= number of outputs)
  int num_neurons;

  // weight matrix where each neuron is a row and each input weight is a column
  gsl_matrix *w;

  // bias vector - note that this is implemented as a matrix -- it will be a
  // Nx1 matrix (column-vector)
  gsl_matrix *b;
} fc_layer; // a fully-connected neural network layer

fc_layer *create_fc_layer(int neurons, int inputs) {
  fc_layer *layer = malloc(sizeof *layer);

  layer->num_inputs = inputs;
  layer->num_neurons = neurons;
  
  // create space for the weight matrix
  layer->w = gsl_matrix_alloc(neurons, inputs);

  // create space for the bias vector and initialize to 0
  layer->b = gsl_matrix_calloc(neurons, 1);

  // construct a random number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
  
  // initialize the weights
  int i;
  int j;
  for (i = 0; i < neurons; i++) {
    for (j = 0; j < inputs; j++) {
      gsl_matrix_set(layer->w, i, j, gsl_ran_gaussian(rng, 0.01));
    }
  }

  return layer;
}

void release_fc_layer(fc_layer *n) {
  gsl_matrix_free(n->w);
  gsl_matrix_free(n->b);
}

gsl_matrix* get_layer_output(fc_layer *n, gsl_matrix *x) {
  printf("Matrix\n");
  int w, y;
  for (w = 0; w < n->num_neurons; w++) {
    for (y = 0; y < n->num_inputs; y++) {
      printf("%f ", gsl_matrix_get(n->w, w, y));
    }
    printf("\n");
  }

  printf("\n\nVector\n");
  for (w = 0; w < n->num_inputs; w++) {
    printf("%f\n", gsl_matrix_get(x, w, 0));
  }
  printf("\n\n");

  gsl_matrix *outputs = gsl_matrix_alloc(n->num_neurons, 1);

  // compute w.x
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, n->w, x, 0.0, outputs);

  // add b
  gsl_matrix_add(outputs, n->b);

  // apply tanh nonlinearity elementwise
  int i;
  for (i = 0; i < n->num_neurons; i++) {
    gsl_matrix_set(outputs, i, 0, tanh(gsl_matrix_get(outputs, i, 0)));
  }

  return outputs;
}

int main(void) {
  int inputs = 4;
  int neurons = 7;
  
  fc_layer *layer = create_fc_layer(neurons, inputs); // input layer
  fc_layer *out_node = create_fc_layer(1, neurons); // feed into a single output node

  gsl_matrix *in = gsl_matrix_alloc(inputs, 1);
  int i;
  for (i = 0; i < inputs; i++) {
    gsl_matrix_set(in, i, 0, i); // in_i0 = i
  }
  
  gsl_matrix *layer_out = get_layer_output(layer, in);
  layer_out = get_layer_output(out_node, layer_out);

  for (i = 0; i < 1; i++) {
    printf("out_%d0 %g\n", i, gsl_matrix_get(layer_out, i, 0));
  }

  release_fc_layer(layer);
  gsl_matrix_free(layer_out);
  gsl_matrix_free(in);

  return 0;
}
