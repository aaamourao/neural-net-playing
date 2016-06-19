all:
	gcc -Wall -c gsl-example.c
	gcc gsl-example.o -lgsl -lgslcblas -lm
