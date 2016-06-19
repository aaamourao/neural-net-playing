all:
	gcc -Wall -c nn-gsl.c
	gcc nn-gsl.o -lgsl -lgslcblas -lm
