#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<strings.h>

#define D 5
#define N 2
#define ITERATIONS 10

typedef struct DataPoint {
	double x[D];
 	double y;
}DataPoint;

double dot(double x[D], double y[D]) {
    double ans = 0.0;
    int i = 0;
    while( i < D) {
        ans += x[i] * y[i];
        i += 1;
    }
    return ans;
}

muls(double result[D], double x[D], double c) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * c;
    }
    return 0;
}

map(double result[D], double x[D], double y, double w[D]) {
    muls(result, x, (1 / (1 + exp(-y * (dot(w, x)))) - 1) * y);
    return 0;
}

add(double result[D], double x[D], double y[D]) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * y[i];
    }
    return 0;
}

subself(double x[D], double y[D]) {
    int i;
    for(i=0; i<D; i++)
        x[i] -= y[i];
    return 0;
}

mapAll(double result[N * D], double x[N * D], double y[D], double w[D]) {
    int i=0;
    for(i=0;i<N;i++) {
        map(&result[i * D], &x[i * D ], y[i],w);
    }
}

reduceAll(double result[D], double inp[N * D]) {
    int i,j;
   for(i=0;i<N;i++)
    for(j=0;j<D;j++)
    {
        result[j] += inp[i * D + j];
    } 
}

print(char *a, double p[D]) {
    int i=0;
    printf("%s",a);
    for(i=0;i<D;i++)
        printf("%e, ",p[i]);
    printf("\n");
}

main(int argc, char * argv[])
{
    DataPoint DataSet[N] = { {{1.0,2.0,3.0,4.0,5.0},1.0 }, {{1.0,2.0,3.0,4.0,5.0},1.0 } };
    double x[N][D] = { {1.0,2.0,3.0,4.0,5.0}, {1.0,2.0,3.0,4.0,5.0}};
    double y[D]    = { 1.0, 1.0 };
    double w[D] = { 0.9706523792091876, 0.9689950601959192, 0.8871562128145087, 0.64360369303987, 0.905075008043427 };
    double mr[N][D];
    int i=0,j;
    double *x1 = (double *) x;
    double *mr1 = (double *) mr;
    double rr[D] = { 0 };

    for(i=0;i<ITERATIONS;i++) {
        bzero((void *)rr,sizeof(rr));
        mapAll(mr1,x1,y,w);
        //print("MAP 0 ", mr1);
        //print("MAP 1 ", mr1 + D);
        reduceAll(rr,mr1);
        //print("rr--->", rr);
        //print("W--->", w);
        subself(w,rr);
        //print("(W(-)>", w);
        //printf("\n\n");

    }

    print("rr-->", rr);
}
