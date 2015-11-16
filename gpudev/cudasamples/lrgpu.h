
typedef struct DataPoint {
	double x[D];
 	double y;
}DataPoint;

__device__ inline
double *  __shfl_down_double(double *var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double**>(&a);
}


__device__
double dot(double x[D], double y[D]) {
    double ans = 0.0;
    int i = 0;
    while( i < D) {
        ans += x[i] * y[i];
        i += 1;
    }
    return ans;
}

__device__ void
muls(double result[D], double x[D], double c) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * c;
    }
}

__device__ void 
add(double result[D], double x[D], double y[D]) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * y[i];
    }
}

void
subself(double x[D], double y[D]) {
    int i;
    for(i=0; i<D; i++)
        x[i] -= y[i];
}

void
print(char a[], double p[D]) {
    int i=0;
    printf("%s",a);
    for(i=0;i<D;i++)
        printf("%e, ",p[i]);
    printf("\n");
}

