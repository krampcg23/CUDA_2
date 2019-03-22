/* CSCI 563 Programming Assignment 3
   Clayton Kramp
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>

#define THREADS_PER_LINE 16
using namespace std;


__global__ void loadBalancedSpMV(float* t, float* b, int* ptr, float* data, int* ind, int n) {


    int myi = blockIdx.x * blockDim.x + threadIdx.x;

    int lb = ptr[myi / THREADS_PER_LINE];
    int ub = ptr[(myi / THREADS_PER_LINE) + 1];
    extern __shared__ float partialSum[];

    partialSum[threadIdx.x] = 0;
    partialSum[threadIdx.x + THREADS_PER_LINE] = 0;
    for (int j = lb + threadIdx.x; j < ub; j += THREADS_PER_LINE) {
        int index = ind[j];
        partialSum[threadIdx.x] += data[j] * b[index];
    }

    for (unsigned int stride = THREADS_PER_LINE; stride > 0; stride /= 2) {
         __syncthreads();
          if (threadIdx.x < stride)
               partialSum[threadIdx.x] += partialSum[threadIdx.x+stride];
    }
    if (threadIdx.x == 0)
        t[myi / THREADS_PER_LINE] = partialSum[threadIdx.x];
}

main (int argc, char **argv) {
  FILE *fp;
  char line[1024]; 
  int *ptr, *indices;
  float *data, *b, *t;
  int i,j;
  int n; // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix

  // Open input file and read to end of comments
  if (argc !=2) abort(); 

  if ((fp = fopen(argv[1], "r")) == NULL) {
    abort();
  }

  fgets(line, 128, fp);
  while (line[0] == '%') {
    fgets(line, 128, fp); 
  }

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for ptr, indices, data, b and t.
  sscanf(line,"%d %d %d\n", &nr, &nc, &n);
  ptr = (int *) malloc ((nr+1)*sizeof(int));
  indices = (int *) malloc(n*sizeof(int));
  data = (float *) malloc(n*sizeof(float));
  b = (float *) malloc(nc*sizeof(float));
  t = (float *) malloc(nr*sizeof(float));

  // Read data in coordinate format and initialize sparse matrix
  int lastr=0;
  for (i=0; i<n; i++) {
    int r;
    fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));  
    indices[i]--;  // start numbering at 0
    if (r!=lastr) { 
      ptr[r-1] = i; 
      lastr = r; 
    }
  }
  ptr[nr] = n;

  // initialize t to 0 and b with random data  
  for (i=0; i<nr; i++) {
    t[i] = 0.0;
  }

  for (i=0; i<nc; i++) {
    b[i] = (float) rand()/1111111111;
  }

    
  // TODO: Compute result on GPU and compare output
  float* deviceT;
  cudaMalloc(&deviceT, nr * sizeof(float));
  cudaMemcpy(deviceT, t, nr * sizeof(float), cudaMemcpyHostToDevice);

  float* deviceB;
  cudaMalloc(&deviceB, nc * sizeof(float));
  cudaMemcpy(deviceB, b, nc * sizeof(float), cudaMemcpyHostToDevice);

  int* devicePtr;
  cudaMalloc(&devicePtr, (nr+1) * sizeof(int));
  cudaMemcpy(devicePtr, ptr, (nr+1) * sizeof(int), cudaMemcpyHostToDevice);

  float* deviceData;
  cudaMalloc(&deviceData, n * sizeof(float));
  cudaMemcpy(deviceData, data, n * sizeof(float), cudaMemcpyHostToDevice);

  int* deviceIndices;
  cudaMalloc(&deviceIndices, n * sizeof(int));
  cudaMemcpy(deviceIndices, indices, n * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16,1,1);
  dim3 numBlocks(nr, 1, 1);
  loadBalancedSpMV<<<numBlocks, threadsPerBlock, nc+THREADS_PER_LINE>>>(deviceT, deviceB, devicePtr, deviceData, deviceIndices, n);

  float* newT = (float *) malloc(nr*sizeof(float));
  cudaMemcpy(newT, deviceT, nr*sizeof(float), cudaMemcpyDeviceToHost);

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<nr; i++) {                                                      
    for (j = ptr[i]; j<ptr[i+1]; j++) {
      t[i] = t[i] + data[j] * b[indices[j]];
    }
  }

  for (int i = 0; i < nr; i++) {
      assert(abs(newT[i] - t[i] < 0.0001));
  }
  printf("%d\n", nr);

  cudaFree(deviceT);
  cudaFree(deviceIndices);
  cudaFree(devicePtr);
  cudaFree(deviceData);
  cudaFree(deviceB);


}
