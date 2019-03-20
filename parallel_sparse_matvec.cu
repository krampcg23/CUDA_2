/* CSCI 563 Programming Assignment 3
   Clayton Kramp
*/

#include <stdio.h>
using namespace std;

__global__ void naiveSpMV(float* t, float* b, int* ptr, float* data, int* ind, int n) {

    int tid = threadIdx.y;
    int bid = blockIdx.y;
    int myi = bid * 16 + tid;

    if (myi < n) {
        float temp = 0;
        int lb = ptr[myi];
        int ub = ptr[myi+1];
        //printf("%d, %d, \n", lb, ub);
        for (int j = lb; j < ub; j++) {
            int index = ind[j];
            temp += data[j] * b[index];
        }
        t[myi] = temp;
    }
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

  dim3 threadsPerBlock(1,16,1);
  dim3 numBlocks(1, 64, 1);
  naiveSpMV<<<numBlocks, threadsPerBlock>>>(deviceT, deviceB, devicePtr, deviceData, deviceIndices, n);

  float* newT = (float *) malloc(nr*sizeof(float));
  cudaMemcpy(newT, deviceT, nr*sizeof(float), cudaMemcpyDeviceToHost);

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<nr; i++) {                                                      
    for (j = ptr[i]; j<ptr[i+1]; j++) {
      t[i] = t[i] + data[j] * b[indices[j]];
    }
  }


  for (int i = 0; i < nr; i++) {
      printf("%f, %f, %f\n", newT[i], t[i], newT[i] - t[i]);
  }
  printf("%d\n", nr);

  cudaFree(deviceT);
  cudaFree(deviceIndices);
  cudaFree(devicePtr);
  cudaFree(deviceData);
  cudaFree(deviceB);


}
