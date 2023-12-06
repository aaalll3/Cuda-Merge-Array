#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) {checkCuda((err),__LINE__);}
inline void checkCuda(cudaError_t err, int line, bool quit = true){
    if (err != cudaSuccess) {
        fprintf(stderr,"Host CudaError: %s at %d\n", cudaGetErrorString(code), line);
        if (quit) exit(err); 
    }
}

#define CHECK_GPU_ERROR(err) {checkGPU((err),__LINE__);}
inline void checkGPU(cudaError_t err, int line, bool quit = true){
        if (err != cudaSuccess) {
            fprintf(stderr,"GPU CudaError: %s at %d\n", cudaGetErrorString(code), line);
            if (quit) assert(false); 
        }
}

const int GridSize = 128;
const int BlockSize = 1024;
const int MemSize = 2**20;

struct {
    int x;
    int y;
}Tpoint;

/// simple explaination
/*  suppose we merge A,B two sorted arrays using CUDA
    each thread in charge for one position in result arary M
    then the kernel function should: found the element corresponding to M[threadIdx.x]
    by doing so, we binary search the diagonal with a constraint x+y=threadIdx.x 
    (since there is only threadIdx.x - 1 elements in front of this thread, the result must follow this constraint)
    Binary search: input (Ki,Pi) -> Qi = (Ki+Pi)/2 -> Ki+1 = Qi or Pi+1 = Qi ->  (Ki+1,Pi+1) 
    Termination condition: at position Qi, B[Qi] > A[Qi-1] and A[Qi] > B[Qi-1], 
        i.e. all elements in front in A,B is small than A,B at Qi

        B0  B1  B2  ...  Bl-2  Bl-1 (x axis)
    A0                         P0
    A1
    A2
    ...
    An-2                 Q0
    An-1 K0
    (y axis)
*/ 
__global__ void mergeSmall_k(int *A, int *B, int *M, int sizeA, int sizeB) {
    // for this simple case, we only consider enough threads solve enough elements
    // implemented the algorithm in paper
    assert(blockIdx.x == 0); // only for one block
    assert(blockDim.x >= sizeA+sizeB); // simplicity check
    int elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int buff[]; // total:2*(sizeA+sizeB)
    int *a,*b,*m;
    a = buff;
    b = buff+sizeof(int)*sizeA;
    m = buff+sizeof(int)*(sizeA+sizeB);
    if(elemIdx<sizeA){
        a[elemIdx] = A[elemIdx];
    }else{
        b[elemIdx] = B[elemIdx];
    }
    __syncthreads(); //copy to shared memory and synchronized

    if (elemIdx < sizeA + sizeB) {
        //TODO
        int Kx, Ky, Px, Py; // K for low point in diag(close to y axis), P for high(close to x axis)
        if (elemIdx > sizeA) {
            Kx = elemIdx - sizeA;
            Ky = sizeA;
            Px = sizeA;
            Py = elemIdx - sizeA;
        } else {
            Kx = 0;
            Ky = elemIdx;
            Px = elemIdx;
            Py = 0;
        }

        while (1) {
            int offset = abs(Ky - Py) / 2;
            int Qx = Kx + offset;
            int Qy = Ky - offset;

            if (Qy >= 0 && Qx <= sizeB &&
                (Qy == sizeA || Qx == 0 || a[Qy] > b[Qx - 1])) {
                if (Qx == sizeB || Qy == 0 || (Qy < sizeA && a[Qy - 1] <= b[Qx])) {
                    if (Qy < sizeA && (Qx == sizeB || a[Qy] <= b[Qx])) {
                        m[elemIdx] = a[Qy];  // Merge in M
                    } else {
                        m[elemIdx] = b[Qx];
                    }
                    break;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }

    M[elemIdx] = m[elemIdx];
    __syncthreads();
}

__global__ void mergeSmallBlocks_k(int *A, int *B, int *M, int sizeA, int sizeB, int sx, int sy, int ex, int ey) {
    // only locates sufficiant array to each block's shared memory, other not in shared memory ==> pass
    // each block's resposible for [startm,endm) <= d in potential result array M
    // x in array B, and y in array A
    // To do so, we spilit Array A in a idx serie y, resp. B in x before inter kernel func
    // (sx,sy) notes startm, (ex,ey) notes endm with sx+sy = startm, ex+ey = endm
    // to simplify, we assume sizesubA + sizesubB = d <= thread num in block
    assert(blockDim.x >= ex+ey-sx-sy); // simplicity check

    int elemIdx = threadIdx.x;
    int sizesubA = ey - sy;
    int sizesubB = ex - sx;
    extern __shared__ int buff[]; //total: 2*(sizesubA+sizesubB)
    int *a,*b,*m;
    a = buff;
    b = buff+sizeof(int)*sizesubA;
    m = buff+sizeof(int)*(sizesubA+sizesubB);
    if(elemIdx >= 0 &&elemIdx <sizesubA){
        a[elemIdx] = A[elemIdx + sy];
    }else{
        b[elemIdx-sizesubA] = B[elemIdx - sizesubA + sx];
    }
    __syncthreads(); //copy to shared memory and synchronized

    if (elemIdx < sizesubA + sizesubB) {
        int Kx, Ky, Px, Py; // K for low point in diag, P for high
        
        if (elemIdx > sizesubA) {
            Kx = elemIdx - sizesubA;
            Ky = sizesubA;
            Px = sizesubA;
            Py = elemIdx - sizesubA;
        } else {
            Kx = 0;
            Ky = elemIdx;
            Px = elemIdx;
            Py = 0;
        }

        while (1) {
            int offset = abs(Ky - Py) / 2;
            int Qx = Kx + offset;
            int Qy = Ky - offset;

            if (Qy >= 0 && Qx <= sizesubB &&
                (Qy == sizesubA || Qx == 0 || a[Qy] > b[Qx - 1])) {
                if (Qx == sizesubB || Qy == 0 || (Qy < sizesubA && a[Qy - 1] <= b[Qx])) {
                    if (Qy < sizesubA && (Qx == sizesubB || a[Qy] <= b[Qx])) {
                        m[i] = a[Qy];  // Merge in M
                    } else {
                        m[i] = b[Qx];
                    }
                    break;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }
    M[elemIdx] = m[elemIdx];
    __syncthreads();
}

__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int* Apoint, int *Bpoint, int Num int d){
    // Apoint(resp. Bpoint) is the start idx of each Ai array in A, i.e prefixsum of sizeAi;
    // Apoint idx in [0,Num] with Apoint[Num] is end idx A exclude
    // each block have one pair of AB, suppose sizeA+sizeB = d <= 1024
    // so each block may have to deal with multiple pairs.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elemIdx = threadIdx.x%d; // thread idx in its array (range in 0,d-1)
    int arrIdxBlk = (threadIdx.x-elemIdx)/d; // pair idx in its block 
    int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // pair idx for all
    int sizeA=Apoint[arrIdxAll+1]-Apoint[arrIdxAll];
    int sizeB=Apoint[arrIdxAll+1]-Apoint[arrIdxAll];

    extern __shared__ int buff[]; // total: d*Num of pairs
    int *a,*b,*m;
    a = buff + arrIdxBlk*d; //thread assign to A_arrIdxBlk
    b = buff + arrIdxBlk*d + sizeof(int)*sizeA;
    m = buff + arrIdxBlk*d + sizeof(int)*(sizeA+sizeB);

    if(elemIdx<sizeA){ // we use sizeA + size B threads
        a[elemIdx] = A[Apoint[arrIdxAll]+elemIdx];
    }else{
        b[elemIdx-sizeA] = B[Apoint[arrIdxAll]+elemIdx-sizeA];
    }
    __syncthreads(); //copy to shared memory and synchronized
    if (elemIdx<sizeA+sizeB) {
        int Kx, Ky, Px, Py; // K for low point in diag, P for high
        if (elemIdx > sizeA) {
            Kx = elemIdx - sizeA;
            Ky = sizeA;
            Px = sizeA;
            Py = elemIdx - sizeA;
        } else {
            Kx = 0;
            Ky = elemIdx;
            Px = elemIdx;
            Py = 0;
        }
        while (1) {
            int offset = abs(Ky - Py) / 2;
            int Qx = Kx + offset;
            int Qy = Ky - offset;

            if (Qy >= 0 && Qx <= sizeB &&
                (Qy == sizeA || Qx == 0 || a[Qy] > b[Qx - 1])) {
                if (Qx == sizeB || Qy == 0 || (Qy < sizeA && a[Qy - 1] <= b[Qx])) {
                    if (Qy < sizeA && (Qx == sizeB || a[Qy] <= b[Qx])) {
                        m[elemIdx] = a[Qy];  // Merge in M
                    } else {
                        m[elemIdx] = b[Qx];
                    }
                    break;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }
    M[gbx*d+tidx] = m[tidx];
    __syncthreads();
}

// 3 funtino for mergesort
__device__ void mergeInKernal(int *sA,int *sB,int sizeA,int sizeB,int *M,int elemIdx){
    // only solve parallel merge on shared memory for [sA,eA) and [sB,eB) and store result in M
    // elemIdx notes the position this thread in charge in M
    // iterate x in B; y in A;
    int Kx,Ky,Px,Py; // K low; P high;
    if(elemIdx>sizeA){
        Kx = elemIdx - sizeA;
        ky = sizeA;
        Px = sizeA;
        Py = elemIdx - sizeA;
    }else{
        Kx = 0;
        Ky = elemIdx;
        Px = elemIdx;
        Py = 0;
    }
    while(1){
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sizeB &&
            (Qy == sizeA || Qx == 0 || sA[Qy] > sB[Qx - 1])) {
            if (Qx == sizeB || Qy == 0 || A[Qy - 1] <= sB[Qx]) {
                if (Qy < sizeA && (Qx == sizeB || sA[Qy] <= sB[Qx])) {
                    M[elemIdx] = sA[Qy];
                } else {
                    M[elemIdx] = sB[Qx];
                }
                break;
            } else { 
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
    return;
}

__global__ void sortSmallBatch_k(int *M, int *Mpoint, int Num, int d){
    // 1.sort Mi using mergeSmallBatch? Each Block covers entiere array of M in 1st stage log(d)
    // 2.recursively merge 2 large array -> mergeSmallBlocks log(N)
    // better d >= num of threads per block
    // Mpoint is the the start idx of each Mi array in M, i.e prefixsum of sizeMi;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global idx
    int elemIdx = threadIdx.x%d; // element idx in one array
    int arrIdxBlk = (threadIdx.x-elemIdx)/d; // local array idx in this block
    int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // global array idx
    int sizeM = Mpoint[arrIdxAll+1] - Mpoint[arrIdxAll];
    // int sizeM1 = 

    // copy to shared memory, only copy needed
    // m -- M
    extern __shared__ int buff[];
    int *m, *m2;
    m = buff + arrIdxBlk*d;
    m2 = buff + arrIdxBlk*d + sizeof(int)*;
    if((threadIdx.x+d-1)/d>arrIdxBlk){
        m[arrIdxBlk*d + elemIdx] = M[Mpoint[arrIdxAll]+elemIdx];
    }
    __syncthreads();
    int subfix = tidx; // till ceil(log2(d)) level
    int cover = 1;
    // stage1:split array to 1 element and merge sort
    // simple strategy: each time, merge 2 neighbor(greedy)
    // worst case 1/3 threads idle, promise log(d)?
    while((threadIdx.x+d-1)/d>arrIdxBlk){
        if(!(subfix=d-1&&subfix%2==0)){ 
            mergeInKernal(m+cover*(subfix/2)*2,m+cover*(subfix/2)*2+cover,cover,cover,m2,threadIdx.x%2);
        }
        subfix/=2;
        cover*=2;
    }
}

__global__ void mergeLargeBlocks(int *M, int *Mpoint, int Num, int d){
    //stage2:merge large array from Mi to Mi/2 and reset Mpoint
}
    
// TODO 
// 1. wrapper and random generator of arrays
// 2. complete sortSmallBatch_k
// 3. debug

typedef int (*Compare)(const void*,const void*);

int compare_little(const void* a, const void *b){
    return (*(const int*)a - *(const int*)b);
}

int compare_large(const void* a, const void *b){
    return (*(const int*)b - *(const int*)a);
}

void randomArray(int *arr, int number, int limit, bool sorted){
    // malloc(buff,sizeof(int)*number);
    for(int i = 0;i<number;i++){
        arr[i]=rand()%limit;
    }
    if (sorted){
        qsort(arr, number, sizeof(int),compare_little);
    }
    return;
}

bool checkOrder(int *arr, int size, Compare cmp){
    for(int i = 0; i < size-1; i++){
        if (!cmp(arr[i],arr[i+1])){
            return false;
        }
    }
    return true;
}

void wrapper_q1(int sizeA,int sizeB,int limit){
    int gridSize = GridSize;
    int blockSize = BlockSize;
    int memSize = MemSize;
    bool sorted = 1;
    float timems = 0;
    cudaEvent_t start, stop; //cuda timer
    cudaEventCreate(&start);
    // play with memory
    int *A,*B,*M;
    A = malloc(sizeA*sizeof(int));
    B = malloc(sizeB*sizeof(int));
    M = malloc((sizeA+sizeB)*sizeof(int));
    randomArray(A,sizeA,limit,sorted);
    randomArray(B,sizeB,limit,sorted);
    // to cuda
    int *d_A, *d_B, *d_M;
    cudaMalloc((void**)&d_A, sizeA * sizeof(int));
    cudaMalloc((void**)&d_B, sizeB * sizeof(int));
    cudaMalloc((void**)&d_M, (sizeA + sizeB) * sizeof(int));
    cudaMemcpy(d_A, A, sizeA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    mergeSmall_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,sizeA,sizeB);    
    cudaEventRecord(stop);

    cudaMemcpy(d_M, M, (sizeA+sizeB)*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timems, start, stop);
    pritnf("kernel spent time: %f ms\n",timems);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

    if(checkOrder(M,sizeA+sizeB,compare_little)){
        printf("Array M is correct");
    }else{
        printf("Array M is incorrect");
    }
    free(A);
    free(B);
    free(M);
    return;
}

void wrapper_q2(int sizeA,int sizeB,int limit){
    
}

void wrapper_q5(int sizeA,int sizeB,int limit){
    
}

void wrapper(int sizeA,int sizeB,int limit){
    int *buff;
    int maxsize = sizeA>sizeB?sizeA:sizeB;
    int blockSize = 256;
    int gridSize = (sizeA + sizeB + blockSize - 1) / blockSize;

    // Allocate memory on the GPU
    int *d_A, *d_B, *d_M;
    cudaMalloc((void**)&d_A, sizeA * sizeof(int));
    cudaMalloc((void**)&d_B, sizeB * sizeof(int));
    cudaMalloc((void**)&d_M, (sizeA + sizeB) * sizeof(int));
    // Copy data from CPU to GPU
    srand(time(NULL));
    malloc(buff,sizeof(int)*sizeA);
    for(int i = 0;i<maxsize;i++){
        buff[i]=rand()%limit;
    }
    cudaMemcpy(d_A, buff, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0;i<sizeB;i++){
        buff[i]=rand()%limit;
    }
    cudaMemcpy(d_B, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);
    
    mergeSmall_k<<<gridSize, blockSize>>>(d_A, d_B, d_M, sizeA, sizeB);

    int result[sizeA + sizeB];
    cudaMemcpy(result, d_M, (sizeA + sizeB) * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

    // Print the merged array
    printf("Merged Array: ");
    for (int i = 0; i < sizeA + sizeB; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");
}


int main() {
    int sizeA, sizeB, limit;
    wrapper(sizeA,sizeB,limit);
    return 0;
}
