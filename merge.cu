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

__global__ void mergeLarge_k(int *A, int *B, int *M, int sizeA, int sizeB, int *result) {
    // NOTICE result should be on GPU
    // only locates sufficiant array to each block's shared memory, other not in shared memory ==> pass
    // each block's resposible for [startm,endm) <= d in potential result array M
    // x in array B, and y in array A
    // To do so, we spilit Array A in a idx serie y, resp. B in x before inter kernel func
    // (sx,sy) notes startm, (ex,ey) notes endm with sx+sy = startm, ex+ey = endm
    // to simplify, we assume sizesubA + sizesubB = d <= thread num in block

    // we decide the partition uniformly
    int block_num = blockDim.x;
    int bidx = blockIdx.x;
    int sx,sy,ex,ey;
    sx = result[2*bidx];
    sy = result[2*bidx+1];
    ex = result[2*(bidx+1)];
    ey = result[2*(bidx+1)+1];
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
    if(threadIdx.x < d){
        M[threadIdx.x] = m2[threadIdx.x];
    }
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

void query(int*A,int *B, int sizeA,int sizeB, int qidx, int *coord){
    if (qidx < sizeA + sizeB) {
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
                    coord[0]=Qx;
                    coord[1]=Qy;
                    return;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }else{
        coord[0] = -1;
        coord[1] = -1;
        return;
    }
}

void wrapper_q1(int sizeA,int sizeB,int limit){
    int gridSize = GridSize;
    int blockSize = BlockSize;
    int memSize = MemSize;
    bool sorted = 1;
    float timems = 0;
    cudaEvent_t start, stop; //cuda timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
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
    int gridSize = GridSize;
    int blockSize = BlockSize;
    int memSize = MemSize;
    bool sorted = 1;
    float timems = 0;
    cudaEvent_t start, stop; //cuda timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // play with memory
    int *A,*B,*M;
    A = malloc(sizeA*sizeof(int));
    B = malloc(sizeB*sizeof(int));
    M = malloc((sizeA+sizeB)*sizeof(int));
    randomArray(A,sizeA,limit,sorted);
    randomArray(B,sizeB,limit,sorted);
    // to cuda
    // to cuda
    int *d_A, *d_B, *d_M;
    cudaMalloc((void**)&d_A, sizeA * sizeof(int));
    cudaMalloc((void**)&d_B, sizeB * sizeof(int));
    cudaMalloc((void**)&d_M, (sizeA + sizeB) * sizeof(int));
    cudaMemcpy(d_A, A, sizeA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB*sizeof(int), cudaMemcpyHostToDevice);
    //split uniformly
    int num = (sizeA+sizeB+gridSize-1)/gridSize;
    int result[2*(gridSize+1)];
    for(int i = 0;i<gridSize;i++){
        query(A,B,sizeA,sizeB,num*i,result[i*2])
    }
    result[2*gridSize]=sizeA;
    result[2*gridSize+1]=sizeB
    int *d_result;
    cudaMalloc((void**)&d_result, 2*(gridSize+1)* sizeof(int));
    cudaMemcpy(d_result, result, 2*(gridSize+1)* sizeof(int), cudaMemcpyHostToDevice)

    cudaEventRecord(start);
    mergeLarge_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,sizeA,sizeB,result);    
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

void wrapper_q5(int sizeA,int sizeB, int num, int limit){
    // d is sizeA + sizeB
    // sizeA*num
    // sizeB*num
    int gridSize = GridSize;
    int blockSize = BlockSize;
    int memSize = MemSize;
    bool sorted = 1;
    float timems = 0;
    cudaEvent_t start, stop; //cuda timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // play with memory
    int *A,*B,*M;
    int *Apoint,*Bpoint;
    A = malloc(sizeA*num*sizeof(int));
    B = malloc(sizeB*num*sizeof(int));
    M = malloc((sizeA+sizeB)*num*sizeof(int));
    Apoint = malloc((num+1)*sizeof(int));
    Bpoint = malloc((num+1)*sizeof(int));
    for(int i=0;i<=num;i++){
        if(i){
            Apoint[i]=Apoint[i-1]+sizeA;
            Bpoint[i]=Bpoint[i-1]+sizeB;
        }else{
            Apoint[0]=0;
            Bpoint[0]=0;
        }
        randomArray(A+sizeA*i,sizeA,limit,sorted);
        randomArray(B+sizeB*i,sizeB,limit,sorted);
    }    
    // to cuda
    int *d_A, *d_B, *d_M,*d_Apoint,*d_Bpoint;
    cudaMalloc((void**)&d_A, sizeA*num*sizeof(int));
    cudaMalloc((void**)&d_B, sizeB*num*sizeof(int));
    cudaMalloc((void**)&d_M, (sizeA+sizeB)*num* sizeof(int));
    cudaMalloc((void**)&d_Apoint, (num+1)*sizeof(int));
    cudaMalloc((void**)&d_Bpoint, (num+1)*sizeof(int));
    cudaMemcpy(d_A, A, num*sizeA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, num*sizeB*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Apoint, Apoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bpoint, Bpoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    mergeSmallBatch_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,Apoint,Bpoint,num,sizeA+sizeB);    
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

void wrapper_q6(int d, int num, int limit){
    int gridSize = GridSize;
    int blockSize = BlockSize;
    int memSize = MemSize;
    bool sorted = 0;
    float timems = 0;
    cudaEvent_t start, stop; //cuda timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // play with memory
    int *M,*R;
    int *Mpoint;
    M = malloc((d)*num*sizeof(int));
    Mpoint = malloc((num+1)*sizeof(int));
    for(int i=0;i<num;i++){
        if(i){
            Mpoint[i]=Mpoint[i-1]+d;
        }else{
            Mpoint[0]=0
        }
        randomArray(M+d*i,d,limit,sorted);
    }    
    // to cuda
    int *d_M, *d_R,*d_Mpoint;
    cudaMalloc((void**)&d_M, d*num* sizeof(int));
    // cudaMalloc((void**)&d_R, d*num* sizeof(int));
    cudaMalloc((void**)&d_Mpoint, (num+1)*sizeof(int));
    cudaMemcpy(d_M, M, d*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mpoint, Bpoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    //stage1
    sortSmallBatch_k<<<gridSize,blockSize,memSize>>>(d_M,Mpoint,num,d);
    cudaEventRecord(stop);

    cudaMemcpy(d_M, M, num*d*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timems, start, stop);
    pritnf("kernel spent time: %f ms\n",timems);

    cudaFree(d_M);
    cudaFree(d_R);
    cudaFree(d_Mpoint);

    if(checkOrder(M,sizeA+sizeB,compare_little)){
        printf("Array M is correct");
    }else{
        printf("Array M is incorrect");
    }
    free(M);
    free(Mpoint);
    return;
}

int main() {
    int sizeA, sizeB, limit;
    wrapper(sizeA,sizeB,limit);
    return 0;
}
