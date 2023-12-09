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

// funtion for merge 2 arrays in 1 block
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

//q1
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

    if (elemIdx<sizeA+sizeB){
        mergeInKernal(a,b,sizeA,sizeB,m,elemIdx);
    }
    M[elemIdx] = m[elemIdx];
    __syncthreads();
}

//q2
__global__ void mergeLarge_k(int *A, int *B, int *M, int sizeA, int sizeB, int *result) {
    // A: array A on GPU global memory
    // B: array B on GPU global memory
    // M: array M on GPU global memory
    // sizeA: number of elements in A total
    // sizeB: number of elements in B total
    // result: array of all partition coordinates on GPU global memorys
    // only locates sufficiant array to each block's shared memory, other not in shared memory ==> pass
    // each block's resposible for [startm,endm) <= d in potential result array M
    // x in array B, and y in array A
    // To do so, we spilit Array A in a idx serie y, resp. B in x before inter kernel func
    // (sx,sy) notes startm, (ex,ey) notes endm with sx+sy = startm, ex+ey = endm
    // to simplify, we assume sizesubA + sizesubB = d <= thread num in block

    // we decide the partition uniformly
    int sx,sy,ex,ey;
    sx = result[2*blockIdx.x];
    sy = result[2*blockIdx.x+1];
    ex = result[2*(blockIdx.x+1)];
    ey = result[2*(blockIdx.x+1)+1];
    assert(blockDim.x >= ex+ey-sx-sy); // simplicity check

    int sizesubA = ey - sy;
    int sizesubB = ex - sx;
    int elemIdx = threadIdx.x%(sizesubA+sizesubB);

    if (threadIdx.x<sizesubA+sizesubB){
        extern __shared__ int buff[]; //total: 2*(sizesubA+sizesubB)
        int *a,*b,*m;
        a = buff;
        b = a+sizeof(int)*sizesubA;
        m = b+sizeof(int)*sizesubB;
        if(elemIdx <sizesubA){
            a[elemIdx] = A[elemIdx + sy];
        }else{
            b[elemIdx-sizesubA] = B[elemIdx - sizesubA + sx];
        }
        __syncthreads(); //copy to shared memory and synchronized
        if (elemIdx<sizesubA+sizesubB){
            mergeInKernal(a,b,sizesubA,sizesubB,m,elemIdx);
        }
        M[elemIdx] = m[elemIdx];
        __syncthreads();
    }
}

//q5
__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int* Apoint, int *Bpoint, int Num int d){
    // A:  array A on GPU global memory
    // B:  array B on GPU global memory
    // M:  array M on GPU global memory
    // Apoint: array of prefix sum of sizeAi
    // Bpoint: array of prefix sum of sizeBi
    // Num: number of A/Bi pairs
    // d: limitatino on sizeA+sizeB == d
    // Apoint(resp. Bpoint) is the start idx of each Ai array in A, i.e prefixsum of sizeAi;
    // Apoint idx in [0,Num] with Apoint[Num] is end idx A exclude
    // each block have one pair of AB, suppose sizeA+sizeB = d <= 1024
    // so each block may have to deal with multiple pairs.
    extern __shared__ int buff[]; // total: 2*d*Num of pairs in block, this thread only reach out to 2*pairs(for Ai,Bi,Mi specifically)
    if(threadIdx.x/d<blockDim.x/d){ // blockDim.x >= threadIdx.x + 1 -> all entire array covered in block sutisfy this condition
        int elemIdx = threadIdx.x%d; // thread idx in its array (range in 0,d-1)
        int arrIdxBlk = (threadIdx.x-elemIdx)/d; // pair idx in its block => threadIdx.x/d any way
        int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // pair idx for all
        int sizeA=Apoint[arrIdxAll+1]-Apoint[arrIdxAll];
        int sizeB=Apoint[arrIdxAll+1]-Apoint[arrIdxAll];
        
        if (elemIdx<sizeA+sizeB) {
            int *a,*b,*m;
            a = buff + arrIdxBlk*d; //thread assign to A_arrIdxBlk
            b = buff + arrIdxBlk*d + sizeof(int)*sizeA;
            m = buff + arrIdxBlk*d + sizeof(int)*(sizeA+sizeB);
            if(elemIdx<sizeA){ // we use sizeA + size B threads
                a[elemIdx] = A[Apoint[arrIdxAll]+elemIdx];
            }else{
                b[elemIdx-sizeA] = B[Bpoint[arrIdxAll]+elemIdx-sizeA];
            }
            __syncthreads(); //copy to shared memory and synchronized
            mergeInKernal(a,b,sizeA,sizeB,m,elemIdx);
        }
        M[arrIdxAll*d+elemIdx] = m[elemIdx];
        __syncthreads();
    }
}


__global__ void sortSmallBatch_k(int *M, int *Mpoint, int Num, int d){
    // sort Mi using mergeSmallBatch? Each Block covers entiere array of M in 1st stage log(d)
    // better d >= num of threads per block
    // Mpoint is the the start idx of each Mi array in M, i.e prefixsum of sizeMi;
    extern __shared__ int buff[];//total:(4d+2)*num of Array in blk, d of buffer for sort, 2*(d+1) for prefix sum of merge tree
    if(threadIdx/d<blockDim.x/x){
        int elemIdx = threadIdx.x%d; // element idx in one array
        int arrIdxBlk = (threadIdx.x-elemIdx)/d; // local array idx in this block
        int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // global array idx
        int sizeM = Mpoint[arrIdxAll+1] - Mpoint[arrIdxAll];
        // copy to shared memory, only copy needed
        int *m, *m2, *preSumM1,*preSumM2;
        m = buff + arrIdxBlk*(4*d+2); // d elements
        m2 = m + d; // delements
        preSumM1 = m2 + d;  // d+1 prefixsum
        preSumM2 = preSumM1 + d+1; // d+1 prefixsum
        if(arrIdxBlk<blockDim.x/d){ // valid thread number
            m[arrIdxBlk*d + elemIdx] = M[Mpoint[arrIdxAll]+elemIdx];
        }
        if(arrIdxBlk<blockDim.x/d){
            preSumM1[elemIdx] = elemIdx;
            preSumM2[elemIdx] = elemIdx;
        }
        if(elemIdx==0){
            preSumM1[d]=d;
            preSumM2[d]=d;
        }
        __syncthreads();
        int hidx = elemIdx; // till ceil(log2(d)) level
        // stage1:split array to 1 element and merge sort
        // simple strategy: each time, merge 2 neighbor(greedy)
        // worst case 1/3 threads idle, promise log(d)?
        // (threadIdx.x+d-1)/d>arrIdxBlk why? TODO
        int turns = 0;
        while(turns<ceil(log2(d))){
            if(!(hidx==d-1/pow(2,turns)&&hidx%2==0)){ 
                // two case: hidx is odd -> paired
                // hidx is even but not at end of serie -> paired
                if(turns%2){// even: m1 to m2; odd: m2 to m1
                    mergeInKernal(m2+preSumM1[hidx/2*2], m2+preSumM1[hidx/2*2+1], // merge even(A) odd(B) block
                        preSumM1[hidx/2*2+1]-preSumM1[hidx/2*2], preSumM1[hidx/2*2+2]-preSumM1[hidx/2*2+1], // sizeA, sizeB
                        m+preSumM1[hidx/2*2], // save to M + with all elements before
                        (threadIdx.x-preSumM1[hidx/2*2])%(preSumM1[hidx/2*2+2]-preSumM1[hidx/2*2])); // elemen idx in sub array
                }else{
                    mergeInKernal(m+preSumM2[hidx/2*2], m2+preSumM2[hidx/2*2+1], // merge even(A) odd(B) block
                        preSumM2[hidx/2*2+1]-preSumM2[hidx/2*2], preSumM2[hidx/2*2+2]-preSumM2[hidx/2*2+1], // sizeA, sizeB
                        m2+preSumM2[hidx/2*2], // save to M + with all elements before
                        (threadIdx.x-preSumM2[hidx/2*2])%(preSumM2[hidx/2*2+2]-preSumM2[hidx/2*2])); // elemen idx in sub array
                }
            }
            if(turns%2){
                preSumM1[hidx/2]=preSumM2[hidx/2*2]
            }else{
                preSumM2[hidx/2]=preSumM1[hidx/2*2]
            }
            hidx/=2;
            turns+=1;
            __syncthreads();
        }
        if(!(threadIdx.x+d-1)/d>arrIdxBlk){
            if(turns%2){// even: result in m1, odd in m2, constrast with if in while
                M[threadIdx.x] = m2[threadIdx.x];
            }else{
                M[threadIdx.x] = m1[threadIdx.x];
            }
        }
    }
}

// TODO 
// 1. debug

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
    wrapper_q1(sizeA,sizeB,limit);
    return 0;
}
