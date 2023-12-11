#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <ctime>

#define CHECK_CUDA_ERROR(err) {checkCuda((err),__LINE__);}
inline void checkCuda(cudaError_t err, int line, bool quit = true){
    if (err != cudaSuccess) {
        fprintf(stderr,"Host CudaError: %s at %d\n", cudaGetErrorString(err), line);
        if (quit) exit(err); 
    }
}

#define CHECK_GPU_ERROR(err) {checkGPU((err),__LINE__);}
__device__ void checkGPU(cudaError_t err, int line, bool quit = true){
        if (err != cudaSuccess) {
            printf("GPU CudaError: %s at %d\n", cudaGetErrorString(err), line);
            if (quit) assert(false); 
        }
}

const int GridSize = 128;
const int BlockSize = 1024;
const int MemSize = pow(2,5);
const int RandMax = pow(2,30);

#define DEBUGGING 0

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

__global__ void stupid_k() {
    printf("Im stupid %d#%dth\n",blockIdx.x,threadIdx.x);
}

__device__ void mergeInKernal(int *sA,int *sB,int sizeA,int sizeB,int *sM,int elemIdx){
    // sA: array of A in shared memory
    // sB: array of B in shared memory
    // sizeA: number of elements in sA
    // sizeB: number of elements in sB
    // sM: array of M results in shared memory
    // elemIdx: idx of element the thread in charge
    // only solve parallel merge on shared memory for [sA,eA) and [sB,eB) and store result in sM
    // elemIdx notes the position this thread in charge in sM
    // iterate x in B; y in A;
    int Kx,Ky,Py; // K low; P high; Px is removed

    if(elemIdx>sizeA){
        Kx = elemIdx - sizeA;
        Ky = sizeA;
        // Px = sizeA;
        Py = elemIdx - sizeA;
    }else{
        Kx = 0;
        Ky = elemIdx;
        // Px = elemIdx;
        Py = 0;
    }
    while(1){
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sizeB &&
            (Qy == sizeA || Qx == 0 || sA[Qy] > sB[Qx - 1])) { 
            
            if (Qx == sizeB || Qy == 0 || sA[Qy - 1] <= sB[Qx]) {
                if (Qy < sizeA && (Qx == sizeB || sA[Qy] <= sB[Qx])) {
                    sM[elemIdx] = sA[Qy];
                } else {
                    sM[elemIdx] = sB[Qx];
                }
                break;
            } else { //get higher
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else { //get lower
            // Px = Qx - 1;
            Py = Qy + 1;
        }
    }
    return;
}

//q1 
__global__ void mergeSmall_k(int *A, int *B, int *M, int sizeA, int sizeB) {
    // A: array A on GPU global memory
    // B: array B on GPU global memory
    // M: array M on GPU global memory
    // sizeA: number of elements in A total
    // sizeB: number of elements in B total
    // for this simple case, we only consider enough threads solve enough elements
    // implemented the algorithm in paper
    // if(DEBUGGING)printf("Im %dth\n",threadIdx.x);
    assert(blockIdx.x == 0); // only for one block
    assert(blockDim.x >= sizeA+sizeB); // simplicity check
    extern __shared__ int buff[]; // total:2*(sizeA+sizeB)
    if(threadIdx.x<sizeA+sizeB){
        int elemIdx = threadIdx.x;
        int *a,*b,*m;
        a = buff;
        b = a+sizeA;
        m = b+sizeB;
        if(elemIdx<sizeA){
            a[elemIdx] = A[elemIdx];
        }else{
            b[elemIdx-sizeA] = B[elemIdx-sizeA];
        }
        if(elemIdx==0&&DEBUGGING){
            printf("im th%d\n",elemIdx);
            printf("a shared:\n");
            for(int i =0;i<sizeA;i++)printf("%d, ",a[i]);
            printf("\n");
            printf("b shared:\n");
            for(int i =0;i<sizeB;i++)printf("%d, ",b[i]);
            printf("\n");
        }
        __syncthreads(); //copy to shared memory and synchronized    
        mergeInKernal(a,b,sizeA,sizeB,m,elemIdx);
        M[elemIdx] = m[elemIdx];
        __syncthreads();
        if(elemIdx==0&&DEBUGGING){
            printf("im th%d",elemIdx);
            printf("m shared:\n");
            for(int i =0;i<sizeA+sizeB;i++)printf("%d, ",m[i]);
            printf("\n");
        }
    }
}

//q2
__global__ void mergeLarge_k(int *A, int *B, int *M, int d, int *partition) {
    // A: array A on GPU global memory
    // B: array B on GPU global memory
    // M: array M on GPU global memory
    // sizeA: number of elements in A total
    // sizeB: number of elements in B total
    // partition: array of all partition coordinates on GPU global memorys
    // only load sufficiant array to each block's shared memory, other not in shared memory ==> pass
    // each block is resposible for [startm,endm) <= d in potential result array M
    // x in array B, and y in array A
    // To do so, we spilit Array A in a idx serie y, resp. B in x before inter kernel func
    // (sx,sy) notes startm, (ex,ey) notes endm with sx+sy = startm, ex+ey = endm
    // to simplify, we assume sizesubA + sizesubB = d <= thread num in block

    // we decide the partition uniformly
    // have to assign just enough blocks
    // if(DEBUGGING)printf("Im %d#%dth\n",blockIdx.x,threadIdx.x);
    extern __shared__ int buff[]; //total: 2*(sizesubA+sizesubB) == 2*d
    if (threadIdx.x<d){
        int sx,sy,ex,ey;
        sx = partition[2*blockIdx.x];
        sy = partition[2*blockIdx.x+1];
        ex = partition[2*(blockIdx.x+1)];
        ey = partition[2*(blockIdx.x+1)+1];
        // assert(blockDim.x >= ex+ey-sx-sy); // simplicity check: thread num >= d
    
        int sizesubA = ey - sy;
        int sizesubB = ex - sx;
        int elemIdx = threadIdx.x%(sizesubA+sizesubB);
        int *a,*b,*m;
        a = buff;
        b = a+sizesubA;
        m = b+sizesubB;
        if(elemIdx <sizesubA){
            a[elemIdx] = A[elemIdx + sy];
        }else{
            b[elemIdx-sizesubA] = B[elemIdx - sizesubA + sx];
        }
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("a shared #%d:\n",sizesubA);
            for(int i =0;i<sizesubA;i++)printf("%d, ",a[i]);
            printf("\n");
            printf("b shared #%d:\n",sizesubB);
            for(int i =0;i<sizesubB;i++)printf("%d, ",b[i]);
            printf("\n");
        }
        __syncthreads(); //copy to shared memory and synchronized
        mergeInKernal(a,b,sizesubA,sizesubB,m,elemIdx);
        M[sx+sy+elemIdx] = m[elemIdx];
        __syncthreads();
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("m shared:\n");
            for(int i =0;i<sizesubA+sizesubB;i++)printf("%d, ",m[i]);
            printf("\n");
        }
    }
}

//q5
__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int* Apoint, int *Bpoint, int Num, int d){
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
    // if(DEBUGGING)printf("Im %d#%dth\n",blockIdx.x,threadIdx.x);
    assert(d<=blockDim.x);
    assert(Num<=gridDim.x*(blockDim.x/d));
    if(blockIdx.x*(blockDim.x/d)+threadIdx.x/d>=Num){
        printf("out of bound return\n");
        return;
    }
    extern __shared__ int buff[]; // total: 2*d*Num of pairs in block, this thread only reach out to 2*pairs(for Ai,Bi,Mi specifically)
    
    
    if(threadIdx.x/d < blockDim.x/d){ // blockDim.x >= threadIdx.x + 1 -> all entire array covered in block sutisfy this condition
        
        int elemIdx = threadIdx.x%d; // thread idx in its array (range in 0,d-1)
        
        int arrIdxBlk = (threadIdx.x - elemIdx) / d; // pair idx in its block => threadIdx.x/d any way
        
        int arrIdxAll = arrIdxBlk + blockIdx.x * (blockDim.x / d); // pair idx for all
        
        int sizeA=Apoint[arrIdxAll+1]-Apoint[arrIdxAll];
        
        int sizeB=Bpoint[arrIdxAll+1]-Bpoint[arrIdxAll];

        
        
        //copy to shared memory and synchronized
        int *a,*b,*m;
        a = buff + arrIdxBlk*2*d; //thread assign to A_arrIdxBlk
        b = a + sizeA;
        m = b + sizeB;
        
        if(elemIdx<sizeA){ // we use sizeA + size B threads
            a[elemIdx] = A[Apoint[arrIdxAll]+elemIdx];
        }else{
            b[elemIdx-sizeA] = B[Bpoint[arrIdxAll]+elemIdx-sizeA];

        }
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("a shared:\n");
            for(int i =0;i<sizeA;i++)printf("%d, ",a[i]);
            printf("\n");
            printf("b shared:\n");
            for(int i =0;i<sizeB;i++)printf("%d, ",b[i]);
            printf("\n");
        }
        __syncthreads(); 
        mergeInKernal(a, b, sizeA, sizeB, m, elemIdx);
        M[arrIdxAll*d+elemIdx] = m[elemIdx];
        __syncthreads();
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("m shared:\n");
            for(int i =0;i<sizeA+sizeB;i++)printf("%d, ",m[i]);
            printf("\n");
        }
    }
}

//q6
__global__ void sortSmallBatch_k(int *M, int *Mpoint, int Num, int d,int height){
    // Merge sort Mi = d <= blockDim
    // Each Block covers entire array of Mi (may be multiple)
    // Mpoint is the the start idx of each Mi array in M, i.e prefixsum of sizeMi
    // if(DEBUGGING)printf("Im %d#%dth\n",blockIdx.x,threadIdx.x);
    assert(d<=blockDim.x);
    assert(Num<=gridDim.x*(blockDim.x/d));
    if(blockIdx.x*(blockDim.x/d)+threadIdx.x/d>=Num)return;
    extern __shared__ int buff[];//total:(4d+2)*num of Array in blk, d of buffer for sort, 2*(d+1) for prefix sum of merge tree
    if(threadIdx.x/d<blockDim.x/d){
        int elemIdx = threadIdx.x%d; // element idx in one array
        int arrIdxBlk = (threadIdx.x-elemIdx)/d; // local array idx in this block
        int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // global array idx
        // copy to shared memory, only copy needed
        int *m1, *m2, *preSumM1,*preSumM2;
        m1 = buff + arrIdxBlk*(4*d+2); // d elements
        m2 = m1 + d; // d elements
        preSumM1 = m2 + d;  // d+1 prefixsum
        preSumM2 = preSumM1 + d+1; // d+1 prefixsum
        m1[elemIdx] = M[Mpoint[arrIdxAll]+elemIdx];//copy to shared memory
        preSumM1[elemIdx] = elemIdx;
        preSumM2[elemIdx] = elemIdx;
        if(elemIdx==0){
            preSumM1[d]=d;
            preSumM2[d]=d;
        }
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("m1 shared:\n");
            for(int i =0;i<d;i++)printf("%d, ",m1[i]);
            printf("\n");
        }
        __syncthreads();
        // split array to 1 element and merge sort
        // simple strategy: each time, merge 2 neighbor(greedy)
        // worst case 1/3 threads idle, promise ceil(log(d))
        int hidx = elemIdx; // idx at current height
        int turns = 0; // turns of iteration also height in the tree
        int last = d-1; // notes last block
        int even,odd,next; // easy to read
        // CHECK_GPU_ERROR(__log2f((float)d));
        if(DEBUGGING&&elemIdx==0)printf("d is %d\n",d);
        if(DEBUGGING&&elemIdx==0)printf("turn max %d\n",height);
        // cudaError_t cudaError = cudaGetLastError();
        // CHECK_GPU_ERROR(cudaError);
        while(turns<height){
            even = hidx/2*2;
            odd = even+1;
            next = odd+1;
            // if(turns%2){
            //     if(DEBUGGING)printf("Im %dblk %dth-%dhidx-%delem @ %dturn\n",blockIdx.x,threadIdx.x,hidx,elemIdx-preSumM2[even],turns);
            // }else{
            //     if(DEBUGGING)printf("Im %dblk %dth-%dhidx-%delem @ %dturn\n",blockIdx.x,threadIdx.x,hidx,elemIdx-preSumM1[even],turns);
            // }
            if(turns%2){
                if(!(hidx==last&&hidx%2==0)){ 
                    // two case: hidx is odd -> paired
                    // hidx is even but not at end of serie -> paired
                    mergeInKernal(m2+preSumM2[even], m2+preSumM2[odd], // merge even(A) odd(B) block
                        preSumM2[odd]-preSumM2[even], preSumM2[next]-preSumM2[odd], // sizeA, sizeB
                        m1+preSumM2[even], // save to M + with all elements before
                        elemIdx-preSumM2[even]); // elemen idx in sub array
                }
                if(elemIdx-preSumM2[even]==0){//avoid conflict access
                    preSumM1[hidx/2]=preSumM2[even];
                }
                if(elemIdx == 0){
                    preSumM1[last/2+1]=preSumM2[last+1];// must put the end to last/2+1 in next round
                }
            }else{
                if(!(hidx==last&&hidx%2==0)){ 
                    mergeInKernal(m1+preSumM1[even], m1+preSumM1[odd], // merge even(A) odd(B) block
                        preSumM1[odd]-preSumM1[even], preSumM1[next]-preSumM1[odd], // sizeA, sizeB
                        m2+preSumM1[even], // save to M + with all elements before
                        elemIdx-preSumM1[even]); // elemen idx in sub array
                }
                if(elemIdx-preSumM1[even]==0){
                    preSumM2[hidx/2]=preSumM1[even];
                }
                if(elemIdx == 0){
                    preSumM2[last/2+1]=preSumM1[last+1];
                }
            }
            hidx/=2;
            last/=2;
            turns+=1;
            __syncthreads();
        }
        if(turns%2){// odd: result in m2
            M[Mpoint[arrIdxAll]+elemIdx] = m2[elemIdx];
        }else{// even: result in m1 
            M[Mpoint[arrIdxAll]+elemIdx] = m1[elemIdx];
        }
        __syncthreads();
        if(elemIdx==0&&DEBUGGING){
            printf("Im %dblk %dth %delem\n",blockIdx.x,threadIdx.x,elemIdx);
            printf("m1 shared:\n");
            for(int i =0;i<d;i++)printf("%d, ",m1[i]);
            printf("\n");
            printf("m2 shared:\n");
            for(int i =0;i<d;i++)printf("%d, ",m2[i]);
            printf("\n");
        }
    }
}

// TODO 
// 1. debug

typedef int (*Compare)(const void*, const void*);

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
        if (cmp((const void*)(arr+i),(const void*)(arr+i+1))>0){
            if(DEBUGGING)printf("err M[%d]:%d  M[%d]:%d\n",i,arr[i],i+1,arr[i+1]);
            return false;
        }
    }
    return true;
}

void query(int*A,int *B, int sizeA,int sizeB, int qidx, int *coord){
    if (qidx < sizeA + sizeB) {
        int Kx, Ky, Py; // K for low point in diag(close to y axis), P for high(close to x axis); Px is removed;
        if (qidx > sizeA) {
            Kx = qidx - sizeA;
            Ky = sizeA;
            // Px = sizeA;
            Py = qidx - sizeA;
        } else {
            Kx = 0;
            Ky = qidx;
            // Px = qidx;
            Py = 0;
        }

        while (1) {
            int offset = abs(Ky - Py) / 2;
            int Qx = Kx + offset;
            int Qy = Ky - offset;

            if (Qy >= 0 && Qx <= sizeB &&
                (Qy == sizeA || Qx == 0 || A[Qy] > B[Qx - 1])) {
                if (Qx == sizeB || Qy == 0 || (Qy < sizeA && A[Qy - 1] <= B[Qx])) {
                    coord[0]=Qx;
                    coord[1]=Qy;
                    return;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                // Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }else{
        coord[0] = -1;
        coord[1] = -1;
        return;
    }
}

void sequential_merge(int*A, int*B, int*M, int sizeA, int sizeB){

    int i = 0; // Index for array A
    int j = 0; // Index for array B
    int k = 0; // Index for array M (result)

    // Merge elements from A and B into M
    while (i < sizeA && j < sizeB) {
        if (A[i] <= B[j]) {
            M[k++] = A[i++];
        } else {
            M[k++] = B[j++];
        }
    }

    // Copy the remaining elements from A, if any
    while (i < sizeA) {
        M[k++] = A[i++];
    }

    // Copy the remaining elements from B, if any
    while (j < sizeB) {
        M[k++] = B[j++];
    }
}

void mergeSort(int M[], int size) {
    if (size <= 1) {
        return; // Array is already sorted (base case)
    }

    int mid = size / 2;

    // Split the array into two halves
    int* left = new int[mid];
    int* right = new int[size - mid];

    std::copy(M, M + mid, left);
    std::copy(M + mid, M + size, right);

    // Recursively sort the two halves
    mergeSort(left, mid);
    mergeSort(right, size - mid);

    // Merge the sorted halves
    sequential_merge(left, right, M, mid, size - mid);

    // Free dynamically allocated memory
    delete[] left;
    delete[] right;
}

void wrapper_q1(int sizeA, int sizeB, int gridSize=GridSize, int blockSize=BlockSize, int memSize=MemSize, int limit=RandMax, bool sorted = true){
    /*  allocation stragies
        must:
            blockDim >= sizeA + sizeB = d
            gridDim = 1
        prefer:
            blockDim = sizeA + sizeB = d
        test:
            blockDim = d, d various
    */
    assert(gridSize==1);
    assert(blockSize>=sizeA+sizeB);

    // generate random array on CPU
    int *A,*B,*M,*M_sequ;
    A = (int*)malloc(sizeA*sizeof(int));
    B = (int*)malloc(sizeB*sizeof(int));
    M = (int*)malloc((sizeA+sizeB)*sizeof(int));
    M_sequ = (int*)malloc((sizeA+sizeB)*sizeof(int));

    randomArray(A,sizeA,limit,sorted);
    randomArray(B,sizeB,limit,sorted);

    // compute in cpu
    double t = clock();
    sequential_merge(A, B, M_sequ, sizeA, sizeB);
    t = clock() - t;
    printf("CPU Execution Time: %f ms \n", t);

    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("wrapper check\n");
        printf("A:\n");
        for(int i =0;i<sizeA;i++)printf("%d, ",A[i]);
        printf("\n");
        printf("B:\n");
        for(int i =0;i<sizeB;i++)printf("%d, ",B[i]);
        printf("\n");
        printf("<<<DEBUGGING\n");
    }
    //cuda timer
    float timems = 0;
    cudaEvent_t start, stop; 
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    // copy to cuda
    int *d_A, *d_B, *d_M;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, sizeA*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeB*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_M, (sizeA+sizeB)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
    // compute in cuda
    CHECK_CUDA_ERROR(cudaEventRecord(start,0));
    assert(gridSize==1);
    assert(blockSize>=sizeA+sizeB);
    mergeSmall_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,sizeA,sizeB);    
    CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
    // copy results from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(M, d_M, (sizeA+sizeB)*sizeof(int), cudaMemcpyDeviceToHost));
    // timer
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timems, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("CUDA Execution Time: %f ms\n",timems);



    
    
    // free
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_M));
    // check result
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("M\n");
        for(int i =0;i<sizeA+sizeB;i++)printf("%d, ",M[i]);
        printf("\n");
        printf("<<<DEBUGGING\n");
    }
    if(checkOrder(M,sizeA+sizeB,compare_little)){
        printf("Array M is correct\n");
    }else{
        printf("Array M is incorrect\n");
    }
    // free
    free(A);
    free(B);
    free(M);
    free(M_sequ);
    return;
}

void wrapper_q2(int sizeA,int sizeB, int gridSize=GridSize, int blockSize=BlockSize, int memSize=MemSize, int limit=RandMax,  bool sorted = true){
    /*  allocation stragies
        must:
            blockDim <= 1024 < sizeA + sizeB 
            gridDim = ceil((sizeA + sizeB)/blockDim)
            gridDim > sizeA + sizeB
        test:
            blockDim various
    */
    // assert((sizeA+sizeB+gridSize-1)/gridSize<=1024); // not too few block
    // assert(gridSize>sizeA+sizeB); // not too much block
    // assert((sizeA+sizeB+blockSize-1)/blockSize==gridSize); // make sure simple solution
    //cuda timer
    float timems = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    // generate random array on CPU
    int *A,*B,*M,*M_sequ;
    A = (int*)malloc(sizeA*sizeof(int));
    B = (int*)malloc(sizeB*sizeof(int));
    M = (int*)malloc((sizeA+sizeB)*sizeof(int));
    M_sequ = (int*)malloc((sizeA+sizeB)*sizeof(int));
    randomArray(A,sizeA,limit,sorted);
    randomArray(B,sizeB,limit,sorted);

    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("wrapper check\n");
        printf("A:\n");
        for(int i =0;i<sizeA;i++)printf("%d, ",A[i]);
        printf("\n");
        printf("B:\n");
        for(int i =0;i<sizeB;i++)printf("%d, ",B[i]);
        printf("\n");
        printf("<<<DEBUGGING\n");
    }

    // compute in cpu
    double t = clock();
    sequential_merge(A, B, M_sequ, sizeA, sizeB);
    t = clock() - t;
    printf("CPU Execution Time: %f ms \n", t);

    // copy to cuda
    int *d_A, *d_B, *d_M;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, sizeA*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeB*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_M, (sizeA+sizeB)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
    //split uniformly
    int length = (sizeA+sizeB+gridSize-1)/gridSize; // upper round, only last sub array not full size
    int partition[2*(gridSize+1)];
    for(int i = 0;i<gridSize;i++){
        query(A,B,sizeA,sizeB,length*i,partition+i*2);
    }
    partition[2*gridSize]=sizeA;
    partition[2*gridSize+1]=sizeB;
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("partition check\n");
        printf("partition:\n");
        for(int i =0;i<gridSize+1;i++)printf("(%d,%d)",partition[2*i],partition[2*i+1]);
        printf("\n");
        printf("<<<DEBUGGING\n");
    }
    int *d_partition;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_partition, 2*(gridSize+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_partition, partition, 2*(gridSize+1)*sizeof(int), cudaMemcpyHostToDevice));
    // compute
    CHECK_CUDA_ERROR(cudaEventRecord(start,0));
    mergeLarge_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,sizeA+sizeB,d_partition);    
    CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
    // copy results from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(M, d_M, (sizeA+sizeB)*sizeof(int), cudaMemcpyDeviceToHost));
    // timer
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timems, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("CUDA Execution Time: %f ms\n",timems);

    // free
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_M));
    // check result
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("M\n");
        for(int i =0;i<sizeA+sizeB;i++)printf("%d, ",M[i]);
        printf("\n");
        printf("<<<DEBUGGING\n");
    }
    if(checkOrder(M,sizeA+sizeB,compare_little)){
        printf("Array M is correct\n");
    }else{
        printf("Array M is incorrect\n");
    }
    // free
    free(A);
    free(B);
    free(M);
    return;
}

void wrapper_q5(int sizeA,int sizeB, int num, int gridSize=GridSize, int blockSize=BlockSize, int memSize=MemSize, int limit=RandMax, bool sorted = true){
    /*  allocation stragies
        must:
            blockDim >= sizeA + sizeB = d
            gridDim >= ceil(num/(blockDim/d))
        prefer:
            gridDim = ceil(num/(blockDim/d))
        test:
            blockDim >= k*(sizeA + sizeB), k various from 1
    */
    assert(blockSize>=sizeA+sizeB);
    assert(gridSize >= (num+(blockSize/(sizeA+sizeB))-1)/(blockSize/(sizeA+sizeB)));
    //cuda timer
    float timems = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    // generate random array on CPU
    int *A,*B,*M, *M_sequ;
    int *Apoint,*Bpoint;
    A = (int*)malloc(sizeA*num*sizeof(int));
    B = (int*)malloc(sizeB*num*sizeof(int));
    M = (int*)malloc((sizeA+sizeB)*num*sizeof(int));
    M_sequ = (int*)malloc((sizeA+sizeB)*num*sizeof(int));

    Apoint = (int*)malloc((num+1)*sizeof(int));
    Bpoint = (int*)malloc((num+1)*sizeof(int));
    for(int i=0;i<num;i++){
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

    Apoint[num]=Apoint[num-1]+sizeA;
    Bpoint[num]=Bpoint[num-1]+sizeB;

    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("wrapper check\n");
        for(int i=0;i<num;i++){
            printf("A%d:\n",i);
            for(int j=0;j<sizeA;j++)printf("%d, ",A[i*sizeA+j]);
            printf("\n");
        }
        for(int i=0;i<num;i++){
            printf("B%d:\n",i);
            for(int j=0;j<sizeB;j++)printf("%d, ",B[i*sizeB+j]);
            printf("\n");
        }
        printf("<<<DEBUGGING\n");
    }

    // compute in cpu
    double t = clock();
    int d = sizeA + sizeB;
    for(int i = 0; i<num; i++){
        sequential_merge(A+Apoint[i], B+Bpoint[i], M_sequ+i*d, sizeA, sizeB);
    }
    t = clock() - t;
    printf("CPU Execution Time: %f ms \n", t);


    // copy to cuda
    int *d_A, *d_B, *d_M, *d_Apoint, *d_Bpoint;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, sizeA*num*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeB*num*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_M, (sizeA+sizeB)*num* sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Apoint, (num+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Bpoint, (num+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, sizeA*num*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, sizeB*num*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Apoint, Apoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Bpoint, Bpoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice));
    //compute
    CHECK_CUDA_ERROR(cudaEventRecord(start,0));
    // stupid_k<<<gridSize,blockSize>>>();
    mergeSmallBatch_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,d_Apoint,d_Bpoint,num,sizeA+sizeB);    
    CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
    // copy results from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(M, d_M, (sizeA+sizeB)*num*sizeof(int), cudaMemcpyDeviceToHost));
    // timer
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timems, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("CUDA Excution Time: %f ms\n",timems);
    // free
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_M));
    CHECK_CUDA_ERROR(cudaFree(d_Apoint));
    CHECK_CUDA_ERROR(cudaFree(d_Bpoint));
    // check result
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        for(int i=0;i<num;i++){
            printf("M%d:\n",i);
            for(int j =0;j<sizeA+sizeB;j++)printf("%d, ",M[j]);
            printf("\n");
        }
        printf("<<<DEBUGGING\n");
    }

    for(int i=0;i<num;i++){
        if(checkOrder(M+(sizeA+sizeB)*i,sizeA+sizeB,compare_little)){
            printf("Array M%d is correct\n",i);
        }else{
            printf("Array M%d is incorrect\n",i);

        }
    }
    // free
    free(A);
    free(B);
    free(M);
    free(M_sequ);
    free(Apoint);
    free(Bpoint);
    return;
}

void wrapper_q6(int d, int num, int gridSize=GridSize, int blockSize=BlockSize, int memSize=MemSize, int limit=RandMax, bool sorted = false){
    /*  allocation stragies
        must:
            blockDim >= d
            gridDim >= ceil(num/(blockDim/d))
        prefer:
            gridDim = ceil(num/(blockDim/d))
        test:
            blockDim >= k*d, k various from 1
    */   
    assert(blockSize>=d);
    assert(gridSize >= (num+(blockSize/d)-1)/(blockSize/d));
    int height = ceil(log2(d));
    //cuda timer
    float timems = 0;
    cudaEvent_t start, stop; 
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    // generate random array on CPU
    int *M, *M_sequ;
    int *Mpoint;
    M = (int*)malloc(d*num*sizeof(int));
    M_sequ = (int*)malloc(d*num*sizeof(int));
    Mpoint = (int*)malloc((num+1)*sizeof(int));
    for(int i=0;i<num;i++){
        if(i){
            Mpoint[i]=Mpoint[i-1]+d;
        }else{
            Mpoint[0]=0;
        }
        randomArray(M+d*i,d,limit,0);
        randomArray(M_sequ+d*i,d,limit,0);
        
    }
    Mpoint[num]=Mpoint[num-1]+d;
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("wrapper check\n");
        for(int i =0;i<num;i++){
            printf("M%d origin:\n",i);
            for(int j=0;j<d;j++){
                printf("%d, ",M[i*d+j]);
            }
            printf("\n");
        }
        printf("<<<DEBUGGING\n");
    }

    // compute in cpu
    double t = clock();
    for(int i = 0; i<num; i++){
        mergeSort(M_sequ+i*d, d);
    }
    t = clock() - t;
    printf("CPU Execution Time: %f ms \n", t);

    // copy to cuda
    int *d_M,*d_Mpoint;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_M, d*num*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Mpoint, (num+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_M, M, d*num*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Mpoint, Mpoint, (num+1)*sizeof(int), cudaMemcpyHostToDevice));
    //compute
    CHECK_CUDA_ERROR(cudaEventRecord(start,0));
    sortSmallBatch_k<<<gridSize,blockSize,memSize>>>(d_M,d_Mpoint,num,d,height);
    CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
    // copy results from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(M, d_M, d*num*sizeof(int), cudaMemcpyDeviceToHost));
    // timer
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timems, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("CUDA Execution Time: %f ms\n",timems);
    // free
    CHECK_CUDA_ERROR(cudaFree(d_M));
    CHECK_CUDA_ERROR(cudaFree(d_Mpoint));
    // check result
    if(DEBUGGING){
        printf(">>>DEBUGGING\n");
        printf("result check\n");
        for(int i =0;i<num;i++){
            printf("M%d new:\n",i);
            for(int j=0;j<d;j++){
                printf("%d, ",M[i*d+j]);
            }
            printf("\n");
        }
        printf("<<<DEBUGGING\n");
    }
    for(int i = 0;i<num;i++){
        if(checkOrder(M+d*i,d,compare_little)){
            printf("Array M%d is correct\n",i);
        }else{
            printf("Array M%d is incorrect\n",i);
        }
    }
    // free
    free(M);
    free(Mpoint);
    return;
}

void debug_q1(){
    int sizeA=400;
    int sizeB=400;
    int d=sizeA+sizeB;
    // int num=4;
    int gridSize =1; 
    int blockSize =sizeA+sizeB;
    int memSize =d*sizeof(int)*2;
    int limit = 1000;

    wrapper_q1(sizeA,sizeB,gridSize,blockSize,memSize,limit);
}

void debug_q2(){
    int sizeA = 512;
    int sizeB = 512;
    int limit = 1000;

    int d = sizeA+sizeB;
    int gridSize = 128; // Arbitrary values that satisfy condition: (d+gridSize-1)/gridSize <= 1024
    int blockSize = (d+gridSize-1)/gridSize;
    int memSize = 2*((d+gridSize-1)/gridSize)*sizeof(int);

    wrapper_q2(sizeA,sizeB,gridSize,blockSize,memSize,limit);
}

void debug_q5(){
    int sizeA = 10;
    int sizeB = 10;
    int num = 100;
    int limit = 1000;

    int d = sizeA + sizeB;
    int gridSize = (num < 1024/d) ? 1: num/(1024/d);
    int blockSize = d*((num+gridSize-1)/gridSize); 
    int memSize = 2*d*((num+gridSize-1)/gridSize)*sizeof(int);

    printf("Gridsize:%d\n", gridSize);
    printf("Blocksize:%d\n", blockSize);
    printf("Memsize:%d\n",memSize);
    
    wrapper_q5(sizeA,sizeB,num,gridSize,blockSize,memSize,limit);
}

void debug_q6(){
    int d = 256;
    int num = 8;
    
    int gridSize = 4;
    int blockSize = 512;
    int memSize = (4*d+2)*((num+gridSize-1)/gridSize)*sizeof(int);
    printf("Memsize:%d\n",memSize);
    int limit = 1000;
    wrapper_q6(d,num,gridSize,blockSize,memSize,limit);
}

int main() {
    debug_q1();
    return 0;
}
