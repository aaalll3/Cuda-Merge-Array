#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

void checkCuda(cudaError_t err){
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE); 
    }
}

const int GridSize = 128;
const int BlockSize = 1024;
const int MemSize = 2**20;

struct {
    int x;
    int y;
}Tpoint;

__global__ void mergeSmall_k(int *A, int *B, int *M, int sizeA, int sizeB) {
    assert(blockIdx.x == 0); // only for one block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int buff[]; // total:(sizeA+sizeB)*2
    int *a,*b,*m;
    a = buff;
    b = buff+sizeof(int)*sizeA;
    m = buff+sizeof(int)*(sizeA+sizeB);
    if(i<sizeA){
        a[i] = A[i];
    }
    if(i<sizeB){
        b[i] = B[i];
    }
    __syncthreads(); //copy to shared memory and synchronized

    if (i < sizeA + sizeB) {
        int Kx, Ky, Px, Py; // K for low point in diag, P for high
        
        if (i > sizeA) {
            Kx = i - sizeA;
            Ky = sizeA;
            Px = sizeA;
            Py = i - sizeA;
        } else {
            Kx = 0;
            Ky = i;
            Px = i;
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

    M[i] = m[i];
    __syncthreads();
}

__global__ void mergeSmallBlocks_k(int *A, int *B, int *M, int sizeA, int sizeB, int sx, int sy, int ex, int ey) {
    // only locates sufficiant array to each block's shared memory, else not in shared memory ==> pass
    // each block's resposible for [startm,endm) <= d in potential result array M
    // x in array A, and y in array b
    // (sx,sy) notes startm, (ex,ey) notes endm with sx+sy = startm, ex+ey = endm
    int big_i = blockIdx.x * blockDim.x + threadIdx.x;
    int small_i = threadIdx.x;
    int sizesubA = ex - sx;
    int sizesubB = ey - sy;
    extern __shared__ int buff[];
    int *a,*b,*m;
    a = buff;
    b = buff+sizeof(int)*sizesubA;
    m = buff+sizeof(int)*(sizesubA+sizesubB);
    if(small_i >= 0 &&small_i <sizesubA){
        a[small_i] = A[small_i + sx];
    }else{
        b[small_i-sizesubA] = B[small_i - sizesubA + sy];
    }
    __syncthreads(); //copy to shared memory and synchronized

    if (small_i < sizesubA + sizesubB) {
        int Kx, Ky, Px, Py;
        
        if (small_i > sizesubA) {
            Kx = small_i - sizesubA;
            Ky = sizesubA;
            Px = sizesubA;
            Py = small_i - sizesubA;
        } else {
            Kx = 0;
            Ky = small_i;
            Px = small_i;
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
    M[small_i] = m[small_i];
    __syncthreads();
}

__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int* Apoint, int *Bpoint, int Num int d){
    // Apoint is the start idx of each Ai array in A;
    // Apoint idx in [0,Num] with Apoint[Num] is end idx A exclude
    // each block have one pair of AB, suppose sizeA+sizeB = d <= 1024
    // so each block may have to deal with multiple pairs.
    int tidx = threadIdx.x%d; // thread idx in its array
    int Qt = (threadIdx.x-tidx)/d; // pair idx in its block 
    int gbx = Qt + blockIdx.x*(blockDim.x/d); // pair idx for all
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sizeA=Apoint[gbx+1]-Apoint[gbx];
    int sizeB=Apoint[gbx+1]-Apoint[gbx];

    extern __shared__ int buff[]; // total: d*Num of pair
    int *a,*b,*m;
    a = buff + Qt*d;
    b = buff + Qt*d + sizeof(int)*sizeA;
    m = buff + Qt*d + sizeof(int)*(sizeA+sizeB);

    if(tidx<sizeA){
        a[tidx] = A[Apoint[gbx]+tidx];
    }else{
        b[tidx-sizeA] = B[Apoint[gbx]+tidx-sizeA];
    }
    __syncthreads(); //copy to shared memory and synchronized
    if (tidx<sizeA+sizeB) {
        int Kx, Ky, Px, Py; // K for low point in diag, P for high
        if (tidx > sizeA) {
            Kx = tidx - sizeA;
            Ky = sizeA;
            Px = sizeA;
            Py = tidx - sizeA;
        } else {
            Kx = 0;
            Ky = tidx;
            Px = tidx;
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
                        m[tidx] = a[Qy];  // Merge in M
                    } else {
                        m[tidx] = b[Qx];
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global idx
    int elemIdx = threadIdx.x%d; // element idx in one array
    int arrIdxBlk = (threadIdx.x-tidx)/d; // local array idx in this block
    int arrIdxAll = arrIdxBlk + blockIdx.x*(blockDim.x/d); // global array idx

    // copy to shared memory, only copy needed
    // m -- M
    extern __shared__ int buff[];
    int *m;
    int *m2;
    m = buff;
    m2 = buff + arrIdxBlk*d;
    if((threadIdx.x+d-1)/d>arrIdxBlk){
        m[arrIdxBlk*d + elemIdx] = M[Mpoint[arrIdxAll]+elemIdx];
    }
    __syncthreads();
    int subfix = tidx;
    int level = tidx; // till ceil(log2(d)) level
    int cover = 1;
    int sib;
    // stage1:split array to 1 element and merge sort
    // simple strategy: each time, merge 2 neighbor(greedy)
    // worst case 1/3 threads idle, promise log(d)?
    while((threadIdx.x+d-1)/d>arrIdxBlk){
        sib = subfix%2;
        if(!(subfix=d-1&&subfix%2==0)){ 
            mergeInKernal(m+cover*(subfix/2)*2,m+cover*(subfix/2)*2+cover,cover,cover,m2,threadIdx.x%2);
        }
        subfix/=2;
        cover*=2;
    }
}

__device__ void mergeLargeBlocks(int *M, int *Mpoint, int Num, int d){
//TODO
}

__device__ void sortSmallBatch(int *M,int *Mpoint, int Num, int d){
//TODO
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

    mergeSmall_k<<<gridSize,blockSize,memSize>>>(d_A,d_B,d_M,sizeA,sizeB);    

    cudaMemcpy(d_M, M, (sizeA+sizeB)*sizeof(int), cudaMemcpyDeviceToHost);

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
