// Scoped experimental CUDA VMM allocator; contents and tensor layouts are unchanged.
#include <cuda.h>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace {
struct Allocation { CUmemGenericAllocationHandle handle; size_t bytes; };
std::mutex lock;
std::unordered_map<void*,Allocation> allocations;
std::atomic<unsigned long long> count{0},compressed{0},bytes{0},failures{0};
bool check(CUresult e,const char* operation){
    if(e==CUDA_SUCCESS)return true;
    const char* description=nullptr;cuGetErrorString(e,&description);
    std::fprintf(stderr,"ILC %s: %d %s\n",operation,int(e),description?description:"");
    failures.fetch_add(1);return false;
}
}

extern "C" void* mw_ilc_malloc(size_t requested,int device,CUstream){
    CUmemAllocationProp prop{};
    prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=device;
    prop.allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC;
    size_t granularity=0;
    if(!check(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"granularity"))return nullptr;
    size_t size=((requested+granularity-1)/granularity)*granularity;
    CUmemGenericAllocationHandle handle;
    if(!check(cuMemCreate(&handle,size,&prop,0),"create"))return nullptr;
    CUmemAllocationProp obtained{};
    if(!check(cuMemGetAllocationPropertiesFromHandle(&obtained,handle),"properties")){cuMemRelease(handle);return nullptr;}
    CUdeviceptr address;
    if(!check(cuMemAddressReserve(&address,size,granularity,0,0),"reserve")){cuMemRelease(handle);return nullptr;}
    if(!check(cuMemMap(address,size,0,handle,0),"map")){cuMemAddressFree(address,size);cuMemRelease(handle);return nullptr;}
    CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if(!check(cuMemSetAccess(address,size,&access,1),"access")){cuMemUnmap(address,size);cuMemAddressFree(address,size);cuMemRelease(handle);return nullptr;}
    void* pointer=reinterpret_cast<void*>(address);
    {std::lock_guard<std::mutex> guard(lock);allocations.emplace(pointer,Allocation{handle,size});}
    count.fetch_add(1);bytes.fetch_add(size);
    if(obtained.allocFlags.compressionType==CU_MEM_ALLOCATION_COMP_GENERIC)compressed.fetch_add(1);
    return pointer;
}

extern "C" void mw_ilc_free(void* pointer,size_t,int,CUstream){
    if(!pointer)return;
    Allocation allocation;
    {std::lock_guard<std::mutex> guard(lock);auto it=allocations.find(pointer);
     if(it==allocations.end()){failures.fetch_add(1);return;}
     allocation=it->second;allocations.erase(it);}
    CUdeviceptr address=reinterpret_cast<CUdeviceptr>(pointer);
    check(cuMemUnmap(address,allocation.bytes),"unmap");
    check(cuMemAddressFree(address,allocation.bytes),"address free");
    check(cuMemRelease(allocation.handle),"release");
}

extern "C" unsigned long long mw_ilc_stat(int key){
    switch(key){case 0:return count.load();case 1:return compressed.load();case 2:return bytes.load();default:return failures.load();}
}
