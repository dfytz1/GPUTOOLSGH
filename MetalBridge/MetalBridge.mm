#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

#include "MetalBridge.h"

namespace {

struct MBContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> benchmarkPso = nil;
    id<MTLComputePipelineState> laplacianPso = nil;
    id<MTLComputePipelineState> closestPso = nil;
    id<MTLComputePipelineState> closestCloudPso = nil;
    id<MTLComputePipelineState> delaunayMarkPso = nil;
    id<MTLComputePipelineState> meshEdgesPso = nil;

    // Laplacian: reuse GPU memory across iterations (topology size must match).
    id<MTLBuffer> lapBxIn = nil;
    id<MTLBuffer> lapByIn = nil;
    id<MTLBuffer> lapBzIn = nil;
    id<MTLBuffer> lapBxOut = nil;
    id<MTLBuffer> lapByOut = nil;
    id<MTLBuffer> lapBzOut = nil;
    id<MTLBuffer> lapAdj = nil;
    id<MTLBuffer> lapOff = nil;
    id<MTLBuffer> lapVc = nil;
    id<MTLBuffer> lapStr = nil;
    int lapCachedVertexCount = -1;
    int lapCachedNnz = -1;
};

NSString* MetallibPathBesideDylib()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&mb_create_context), &info) == 0 || info.dli_fname == nullptr)
        return nil;

    NSString* dylibPath = [NSString stringWithUTF8String:info.dli_fname];
    NSString* dir = [dylibPath stringByDeletingLastPathComponent];
    NSString* a = [dir stringByAppendingPathComponent:@"default.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:a])
        return a;
    NSString* b = [dir stringByAppendingPathComponent:@"Kernels/default.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:b])
        return b;
    return a;
}

id<MTLComputePipelineState> MakePso(id<MTLDevice> device, id<MTLLibrary> library, NSString* name, NSError** outErr)
{
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil)
        return nil;
    return [device newComputePipelineStateWithFunction:fn error:outErr];
}

} // namespace

int mb_create_context(void** outCtx)
{
    if (outCtx == nullptr)
        return -1;
    *outCtx = nullptr;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil)
            return -2;

        NSString* metallibPath = MetallibPathBesideDylib();
        if (metallibPath == nil)
            return -3;

        NSError* err = nil;
        id<MTLLibrary> library = [device newLibraryWithFile:metallibPath error:&err];
        if (library == nil)
            return -4;

        id<MTLComputePipelineState> bench = MakePso(device, library, @"benchmarkKernel", &err);
        id<MTLComputePipelineState> lap = MakePso(device, library, @"laplacianSmoothKernel", &err);
        id<MTLComputePipelineState> cls = MakePso(device, library, @"closestPointMeshKernel", &err);
        id<MTLComputePipelineState> cld = MakePso(device, library, @"closestPointCloudKernel", &err);
        id<MTLComputePipelineState> del = MakePso(device, library, @"delaunayMarkBadTrisKernel", &err);
        id<MTLComputePipelineState> edg = MakePso(device, library, @"csrDirectedWeightedEdgesKernel", &err);
        if (bench == nil || lap == nil || cls == nil || cld == nil || del == nil || edg == nil)
            return -5;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil)
            return -7;

        auto* ctx = new MBContext();
        ctx->device = device;
        ctx->queue = queue;
        ctx->benchmarkPso = bench;
        ctx->laplacianPso = lap;
        ctx->closestPso = cls;
        ctx->closestCloudPso = cld;
        ctx->delaunayMarkPso = del;
        ctx->meshEdgesPso = edg;
        *outCtx = ctx;
        return 0;
    }
}

void mb_destroy_context(void* ctx)
{
    if (ctx == nullptr)
        return;
    auto* mb = static_cast<MBContext*>(ctx);
    mb->benchmarkPso = nil;
    mb->laplacianPso = nil;
    mb->closestPso = nil;
    mb->closestCloudPso = nil;
    mb->delaunayMarkPso = nil;
    mb->meshEdgesPso = nil;
    mb->lapBxIn = mb->lapByIn = mb->lapBzIn = nil;
    mb->lapBxOut = mb->lapByOut = mb->lapBzOut = nil;
    mb->lapAdj = mb->lapOff = mb->lapVc = mb->lapStr = nil;
    mb->lapCachedVertexCount = mb->lapCachedNnz = -1;
    mb->queue = nil;
    mb->device = nil;
    delete mb;
}

int mb_run_benchmark(void* ctx, float* buffer, int elementCount, int innerIters, int outerIters)
{
    if (ctx == nullptr || buffer == nullptr || elementCount <= 0 || innerIters <= 0 || outerIters <= 0)
        return -10;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->benchmarkPso == nil)
        return -11;

    const NSUInteger byteLen = static_cast<NSUInteger>(elementCount) * sizeof(float);
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> dataBuf = [mb->device newBufferWithLength:byteLen options:opts];
    if (dataBuf == nil)
        return -12;
    memcpy([dataBuf contents], buffer, byteLen);

    int params[2] = { elementCount, innerIters };
    id<MTLBuffer> paramBuf = [mb->device newBufferWithBytes:params
                                                     length:sizeof(params)
                                                    options:opts];
    if (paramBuf == nil)
        return -13;

    id<MTLComputePipelineState> pso = mb->benchmarkPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadCount = static_cast<NSUInteger>(elementCount);
    const NSUInteger tpg = MIN(maxTpg, MAX(threadCount, 1UL));

    @autoreleasepool {
        for (int o = 0; o < outerIters; o++) {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -14;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -15;

            [enc setComputePipelineState:pso];
            [enc setBuffer:dataBuf offset:0 atIndex:0];
            [enc setBuffer:paramBuf offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    memcpy(buffer, [dataBuf contents], byteLen);
    return 0;
}

int mb_run_laplacian(
    void* ctx,
    float* posXIn,
    float* posYIn,
    float* posZIn,
    float* posXOut,
    float* posYOut,
    float* posZOut,
    int* adjFlat,
    int* rowOffsets,
    int vertexCount,
    float strength)
{
    if (ctx == nullptr || vertexCount <= 0 || posXIn == nullptr || posYIn == nullptr || posZIn == nullptr || posXOut == nullptr
        || posYOut == nullptr || posZOut == nullptr || adjFlat == nullptr || rowOffsets == nullptr)
        return -20;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->laplacianPso == nil)
        return -21;

    const NSUInteger v = static_cast<NSUInteger>(vertexCount);
    const NSUInteger fbytes = v * sizeof(float);
    const NSUInteger offCount = v + 1;
    const NSUInteger offsetBytes = offCount * sizeof(int);
    const int nnz = rowOffsets[vertexCount];
    if (nnz < 0)
        return -22;
    const NSUInteger adjBytes = static_cast<NSUInteger>(nnz) * sizeof(int);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    if (mb->lapCachedVertexCount != vertexCount || mb->lapCachedNnz != nnz) {
        mb->lapBxIn = mb->lapByIn = mb->lapBzIn = nil;
        mb->lapBxOut = mb->lapByOut = mb->lapBzOut = nil;
        mb->lapAdj = mb->lapOff = mb->lapVc = mb->lapStr = nil;

        mb->lapBxIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapByIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBzIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBxOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapByOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBzOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapAdj = [mb->device newBufferWithLength:adjBytes options:opts];
        mb->lapOff = [mb->device newBufferWithLength:offsetBytes options:opts];
        mb->lapVc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->lapStr = [mb->device newBufferWithLength:sizeof(float) options:opts];

        if (mb->lapBxIn == nil || mb->lapByIn == nil || mb->lapBzIn == nil || mb->lapBxOut == nil || mb->lapByOut == nil
            || mb->lapBzOut == nil || mb->lapAdj == nil || mb->lapOff == nil || mb->lapVc == nil || mb->lapStr == nil)
            return -23;

        memcpy([mb->lapAdj contents], adjFlat, adjBytes);
        memcpy([mb->lapOff contents], rowOffsets, offsetBytes);
        mb->lapCachedVertexCount = vertexCount;
        mb->lapCachedNnz = nnz;
    }

    memcpy([mb->lapBxIn contents], posXIn, fbytes);
    memcpy([mb->lapByIn contents], posYIn, fbytes);
    memcpy([mb->lapBzIn contents], posZIn, fbytes);
    memcpy([mb->lapVc contents], &vertexCount, sizeof(int));
    memcpy([mb->lapStr contents], &strength, sizeof(float));

    id<MTLComputePipelineState> pso = mb->laplacianPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(v, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -24;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -25;

        [enc setComputePipelineState:pso];
        [enc setBuffer:mb->lapBxIn offset:0 atIndex:0];
        [enc setBuffer:mb->lapByIn offset:0 atIndex:1];
        [enc setBuffer:mb->lapBzIn offset:0 atIndex:2];
        [enc setBuffer:mb->lapBxOut offset:0 atIndex:3];
        [enc setBuffer:mb->lapByOut offset:0 atIndex:4];
        [enc setBuffer:mb->lapBzOut offset:0 atIndex:5];
        [enc setBuffer:mb->lapAdj offset:0 atIndex:6];
        [enc setBuffer:mb->lapOff offset:0 atIndex:7];
        [enc setBuffer:mb->lapVc offset:0 atIndex:8];
        [enc setBuffer:mb->lapStr offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(v, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(posXOut, [mb->lapBxOut contents], fbytes);
    memcpy(posYOut, [mb->lapByOut contents], fbytes);
    memcpy(posZOut, [mb->lapBzOut contents], fbytes);
    return 0;
}

int mb_run_laplacian_iterations(
    void* ctx,
    float* posX,
    float* posY,
    float* posZ,
    int* adjFlat,
    int* rowOffsets,
    int vertexCount,
    float strength,
    int iterations)
{
    if (ctx == nullptr || vertexCount <= 0)
        return -20;
    if (iterations <= 0)
        return 0;
    if (posX == nullptr || posY == nullptr || posZ == nullptr || adjFlat == nullptr || rowOffsets == nullptr)
        return -20;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->laplacianPso == nil)
        return -21;

    const NSUInteger v = static_cast<NSUInteger>(vertexCount);
    const NSUInteger fbytes = v * sizeof(float);
    const NSUInteger offCount = v + 1;
    const NSUInteger offsetBytes = offCount * sizeof(int);
    const int nnz = rowOffsets[vertexCount];
    if (nnz < 0)
        return -22;
    const NSUInteger adjBytes = static_cast<NSUInteger>(nnz) * sizeof(int);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    if (mb->lapCachedVertexCount != vertexCount || mb->lapCachedNnz != nnz) {
        mb->lapBxIn = mb->lapByIn = mb->lapBzIn = nil;
        mb->lapBxOut = mb->lapByOut = mb->lapBzOut = nil;
        mb->lapAdj = mb->lapOff = mb->lapVc = mb->lapStr = nil;

        mb->lapBxIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapByIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBzIn = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBxOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapByOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapBzOut = [mb->device newBufferWithLength:fbytes options:opts];
        mb->lapAdj = [mb->device newBufferWithLength:adjBytes options:opts];
        mb->lapOff = [mb->device newBufferWithLength:offsetBytes options:opts];
        mb->lapVc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->lapStr = [mb->device newBufferWithLength:sizeof(float) options:opts];

        if (mb->lapBxIn == nil || mb->lapByIn == nil || mb->lapBzIn == nil || mb->lapBxOut == nil || mb->lapByOut == nil
            || mb->lapBzOut == nil || mb->lapAdj == nil || mb->lapOff == nil || mb->lapVc == nil || mb->lapStr == nil)
            return -23;

        memcpy([mb->lapAdj contents], adjFlat, adjBytes);
        memcpy([mb->lapOff contents], rowOffsets, offsetBytes);
        mb->lapCachedVertexCount = vertexCount;
        mb->lapCachedNnz = nnz;
    }

    memcpy([mb->lapBxIn contents], posX, fbytes);
    memcpy([mb->lapByIn contents], posY, fbytes);
    memcpy([mb->lapBzIn contents], posZ, fbytes);
    memcpy([mb->lapVc contents], &vertexCount, sizeof(int));
    memcpy([mb->lapStr contents], &strength, sizeof(float));

    id<MTLComputePipelineState> pso = mb->laplacianPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(v, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -24;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -25;

        [enc setComputePipelineState:pso];
        [enc setBuffer:mb->lapAdj offset:0 atIndex:6];
        [enc setBuffer:mb->lapOff offset:0 atIndex:7];
        [enc setBuffer:mb->lapVc offset:0 atIndex:8];
        [enc setBuffer:mb->lapStr offset:0 atIndex:9];

        for (int it = 0; it < iterations; it++) {
            const bool readA = (it % 2) == 0;
            id<MTLBuffer> inX = readA ? mb->lapBxIn : mb->lapBxOut;
            id<MTLBuffer> inY = readA ? mb->lapByIn : mb->lapByOut;
            id<MTLBuffer> inZ = readA ? mb->lapBzIn : mb->lapBzOut;
            id<MTLBuffer> outX = readA ? mb->lapBxOut : mb->lapBxIn;
            id<MTLBuffer> outY = readA ? mb->lapByOut : mb->lapByIn;
            id<MTLBuffer> outZ = readA ? mb->lapBzOut : mb->lapBzIn;
            [enc setBuffer:inX offset:0 atIndex:0];
            [enc setBuffer:inY offset:0 atIndex:1];
            [enc setBuffer:inZ offset:0 atIndex:2];
            [enc setBuffer:outX offset:0 atIndex:3];
            [enc setBuffer:outY offset:0 atIndex:4];
            [enc setBuffer:outZ offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(v, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    const bool resultInB = (iterations % 2) == 1;
    id<MTLBuffer> fx = resultInB ? mb->lapBxOut : mb->lapBxIn;
    id<MTLBuffer> fy = resultInB ? mb->lapByOut : mb->lapByIn;
    id<MTLBuffer> fz = resultInB ? mb->lapBzOut : mb->lapBzIn;
    memcpy(posX, [fx contents], fbytes);
    memcpy(posY, [fy contents], fbytes);
    memcpy(posZ, [fz contents], fbytes);
    return 0;
}

int mb_closest_points_mesh(
    void* ctx,
    float* qx,
    float* qy,
    float* qz,
    int queryCount,
    float* vx,
    float* vy,
    float* vz,
    int vertexCount,
    int* triIndices,
    int triangleCount,
    float* outCx,
    float* outCy,
    float* outCz,
    float* outDistSq,
    int* outTriIndex)
{
    if (ctx == nullptr || queryCount <= 0 || triangleCount <= 0 || vertexCount <= 0 || qx == nullptr || qy == nullptr
        || qz == nullptr || vx == nullptr || vy == nullptr || vz == nullptr || triIndices == nullptr || outCx == nullptr
        || outCy == nullptr || outCz == nullptr || outDistSq == nullptr || outTriIndex == nullptr)
        return -30;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->closestPso == nil)
        return -31;

    const NSUInteger qc = static_cast<NSUInteger>(queryCount);
    const NSUInteger tc = static_cast<NSUInteger>(triangleCount);
    const NSUInteger vc = static_cast<NSUInteger>(vertexCount);
    const NSUInteger triIdxBytes = tc * 3u * sizeof(int);
    const NSUInteger vbytes = vc * sizeof(float);
    const NSUInteger qbytes = qc * sizeof(float);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bqx = [mb->device newBufferWithBytes:qx length:qbytes options:opts];
    id<MTLBuffer> bqy = [mb->device newBufferWithBytes:qy length:qbytes options:opts];
    id<MTLBuffer> bqz = [mb->device newBufferWithBytes:qz length:qbytes options:opts];
    id<MTLBuffer> bvx = [mb->device newBufferWithBytes:vx length:vbytes options:opts];
    id<MTLBuffer> bvy = [mb->device newBufferWithBytes:vy length:vbytes options:opts];
    id<MTLBuffer> bvz = [mb->device newBufferWithBytes:vz length:vbytes options:opts];
    id<MTLBuffer> bTri = [mb->device newBufferWithBytes:triIndices length:triIdxBytes options:opts];
    id<MTLBuffer> bQc = [mb->device newBufferWithBytes:&queryCount length:sizeof(int) options:opts];
    id<MTLBuffer> bTc = [mb->device newBufferWithBytes:&triangleCount length:sizeof(int) options:opts];
    id<MTLBuffer> bOutX = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutY = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutZ = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutD = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutI = [mb->device newBufferWithLength:qc * sizeof(int) options:opts];

    if (bqx == nil || bqy == nil || bqz == nil || bvx == nil || bvy == nil || bvz == nil || bTri == nil || bQc == nil
        || bTc == nil || bOutX == nil || bOutY == nil || bOutZ == nil || bOutD == nil || bOutI == nil)
        return -32;

    id<MTLComputePipelineState> pso = mb->closestPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(qc, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -33;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -34;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bqx offset:0 atIndex:0];
        [enc setBuffer:bqy offset:0 atIndex:1];
        [enc setBuffer:bqz offset:0 atIndex:2];
        [enc setBuffer:bvx offset:0 atIndex:3];
        [enc setBuffer:bvy offset:0 atIndex:4];
        [enc setBuffer:bvz offset:0 atIndex:5];
        [enc setBuffer:bTri offset:0 atIndex:6];
        [enc setBuffer:bQc offset:0 atIndex:7];
        [enc setBuffer:bTc offset:0 atIndex:8];
        [enc setBuffer:bOutX offset:0 atIndex:9];
        [enc setBuffer:bOutY offset:0 atIndex:10];
        [enc setBuffer:bOutZ offset:0 atIndex:11];
        [enc setBuffer:bOutD offset:0 atIndex:12];
        [enc setBuffer:bOutI offset:0 atIndex:13];
        [enc dispatchThreads:MTLSizeMake(qc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(outCx, [bOutX contents], qbytes);
    memcpy(outCy, [bOutY contents], qbytes);
    memcpy(outCz, [bOutZ contents], qbytes);
    memcpy(outDistSq, [bOutD contents], qbytes);
    memcpy(outTriIndex, [bOutI contents], qc * sizeof(int));
    return 0;
}

int mb_closest_points_cloud(
    void* ctx,
    float* qx,
    float* qy,
    float* qz,
    int queryCount,
    float* px,
    float* py,
    float* pz,
    int targetCount,
    float* outCx,
    float* outCy,
    float* outCz,
    float* outDistSq,
    int* outIndex)
{
    if (ctx == nullptr || queryCount <= 0 || qx == nullptr || qy == nullptr || qz == nullptr || px == nullptr || py == nullptr
        || pz == nullptr || targetCount <= 0 || outCx == nullptr || outCy == nullptr || outCz == nullptr || outDistSq == nullptr
        || outIndex == nullptr)
        return -40;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->closestCloudPso == nil)
        return -41;

    const NSUInteger qc = static_cast<NSUInteger>(queryCount);
    const NSUInteger tc = static_cast<NSUInteger>(targetCount);
    const NSUInteger qbytes = qc * sizeof(float);
    const NSUInteger tbytes = tc * sizeof(float);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bqx = [mb->device newBufferWithBytes:qx length:qbytes options:opts];
    id<MTLBuffer> bqy = [mb->device newBufferWithBytes:qy length:qbytes options:opts];
    id<MTLBuffer> bqz = [mb->device newBufferWithBytes:qz length:qbytes options:opts];
    id<MTLBuffer> bpx = [mb->device newBufferWithBytes:px length:tbytes options:opts];
    id<MTLBuffer> bpy = [mb->device newBufferWithBytes:py length:tbytes options:opts];
    id<MTLBuffer> bpz = [mb->device newBufferWithBytes:pz length:tbytes options:opts];
    id<MTLBuffer> bQc = [mb->device newBufferWithBytes:&queryCount length:sizeof(int) options:opts];
    id<MTLBuffer> bTc = [mb->device newBufferWithBytes:&targetCount length:sizeof(int) options:opts];
    id<MTLBuffer> bOutX = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutY = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutZ = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutD = [mb->device newBufferWithLength:qbytes options:opts];
    id<MTLBuffer> bOutI = [mb->device newBufferWithLength:qc * sizeof(int) options:opts];

    if (bqx == nil || bqy == nil || bqz == nil || bpx == nil || bpy == nil || bpz == nil || bQc == nil || bTc == nil
        || bOutX == nil || bOutY == nil || bOutZ == nil || bOutD == nil || bOutI == nil)
        return -42;

    id<MTLComputePipelineState> pso = mb->closestCloudPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(qc, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -43;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -44;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bqx offset:0 atIndex:0];
        [enc setBuffer:bqy offset:0 atIndex:1];
        [enc setBuffer:bqz offset:0 atIndex:2];
        [enc setBuffer:bpx offset:0 atIndex:3];
        [enc setBuffer:bpy offset:0 atIndex:4];
        [enc setBuffer:bpz offset:0 atIndex:5];
        [enc setBuffer:bQc offset:0 atIndex:6];
        [enc setBuffer:bTc offset:0 atIndex:7];
        [enc setBuffer:bOutX offset:0 atIndex:8];
        [enc setBuffer:bOutY offset:0 atIndex:9];
        [enc setBuffer:bOutZ offset:0 atIndex:10];
        [enc setBuffer:bOutD offset:0 atIndex:11];
        [enc setBuffer:bOutI offset:0 atIndex:12];
        [enc dispatchThreads:MTLSizeMake(qc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(outCx, [bOutX contents], qbytes);
    memcpy(outCy, [bOutY contents], qbytes);
    memcpy(outCz, [bOutZ contents], qbytes);
    memcpy(outDistSq, [bOutD contents], qbytes);
    memcpy(outIndex, [bOutI contents], qc * sizeof(int));
    return 0;
}

int mb_delaunay_mark_bad_triangles(
    void* ctx,
    float* px,
    float* py,
    int vertexCount,
    int* triFlat,
    int triCount,
    float queryX,
    float queryY,
    int* outBad)
{
    if (ctx == nullptr || vertexCount <= 0 || triCount <= 0 || px == nullptr || py == nullptr || triFlat == nullptr
        || outBad == nullptr)
        return -50;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->delaunayMarkPso == nil)
        return -51;

    for (int t = 0; t < triCount; t++)
    {
        int ia = triFlat[3 * t];
        int ib = triFlat[3 * t + 1];
        int ic = triFlat[3 * t + 2];
        if (ia < 0 || ia >= vertexCount || ib < 0 || ib >= vertexCount || ic < 0 || ic >= vertexCount)
            return -52;
    }

    const NSUInteger vc = static_cast<NSUInteger>(vertexCount);
    const NSUInteger tc = static_cast<NSUInteger>(triCount);
    const NSUInteger vbytes = vc * sizeof(float);
    const NSUInteger triBytes = tc * 3u * sizeof(int);
    const NSUInteger badBytes = tc * sizeof(int);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bpx = [mb->device newBufferWithBytes:px length:vbytes options:opts];
    id<MTLBuffer> bpy = [mb->device newBufferWithBytes:py length:vbytes options:opts];
    id<MTLBuffer> bTri = [mb->device newBufferWithBytes:triFlat length:triBytes options:opts];
    id<MTLBuffer> bQx = [mb->device newBufferWithBytes:&queryX length:sizeof(float) options:opts];
    id<MTLBuffer> bQy = [mb->device newBufferWithBytes:&queryY length:sizeof(float) options:opts];
    id<MTLBuffer> bOut = [mb->device newBufferWithLength:badBytes options:opts];
    id<MTLBuffer> bTc = [mb->device newBufferWithBytes:&triCount length:sizeof(int) options:opts];

    if (bpx == nil || bpy == nil || bTri == nil || bQx == nil || bQy == nil || bOut == nil || bTc == nil)
        return -53;

    id<MTLComputePipelineState> pso = mb->delaunayMarkPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(tc, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -54;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -55;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bpx offset:0 atIndex:0];
        [enc setBuffer:bpy offset:0 atIndex:1];
        [enc setBuffer:bTri offset:0 atIndex:2];
        [enc setBuffer:bQx offset:0 atIndex:3];
        [enc setBuffer:bQy offset:0 atIndex:4];
        [enc setBuffer:bOut offset:0 atIndex:5];
        [enc setBuffer:bTc offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(outBad, [bOut contents], badBytes);
    return 0;
}

int mb_build_weighted_edges_csr(
    void* ctx,
    float* vx,
    float* vy,
    float* vz,
    int vertexCount,
    int* rowOffsets,
    int* adjFlat,
    int* edgeWriteBase,
    int edgeCount,
    int* edgeU,
    int* edgeV,
    float* edgeW)
{
    if (ctx == nullptr || vertexCount <= 0 || edgeCount < 0 || vx == nullptr || vy == nullptr || vz == nullptr
        || rowOffsets == nullptr || adjFlat == nullptr || edgeWriteBase == nullptr || edgeU == nullptr || edgeV == nullptr
        || edgeW == nullptr)
        return -60;

    const int nnz = rowOffsets[vertexCount];
    if (nnz != edgeCount)
        return -61;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->meshEdgesPso == nil)
        return -62;

    const NSUInteger v = static_cast<NSUInteger>(vertexCount);
    const NSUInteger vbytes = v * sizeof(float);
    const NSUInteger offBytes = (v + 1u) * sizeof(int);
    const NSUInteger ebytes = static_cast<NSUInteger>(edgeCount) * sizeof(int);
    const NSUInteger wbytes = static_cast<NSUInteger>(edgeCount) * sizeof(float);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bvx = [mb->device newBufferWithBytes:vx length:vbytes options:opts];
    id<MTLBuffer> bvy = [mb->device newBufferWithBytes:vy length:vbytes options:opts];
    id<MTLBuffer> bvz = [mb->device newBufferWithBytes:vz length:vbytes options:opts];
    id<MTLBuffer> bOff = [mb->device newBufferWithBytes:rowOffsets length:offBytes options:opts];
    id<MTLBuffer> bAdj = [mb->device newBufferWithBytes:adjFlat length:static_cast<NSUInteger>(nnz) * sizeof(int) options:opts];
    id<MTLBuffer> bBase = [mb->device newBufferWithBytes:edgeWriteBase length:offBytes options:opts];
    id<MTLBuffer> bU = [mb->device newBufferWithLength:ebytes options:opts];
    id<MTLBuffer> bV = [mb->device newBufferWithLength:ebytes options:opts];
    id<MTLBuffer> bW = [mb->device newBufferWithLength:wbytes options:opts];
    id<MTLBuffer> bVc = [mb->device newBufferWithBytes:&vertexCount length:sizeof(int) options:opts];

    if (bvx == nil || bvy == nil || bvz == nil || bOff == nil || bAdj == nil || bBase == nil || bU == nil || bV == nil
        || bW == nil || bVc == nil)
        return -63;

    id<MTLComputePipelineState> pso = mb->meshEdgesPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, MAX(v, 1UL));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -64;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -65;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bvx offset:0 atIndex:0];
        [enc setBuffer:bvy offset:0 atIndex:1];
        [enc setBuffer:bvz offset:0 atIndex:2];
        [enc setBuffer:bOff offset:0 atIndex:3];
        [enc setBuffer:bAdj offset:0 atIndex:4];
        [enc setBuffer:bBase offset:0 atIndex:5];
        [enc setBuffer:bU offset:0 atIndex:6];
        [enc setBuffer:bV offset:0 atIndex:7];
        [enc setBuffer:bW offset:0 atIndex:8];
        [enc setBuffer:bVc offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(v, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(edgeU, [bU contents], ebytes);
    memcpy(edgeV, [bV contents], ebytes);
    memcpy(edgeW, [bW contents], wbytes);
    return 0;
}
