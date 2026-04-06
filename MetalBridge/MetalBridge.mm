#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "MetalBridge.h"

#define MB_MG_MAX_LEVELS 4

namespace {

struct MBContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> benchmarkPso = nil;
    id<MTLComputePipelineState> laplacianPso = nil;
    id<MTLComputePipelineState> closestPso = nil;
    id<MTLComputePipelineState> closestCloudPso = nil;
    id<MTLComputePipelineState> meshEdgesPso = nil;
    id<MTLComputePipelineState> jfaInitPso = nil;
    id<MTLComputePipelineState> jfaStepPso = nil;
    id<MTLComputePipelineState> jfaEdgePso = nil;
    id<MTLComputePipelineState> laplaceJacobi3dPso = nil;
    id<MTLComputePipelineState> gradientMag3dPso = nil;
    id<MTLComputePipelineState> normalizeContrast3dPso = nil;
    id<MTLComputePipelineState> zeroVoxelBoundaryPso = nil;
    id<MTLComputePipelineState> laplacianConstrainedPso = nil;
    id<MTLComputePipelineState> femMatVecPso = nil;
    id<MTLComputePipelineState> femFixedPenaltyPso = nil;
    id<MTLComputePipelineState> pcgAxpyPso = nil;
    id<MTLComputePipelineState> pcgPrecondPso = nil;
    id<MTLComputePipelineState> pcgDotPartialPso = nil;
    id<MTLComputePipelineState> pcgReduceLevelPso = nil;
    id<MTLComputePipelineState> pcgReducePso = nil;
    id<MTLComputePipelineState> pcgCopyPso = nil;
    id<MTLComputePipelineState> pcgUintToFloatPso = nil;
    id<MTLComputePipelineState> voxelSamplePso = nil;
    id<MTLComputePipelineState> proximityBlendPso = nil;

    id<MTLComputePipelineState> femMatVecUniquePso = nil;
    id<MTLComputePipelineState> mgJacobiPso = nil;
    id<MTLComputePipelineState> mgResidualPso = nil;
    id<MTLComputePipelineState> mgRestrictPso = nil;
    id<MTLComputePipelineState> mgProlongatePso = nil;
    id<MTLComputePipelineState> mgZeroPso = nil;
    id<MTLComputePipelineState> mgZeroUintPso = nil;
    id<MTLComputePipelineState> mgPcgAxpyPso = nil;
    id<MTLComputePipelineState> mgPcgPrecondPso = nil;
    id<MTLComputePipelineState> mgPcgDotPartialPso = nil;
    id<MTLComputePipelineState> mgPcgReduceLevelPso = nil;
    id<MTLComputePipelineState> mgPcgReducePso = nil;

    id<MTLBuffer> fmuKe = nil;
    id<MTLBuffer> fmuKeIdx = nil;
    id<MTLBuffer> fmuDofMap = nil;
    id<MTLBuffer> fmuRho = nil;
    id<MTLBuffer> fmuV = nil;
    id<MTLBuffer> fmuAv = nil;
    id<MTLBuffer> fmuNElem = nil;
    int fmuCachedNElem = -1;
    int fmuCachedNumUniq = -1;

    id<MTLBuffer> mgKeUnique[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgKeIdx[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgDofMap[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgDiag[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgFixed[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgProlong[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgProlongW[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgNElemBuf[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgNDofBuf[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgNumUniqBuf[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgX[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgB[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgR[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgAx[MB_MG_MAX_LEVELS];
    id<MTLBuffer> mgRho[MB_MG_MAX_LEVELS];

    id<MTLBuffer> mgPcgP = nil;
    id<MTLBuffer> mgPcgZ = nil;
    id<MTLBuffer> mgVcRhs = nil;
    id<MTLBuffer> mgVcSol = nil;
    id<MTLBuffer> mgPcgPartials = nil;
    id<MTLBuffer> mgPcgPartials2 = nil;
    id<MTLBuffer> mgPcgReduceCount = nil;
    id<MTLBuffer> mgPcgResult = nil;
    id<MTLBuffer> mgOmega = nil;
    id<MTLBuffer> mgPenaltyBuf = nil;
    id<MTLBuffer> mgScratchN = nil;
    id<MTLBuffer> mgAxpyOne = nil;
    id<MTLBuffer> mgAxpyZero = nil;
    id<MTLBuffer> mgPcgAlpha = nil;
    id<MTLBuffer> mgPcgNegAlpha = nil;
    id<MTLBuffer> mgPcgBeta = nil;
    id<MTLBuffer> mgPcgNegOne = nil;

    int mgCachedNElem[MB_MG_MAX_LEVELS];
    int mgCachedNDof[MB_MG_MAX_LEVELS];
    int mgCachedNumUnique[MB_MG_MAX_LEVELS];
    int mgCachedNElem0 = -1;
    int mgNumLevels = 0;
    NSUInteger mgDotTpg = 256u;

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

    // Closest-point mesh cache
    id<MTLBuffer> cpMeshVx = nil;
    id<MTLBuffer> cpMeshVy = nil;
    id<MTLBuffer> cpMeshVz = nil;
    id<MTLBuffer> cpMeshTri = nil;
    id<MTLBuffer> cpMeshQx = nil;
    id<MTLBuffer> cpMeshQy = nil;
    id<MTLBuffer> cpMeshQz = nil;
    id<MTLBuffer> cpMeshQc = nil;
    id<MTLBuffer> cpMeshTc = nil;
    id<MTLBuffer> cpMeshOutX = nil;
    id<MTLBuffer> cpMeshOutY = nil;
    id<MTLBuffer> cpMeshOutZ = nil;
    id<MTLBuffer> cpMeshOutD = nil;
    id<MTLBuffer> cpMeshOutI = nil;
    int cpMeshCachedVertexCount = -1;
    int cpMeshCachedTriCount = -1;
    int cpMeshCachedQueryCount = -1;

    // Closest-point cloud cache
    id<MTLBuffer> cpCloudPx = nil;
    id<MTLBuffer> cpCloudPy = nil;
    id<MTLBuffer> cpCloudPz = nil;
    id<MTLBuffer> cpCloudQx = nil;
    id<MTLBuffer> cpCloudQy = nil;
    id<MTLBuffer> cpCloudQz = nil;
    id<MTLBuffer> cpCloudQc = nil;
    id<MTLBuffer> cpCloudTc = nil;
    id<MTLBuffer> cpCloudOutX = nil;
    id<MTLBuffer> cpCloudOutY = nil;
    id<MTLBuffer> cpCloudOutZ = nil;
    id<MTLBuffer> cpCloudOutD = nil;
    id<MTLBuffer> cpCloudOutI = nil;
    int cpCloudCachedTargetCount = -1;
    int cpCloudCachedQueryCount = -1;

    // JFA cache
    id<MTLBuffer> jfaGridA = nil;
    id<MTLBuffer> jfaGridB = nil;
    id<MTLBuffer> jfaBPx = nil;
    id<MTLBuffer> jfaBPy = nil;
    id<MTLBuffer> jfaBN = nil;
    id<MTLBuffer> jfaBRes = nil;
    id<MTLBuffer> jfaBEdges = nil;
    id<MTLBuffer> jfaBStep = nil;
    int jfaCachedPointCount = -1;
    int jfaCachedResolution = -1;

    // FEM / PCG persistent cache (sizes keyed by femCachedNElem / femCachedNDof)
    id<MTLBuffer> femKe = nil;
    id<MTLBuffer> femDofMap = nil;
    id<MTLBuffer> femRho = nil;
    id<MTLBuffer> femV = nil;
    id<MTLBuffer> femAv = nil;
    id<MTLBuffer> femAvFloat = nil;
    id<MTLBuffer> femFixedMask = nil;
    id<MTLBuffer> femPenalty = nil;
    id<MTLBuffer> femNElem = nil;
    id<MTLBuffer> femNDof = nil;

    id<MTLBuffer> pcgU = nil;
    id<MTLBuffer> pcgR = nil;
    id<MTLBuffer> pcgZ = nil;
    id<MTLBuffer> pcgP = nil;
    id<MTLBuffer> pcgDiag = nil;
    id<MTLBuffer> pcgF = nil;

    id<MTLBuffer> pcgAlpha = nil;
    id<MTLBuffer> pcgBeta = nil;
    id<MTLBuffer> pcgNegAlpha = nil;
    id<MTLBuffer> pcgOne = nil;
    id<MTLBuffer> pcgNegOne = nil;
    id<MTLBuffer> pcgScalarOut = nil;
    id<MTLBuffer> pcgPartials = nil;
    id<MTLBuffer> pcgPartials2 = nil;
    id<MTLBuffer> pcgReduceCount = nil;

    int femCachedNElem = -1;
    int femCachedNDof = -1;
    NSUInteger pcgDotTpg = 256u;
};

void ReleaseCpMeshCache(MBContext* mb)
{
    mb->cpMeshVx = mb->cpMeshVy = mb->cpMeshVz = nil;
    mb->cpMeshTri = nil;
    mb->cpMeshQx = mb->cpMeshQy = mb->cpMeshQz = nil;
    mb->cpMeshQc = mb->cpMeshTc = nil;
    mb->cpMeshOutX = mb->cpMeshOutY = mb->cpMeshOutZ = mb->cpMeshOutD = mb->cpMeshOutI = nil;
    mb->cpMeshCachedVertexCount = mb->cpMeshCachedTriCount = mb->cpMeshCachedQueryCount = -1;
}

void ReleaseCpCloudCache(MBContext* mb)
{
    mb->cpCloudPx = mb->cpCloudPy = mb->cpCloudPz = nil;
    mb->cpCloudQx = mb->cpCloudQy = mb->cpCloudQz = nil;
    mb->cpCloudQc = mb->cpCloudTc = nil;
    mb->cpCloudOutX = mb->cpCloudOutY = mb->cpCloudOutZ = mb->cpCloudOutD = mb->cpCloudOutI = nil;
    mb->cpCloudCachedTargetCount = mb->cpCloudCachedQueryCount = -1;
}

void ReleaseJfaCache(MBContext* mb)
{
    mb->jfaGridA = mb->jfaGridB = nil;
    mb->jfaBPx = mb->jfaBPy = nil;
    mb->jfaBN = mb->jfaBRes = nil;
    mb->jfaBEdges = mb->jfaBStep = nil;
    mb->jfaCachedPointCount = mb->jfaCachedResolution = -1;
}

void ReleaseFemPcgCache(MBContext* mb)
{
    mb->femKe = mb->femDofMap = mb->femRho = mb->femV = mb->femAv = nil;
    mb->femAvFloat = nil;
    mb->femFixedMask = mb->femPenalty = mb->femNElem = mb->femNDof = nil;
    mb->pcgU = mb->pcgR = mb->pcgZ = mb->pcgP = nil;
    mb->pcgDiag = mb->pcgF = nil;
    mb->pcgAlpha = mb->pcgBeta = mb->pcgNegAlpha = mb->pcgOne = mb->pcgNegOne = mb->pcgScalarOut = nil;
    mb->pcgPartials = mb->pcgPartials2 = mb->pcgReduceCount = nil;
    mb->femCachedNElem = mb->femCachedNDof = -1;
}

void ReleaseFmuCache(MBContext* mb)
{
    mb->fmuKe = mb->fmuKeIdx = mb->fmuDofMap = mb->fmuRho = mb->fmuV = mb->fmuAv = nil;
    mb->fmuNElem = nil;
    mb->fmuCachedNElem = mb->fmuCachedNumUniq = -1;
}

void ReleaseMgHierarchy(MBContext* mb)
{
    for (int i = 0; i < MB_MG_MAX_LEVELS; i++) {
        mb->mgKeUnique[i] = mb->mgKeIdx[i] = mb->mgDofMap[i] = mb->mgDiag[i] = mb->mgFixed[i] = nil;
        mb->mgProlong[i] = mb->mgProlongW[i] = mb->mgNElemBuf[i] = mb->mgNDofBuf[i] = mb->mgNumUniqBuf[i] = nil;
        mb->mgX[i] = mb->mgB[i] = mb->mgR[i] = mb->mgAx[i] = mb->mgRho[i] = nil;
        mb->mgCachedNElem[i] = mb->mgCachedNDof[i] = mb->mgCachedNumUnique[i] = -1;
    }
    mb->mgPcgP = mb->mgPcgZ = mb->mgVcRhs = mb->mgVcSol = nil;
    mb->mgPcgPartials = mb->mgPcgPartials2 = mb->mgPcgReduceCount = mb->mgPcgResult = nil;
    mb->mgOmega = mb->mgPenaltyBuf = mb->mgScratchN = nil;
    mb->mgAxpyOne = mb->mgAxpyZero = mb->mgPcgAlpha = mb->mgPcgNegAlpha = mb->mgPcgBeta = mb->mgPcgNegOne = nil;
    mb->mgCachedNElem0 = -1;
    mb->mgNumLevels = 0;
}

int FemPcgEnsureBuffers(MBContext* mb, int nElem, int ndof)
{
    if (mb->device == nil)
        return -1;
    if (nElem <= 0 || ndof <= 0)
        return -1;
    if (mb->femCachedNElem == nElem && mb->femCachedNDof == ndof)
        return 0;

    ReleaseFemPcgCache(mb);

    NSUInteger dotTpg = 256u;
    if (mb->pcgDotPartialPso != nil) {
        const NSUInteger m = mb->pcgDotPartialPso.maxTotalThreadsPerThreadgroup;
        if (m < dotTpg) {
            dotTpg = 1u;
            while (dotTpg * 2u <= m)
                dotTpg <<= 1u;
        }
    }
    mb->pcgDotTpg = dotTpg;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    const NSUInteger keLen = static_cast<NSUInteger>(nElem) * 576u * sizeof(float);
    const NSUInteger dmLen = static_cast<NSUInteger>(nElem) * 24u * sizeof(int32_t);
    const NSUInteger rhoLen = static_cast<NSUInteger>(nElem) * sizeof(float);
    const NSUInteger dofF = static_cast<NSUInteger>(ndof) * sizeof(float);
    const NSUInteger dofU = static_cast<NSUInteger>(ndof) * sizeof(uint32_t);

    mb->femKe = [mb->device newBufferWithLength:keLen options:opts];
    mb->femDofMap = [mb->device newBufferWithLength:dmLen options:opts];
    mb->femRho = [mb->device newBufferWithLength:rhoLen options:opts];
    mb->femV = [mb->device newBufferWithLength:dofF options:opts];
    mb->femAv = [mb->device newBufferWithLength:dofU options:opts];
    mb->femAvFloat = [mb->device newBufferWithLength:dofF options:opts];
    mb->femFixedMask = [mb->device newBufferWithLength:static_cast<NSUInteger>(ndof) * sizeof(unsigned char) options:opts];
    mb->femPenalty = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->femNElem = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
    mb->femNDof = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];

    mb->pcgU = [mb->device newBufferWithLength:dofF options:opts];
    mb->pcgR = [mb->device newBufferWithLength:dofF options:opts];
    mb->pcgZ = [mb->device newBufferWithLength:dofF options:opts];
    mb->pcgP = [mb->device newBufferWithLength:dofF options:opts];
    mb->pcgDiag = [mb->device newBufferWithLength:dofF options:opts];
    mb->pcgF = [mb->device newBufferWithLength:dofF options:opts];

    const int tpgI = static_cast<int>(mb->pcgDotTpg);
    const int nPartCap = std::max(1, (ndof + tpgI - 1) / tpgI);
    const NSUInteger partBytes = static_cast<NSUInteger>(nPartCap) * sizeof(float);
    mb->pcgPartials = [mb->device newBufferWithLength:partBytes options:opts];
    mb->pcgPartials2 = [mb->device newBufferWithLength:partBytes options:opts];

    mb->pcgAlpha = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgBeta = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgNegAlpha = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgOne = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgNegOne = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgScalarOut = [mb->device newBufferWithLength:sizeof(float) options:opts];
    mb->pcgReduceCount = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];

    if (mb->femKe == nil || mb->femDofMap == nil || mb->femRho == nil || mb->femV == nil || mb->femAv == nil
        || mb->femAvFloat == nil || mb->femFixedMask == nil || mb->femPenalty == nil || mb->femNElem == nil
        || mb->femNDof == nil || mb->pcgU == nil || mb->pcgR == nil || mb->pcgZ == nil || mb->pcgP == nil
        || mb->pcgDiag == nil || mb->pcgF == nil || mb->pcgPartials == nil || mb->pcgPartials2 == nil
        || mb->pcgAlpha == nil || mb->pcgBeta == nil || mb->pcgNegAlpha == nil || mb->pcgOne == nil
        || mb->pcgNegOne == nil || mb->pcgScalarOut == nil || mb->pcgReduceCount == nil) {
        ReleaseFemPcgCache(mb);
        return -1;
    }

    *static_cast<float*>([mb->pcgOne contents]) = 1.f;
    *static_cast<float*>([mb->pcgNegOne contents]) = -1.f;
    mb->femCachedNElem = nElem;
    mb->femCachedNDof = ndof;
    return 0;
}

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
        id<MTLComputePipelineState> edg = MakePso(device, library, @"csrDirectedWeightedEdgesKernel", &err);
        id<MTLComputePipelineState> jfaI = MakePso(device, library, @"jfaInitKernel", &err);
        id<MTLComputePipelineState> jfaS = MakePso(device, library, @"jfaStepKernel", &err);
        id<MTLComputePipelineState> jfaE = MakePso(device, library, @"jfaExtractEdgesKernel", &err);
        id<MTLComputePipelineState> lj3 = MakePso(device, library, @"laplace_jacobi_3d", &err);
        id<MTLComputePipelineState> gm3 = MakePso(device, library, @"gradient_magnitude_3d", &err);
        id<MTLComputePipelineState> nc3 = MakePso(device, library, @"normalize_contrast_3d", &err);
        id<MTLComputePipelineState> zvb = MakePso(device, library, @"zero_voxel_boundary", &err);
        id<MTLComputePipelineState> lapC = MakePso(device, library, @"laplacianConstrainedKernel", &err);
        id<MTLComputePipelineState> fmv = MakePso(device, library, @"fem_matvec", &err);
        id<MTLComputePipelineState> fmfp = MakePso(device, library, @"fem_apply_fixed_penalty", &err);
        id<MTLComputePipelineState> pcgAx = MakePso(device, library, @"pcg_axpy", &err);
        id<MTLComputePipelineState> pcgPc = MakePso(device, library, @"pcg_precond", &err);
        id<MTLComputePipelineState> pcgDp = MakePso(device, library, @"pcg_dot_partial", &err);
        id<MTLComputePipelineState> pcgRl = MakePso(device, library, @"pcg_reduce_level", &err);
        id<MTLComputePipelineState> pcgRd = MakePso(device, library, @"pcg_reduce", &err);
        id<MTLComputePipelineState> pcgCp = MakePso(device, library, @"pcg_copy", &err);
        id<MTLComputePipelineState> pcgU2f = MakePso(device, library, @"pcg_uint_to_float", &err);
        id<MTLComputePipelineState> vsmp = MakePso(device, library, @"voxel_sample_kernel", &err);
        id<MTLComputePipelineState> prxb = MakePso(device, library, @"proximity_blend_kernel", &err);
        id<MTLComputePipelineState> fmu = MakePso(device, library, @"fem_matvec_unique", &err);
        id<MTLComputePipelineState> mgJ = MakePso(device, library, @"mg_jacobi_update", &err);
        id<MTLComputePipelineState> mgRes = MakePso(device, library, @"mg_residual", &err);
        id<MTLComputePipelineState> mgRest = MakePso(device, library, @"mg_restrict", &err);
        id<MTLComputePipelineState> mgProl = MakePso(device, library, @"mg_prolongate", &err);
        id<MTLComputePipelineState> mgZ = MakePso(device, library, @"mg_zero", &err);
        id<MTLComputePipelineState> mgZu = MakePso(device, library, @"mg_zero_uint", &err);
        id<MTLComputePipelineState> mgAxpy = MakePso(device, library, @"pcg_axpy_mg", &err);
        id<MTLComputePipelineState> mgPc = MakePso(device, library, @"pcg_precond_mg", &err);
        id<MTLComputePipelineState> mgDp = MakePso(device, library, @"mg_pcg_dot_partial", &err);
        id<MTLComputePipelineState> mgRl = MakePso(device, library, @"mg_pcg_reduce_level", &err);
        id<MTLComputePipelineState> mgRd = MakePso(device, library, @"mg_pcg_reduce", &err);
        if (bench == nil || lap == nil || cls == nil || cld == nil || edg == nil || jfaI == nil || jfaS == nil || jfaE == nil
            || lj3 == nil || gm3 == nil || nc3 == nil || zvb == nil || lapC == nil || fmv == nil || fmfp == nil || pcgAx == nil
            || pcgPc == nil || pcgDp == nil || pcgRl == nil || pcgRd == nil || pcgCp == nil || pcgU2f == nil || vsmp == nil
            || prxb == nil || fmu == nil || mgJ == nil || mgRes == nil || mgRest == nil || mgProl == nil || mgZ == nil
            || mgZu == nil || mgAxpy == nil || mgPc == nil || mgDp == nil || mgRl == nil || mgRd == nil)
            return -5;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil)
            return -7;

        auto* ctx = new MBContext();
        for (int mi = 0; mi < MB_MG_MAX_LEVELS; mi++) {
            ctx->mgKeUnique[mi] = ctx->mgKeIdx[mi] = ctx->mgDofMap[mi] = ctx->mgDiag[mi] = ctx->mgFixed[mi] = nil;
            ctx->mgProlong[mi] = ctx->mgProlongW[mi] = ctx->mgNElemBuf[mi] = ctx->mgNDofBuf[mi] = ctx->mgNumUniqBuf[mi] = nil;
            ctx->mgX[mi] = ctx->mgB[mi] = ctx->mgR[mi] = ctx->mgAx[mi] = ctx->mgRho[mi] = nil;
            ctx->mgCachedNElem[mi] = ctx->mgCachedNDof[mi] = ctx->mgCachedNumUnique[mi] = -1;
        }
        ctx->device = device;
        ctx->queue = queue;
        ctx->benchmarkPso = bench;
        ctx->laplacianPso = lap;
        ctx->closestPso = cls;
        ctx->closestCloudPso = cld;
        ctx->meshEdgesPso = edg;
        ctx->jfaInitPso = jfaI;
        ctx->jfaStepPso = jfaS;
        ctx->jfaEdgePso = jfaE;
        ctx->laplaceJacobi3dPso = lj3;
        ctx->gradientMag3dPso = gm3;
        ctx->normalizeContrast3dPso = nc3;
        ctx->zeroVoxelBoundaryPso = zvb;
        ctx->laplacianConstrainedPso = lapC;
        ctx->femMatVecPso = fmv;
        ctx->femFixedPenaltyPso = fmfp;
        ctx->pcgAxpyPso = pcgAx;
        ctx->pcgPrecondPso = pcgPc;
        ctx->pcgDotPartialPso = pcgDp;
        ctx->pcgReduceLevelPso = pcgRl;
        ctx->pcgReducePso = pcgRd;
        ctx->pcgCopyPso = pcgCp;
        ctx->pcgUintToFloatPso = pcgU2f;
        ctx->voxelSamplePso = vsmp;
        ctx->proximityBlendPso = prxb;
        ctx->femMatVecUniquePso = fmu;
        ctx->mgJacobiPso = mgJ;
        ctx->mgResidualPso = mgRes;
        ctx->mgRestrictPso = mgRest;
        ctx->mgProlongatePso = mgProl;
        ctx->mgZeroPso = mgZ;
        ctx->mgZeroUintPso = mgZu;
        ctx->mgPcgAxpyPso = mgAxpy;
        ctx->mgPcgPrecondPso = mgPc;
        ctx->mgPcgDotPartialPso = mgDp;
        ctx->mgPcgReduceLevelPso = mgRl;
        ctx->mgPcgReducePso = mgRd;
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
    mb->meshEdgesPso = nil;
    mb->jfaInitPso = nil;
    mb->jfaStepPso = nil;
    mb->jfaEdgePso = nil;
    mb->laplaceJacobi3dPso = nil;
    mb->gradientMag3dPso = nil;
    mb->normalizeContrast3dPso = nil;
    mb->zeroVoxelBoundaryPso = nil;
    mb->laplacianConstrainedPso = nil;
    mb->femMatVecPso = nil;
    mb->femFixedPenaltyPso = nil;
    mb->pcgAxpyPso = nil;
    mb->pcgPrecondPso = nil;
    mb->pcgDotPartialPso = nil;
    mb->pcgReduceLevelPso = nil;
    mb->pcgReducePso = nil;
    mb->pcgCopyPso = nil;
    mb->pcgUintToFloatPso = nil;
    mb->voxelSamplePso = nil;
    mb->proximityBlendPso = nil;
    mb->femMatVecUniquePso = nil;
    mb->mgJacobiPso = nil;
    mb->mgResidualPso = nil;
    mb->mgRestrictPso = nil;
    mb->mgProlongatePso = nil;
    mb->mgZeroPso = nil;
    mb->mgZeroUintPso = nil;
    mb->mgPcgAxpyPso = nil;
    mb->mgPcgPrecondPso = nil;
    mb->mgPcgDotPartialPso = nil;
    mb->mgPcgReduceLevelPso = nil;
    mb->mgPcgReducePso = nil;
    ReleaseFmuCache(mb);
    ReleaseMgHierarchy(mb);
    mb->lapBxIn = mb->lapByIn = mb->lapBzIn = nil;
    mb->lapBxOut = mb->lapByOut = mb->lapBzOut = nil;
    mb->lapAdj = mb->lapOff = mb->lapVc = mb->lapStr = nil;
    mb->lapCachedVertexCount = mb->lapCachedNnz = -1;
    ReleaseCpMeshCache(mb);
    ReleaseCpCloudCache(mb);
    ReleaseJfaCache(mb);
    ReleaseFemPcgCache(mb);
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
    id<MTLBuffer> paramBuf = [mb->device newBufferWithBytes:params length:sizeof(params) options:opts];
    if (paramBuf == nil)
        return -13;

    id<MTLComputePipelineState> pso = mb->benchmarkPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadCount = static_cast<NSUInteger>(elementCount);
    const NSUInteger tpg = MIN(maxTpg, 256UL);

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
            [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
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
    const NSUInteger tpg = MIN(maxTpg, 256UL);

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
    const NSUInteger tpg = MIN(maxTpg, 256UL);

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
            if (it > 0)
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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

    const bool meshGeomChanged = mb->cpMeshCachedVertexCount != vertexCount || mb->cpMeshCachedTriCount != triangleCount;
    const bool meshQueryChanged = mb->cpMeshCachedQueryCount != queryCount;

    if (meshGeomChanged) {
        mb->cpMeshVx = mb->cpMeshVy = mb->cpMeshVz = nil;
        mb->cpMeshTri = nil;
        mb->cpMeshCachedVertexCount = mb->cpMeshCachedTriCount = -1;
    }
    if (meshQueryChanged) {
        mb->cpMeshQx = mb->cpMeshQy = mb->cpMeshQz = nil;
        mb->cpMeshQc = mb->cpMeshTc = nil;
        mb->cpMeshOutX = mb->cpMeshOutY = mb->cpMeshOutZ = mb->cpMeshOutD = mb->cpMeshOutI = nil;
        mb->cpMeshCachedQueryCount = -1;
    }

    if (mb->cpMeshVx == nil) {
        mb->cpMeshVx = [mb->device newBufferWithLength:vbytes options:opts];
        mb->cpMeshVy = [mb->device newBufferWithLength:vbytes options:opts];
        mb->cpMeshVz = [mb->device newBufferWithLength:vbytes options:opts];
        mb->cpMeshTri = [mb->device newBufferWithLength:triIdxBytes options:opts];
        if (mb->cpMeshVx == nil || mb->cpMeshVy == nil || mb->cpMeshVz == nil || mb->cpMeshTri == nil)
            return -32;
        memcpy([mb->cpMeshVx contents], vx, vbytes);
        memcpy([mb->cpMeshVy contents], vy, vbytes);
        memcpy([mb->cpMeshVz contents], vz, vbytes);
        memcpy([mb->cpMeshTri contents], triIndices, triIdxBytes);
        mb->cpMeshCachedVertexCount = vertexCount;
        mb->cpMeshCachedTriCount = triangleCount;
    }

    if (mb->cpMeshQx == nil) {
        mb->cpMeshQx = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshQy = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshQz = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshQc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->cpMeshTc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->cpMeshOutX = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshOutY = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshOutZ = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshOutD = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpMeshOutI = [mb->device newBufferWithLength:qc * sizeof(int) options:opts];
        if (mb->cpMeshQx == nil || mb->cpMeshQy == nil || mb->cpMeshQz == nil || mb->cpMeshQc == nil || mb->cpMeshTc == nil
            || mb->cpMeshOutX == nil || mb->cpMeshOutY == nil || mb->cpMeshOutZ == nil || mb->cpMeshOutD == nil
            || mb->cpMeshOutI == nil)
            return -32;
        mb->cpMeshCachedQueryCount = queryCount;
    }

    memcpy([mb->cpMeshQx contents], qx, qbytes);
    memcpy([mb->cpMeshQy contents], qy, qbytes);
    memcpy([mb->cpMeshQz contents], qz, qbytes);
    memcpy([mb->cpMeshQc contents], &queryCount, sizeof(int));
    memcpy([mb->cpMeshTc contents], &triangleCount, sizeof(int));

    id<MTLComputePipelineState> pso = mb->closestPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, 256UL);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -33;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -34;

        [enc setComputePipelineState:pso];
        [enc setBuffer:mb->cpMeshQx offset:0 atIndex:0];
        [enc setBuffer:mb->cpMeshQy offset:0 atIndex:1];
        [enc setBuffer:mb->cpMeshQz offset:0 atIndex:2];
        [enc setBuffer:mb->cpMeshVx offset:0 atIndex:3];
        [enc setBuffer:mb->cpMeshVy offset:0 atIndex:4];
        [enc setBuffer:mb->cpMeshVz offset:0 atIndex:5];
        [enc setBuffer:mb->cpMeshTri offset:0 atIndex:6];
        [enc setBuffer:mb->cpMeshQc offset:0 atIndex:7];
        [enc setBuffer:mb->cpMeshTc offset:0 atIndex:8];
        [enc setBuffer:mb->cpMeshOutX offset:0 atIndex:9];
        [enc setBuffer:mb->cpMeshOutY offset:0 atIndex:10];
        [enc setBuffer:mb->cpMeshOutZ offset:0 atIndex:11];
        [enc setBuffer:mb->cpMeshOutD offset:0 atIndex:12];
        [enc setBuffer:mb->cpMeshOutI offset:0 atIndex:13];
        [enc dispatchThreads:MTLSizeMake(qc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(outCx, [mb->cpMeshOutX contents], qbytes);
    memcpy(outCy, [mb->cpMeshOutY contents], qbytes);
    memcpy(outCz, [mb->cpMeshOutZ contents], qbytes);
    memcpy(outDistSq, [mb->cpMeshOutD contents], qbytes);
    memcpy(outTriIndex, [mb->cpMeshOutI contents], qc * sizeof(int));
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
    const NSUInteger tcount = static_cast<NSUInteger>(targetCount);
    const NSUInteger qbytes = qc * sizeof(float);
    const NSUInteger tbytes = tcount * sizeof(float);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    const bool cloudTargetChanged = mb->cpCloudCachedTargetCount != targetCount;
    const bool cloudQueryChanged = mb->cpCloudCachedQueryCount != queryCount;

    if (cloudTargetChanged) {
        mb->cpCloudPx = mb->cpCloudPy = mb->cpCloudPz = nil;
        mb->cpCloudCachedTargetCount = -1;
    }
    if (cloudQueryChanged) {
        mb->cpCloudQx = mb->cpCloudQy = mb->cpCloudQz = nil;
        mb->cpCloudQc = mb->cpCloudTc = nil;
        mb->cpCloudOutX = mb->cpCloudOutY = mb->cpCloudOutZ = mb->cpCloudOutD = mb->cpCloudOutI = nil;
        mb->cpCloudCachedQueryCount = -1;
    }

    if (mb->cpCloudPx == nil) {
        mb->cpCloudPx = [mb->device newBufferWithLength:tbytes options:opts];
        mb->cpCloudPy = [mb->device newBufferWithLength:tbytes options:opts];
        mb->cpCloudPz = [mb->device newBufferWithLength:tbytes options:opts];
        if (mb->cpCloudPx == nil || mb->cpCloudPy == nil || mb->cpCloudPz == nil)
            return -42;
        memcpy([mb->cpCloudPx contents], px, tbytes);
        memcpy([mb->cpCloudPy contents], py, tbytes);
        memcpy([mb->cpCloudPz contents], pz, tbytes);
        mb->cpCloudCachedTargetCount = targetCount;
    }

    if (mb->cpCloudQx == nil) {
        mb->cpCloudQx = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudQy = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudQz = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudQc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->cpCloudTc = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->cpCloudOutX = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudOutY = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudOutZ = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudOutD = [mb->device newBufferWithLength:qbytes options:opts];
        mb->cpCloudOutI = [mb->device newBufferWithLength:qc * sizeof(int) options:opts];
        if (mb->cpCloudQx == nil || mb->cpCloudQy == nil || mb->cpCloudQz == nil || mb->cpCloudQc == nil || mb->cpCloudTc == nil
            || mb->cpCloudOutX == nil || mb->cpCloudOutY == nil || mb->cpCloudOutZ == nil || mb->cpCloudOutD == nil
            || mb->cpCloudOutI == nil)
            return -42;
        mb->cpCloudCachedQueryCount = queryCount;
    }

    memcpy([mb->cpCloudQx contents], qx, qbytes);
    memcpy([mb->cpCloudQy contents], qy, qbytes);
    memcpy([mb->cpCloudQz contents], qz, qbytes);
    memcpy([mb->cpCloudQc contents], &queryCount, sizeof(int));
    memcpy([mb->cpCloudTc contents], &targetCount, sizeof(int));

    id<MTLComputePipelineState> pso = mb->closestCloudPso;
    const NSUInteger maxTpg = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpg = MIN(maxTpg, 256UL);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -43;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -44;

        [enc setComputePipelineState:pso];
        [enc setBuffer:mb->cpCloudQx offset:0 atIndex:0];
        [enc setBuffer:mb->cpCloudQy offset:0 atIndex:1];
        [enc setBuffer:mb->cpCloudQz offset:0 atIndex:2];
        [enc setBuffer:mb->cpCloudPx offset:0 atIndex:3];
        [enc setBuffer:mb->cpCloudPy offset:0 atIndex:4];
        [enc setBuffer:mb->cpCloudPz offset:0 atIndex:5];
        [enc setBuffer:mb->cpCloudQc offset:0 atIndex:6];
        [enc setBuffer:mb->cpCloudTc offset:0 atIndex:7];
        [enc setBuffer:mb->cpCloudOutX offset:0 atIndex:8];
        [enc setBuffer:mb->cpCloudOutY offset:0 atIndex:9];
        [enc setBuffer:mb->cpCloudOutZ offset:0 atIndex:10];
        [enc setBuffer:mb->cpCloudOutD offset:0 atIndex:11];
        [enc setBuffer:mb->cpCloudOutI offset:0 atIndex:12];
        [enc dispatchThreads:MTLSizeMake(qc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(outCx, [mb->cpCloudOutX contents], qbytes);
    memcpy(outCy, [mb->cpCloudOutY contents], qbytes);
    memcpy(outCz, [mb->cpCloudOutZ contents], qbytes);
    memcpy(outDistSq, [mb->cpCloudOutD contents], qbytes);
    memcpy(outIndex, [mb->cpCloudOutI contents], qc * sizeof(int));
    return 0;
}

int mb_jfa_delaunay_2d(
    void* ctx,
    const float* px,
    const float* py,
    int pointCount,
    int* outEdgeA,
    int* outEdgeB,
    int* outEdgeCount,
    int maxEdges,
    int gridResolution)
{
    if (ctx == nullptr || pointCount < 3 || px == nullptr || py == nullptr || outEdgeA == nullptr || outEdgeB == nullptr
        || outEdgeCount == nullptr || maxEdges <= 0)
        return -70;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->jfaInitPso == nil || mb->jfaStepPso == nil || mb->jfaEdgePso == nil)
        return -71;

    int res = 64;
    while (res < gridResolution)
        res *= 2;

    const NSUInteger cellCount = static_cast<NSUInteger>(res * res);
    const NSUInteger gridBytes = cellCount * sizeof(int);
    const NSUInteger edgeBytes = cellCount * 4u * sizeof(int);
    const NSUInteger pBytes = static_cast<NSUInteger>(pointCount) * sizeof(float);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    const bool jfaNeedsRealloc = mb->jfaCachedPointCount != pointCount || mb->jfaCachedResolution != res;
    if (jfaNeedsRealloc) {
        ReleaseJfaCache(mb);
        mb->jfaGridA = [mb->device newBufferWithLength:gridBytes options:opts];
        mb->jfaGridB = [mb->device newBufferWithLength:gridBytes options:opts];
        mb->jfaBPx = [mb->device newBufferWithLength:pBytes options:opts];
        mb->jfaBPy = [mb->device newBufferWithLength:pBytes options:opts];
        mb->jfaBN = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->jfaBRes = [mb->device newBufferWithLength:sizeof(int) options:opts];
        mb->jfaBEdges = [mb->device newBufferWithLength:edgeBytes options:opts];
        mb->jfaBStep = [mb->device newBufferWithLength:sizeof(int) options:opts];
        if (mb->jfaGridA == nil || mb->jfaGridB == nil || mb->jfaBPx == nil || mb->jfaBPy == nil || mb->jfaBN == nil
            || mb->jfaBRes == nil || mb->jfaBEdges == nil || mb->jfaBStep == nil)
            return -72;
        memcpy([mb->jfaBPx contents], px, pBytes);
        memcpy([mb->jfaBPy contents], py, pBytes);
        memcpy([mb->jfaBN contents], &pointCount, sizeof(int));
        memcpy([mb->jfaBRes contents], &res, sizeof(int));
        mb->jfaCachedPointCount = pointCount;
        mb->jfaCachedResolution = res;
    }

    memset([mb->jfaGridA contents], 0xFF, gridBytes);
    memset([mb->jfaGridB contents], 0xFF, gridBytes);

    const NSUInteger maxTpgInit = mb->jfaInitPso.maxTotalThreadsPerThreadgroup;
    const NSUInteger tpgInit = MIN(maxTpgInit, 256UL);
    const NSUInteger tpg2d = 16;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -73;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -74;

        [enc setComputePipelineState:mb->jfaInitPso];
        [enc setBuffer:mb->jfaGridA offset:0 atIndex:0];
        [enc setBuffer:mb->jfaBPx offset:0 atIndex:1];
        [enc setBuffer:mb->jfaBPy offset:0 atIndex:2];
        [enc setBuffer:mb->jfaBN offset:0 atIndex:3];
        [enc setBuffer:mb->jfaBRes offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tpgInit, 1, 1)];

        int step = res / 2;
        id<MTLBuffer> src = mb->jfaGridA;
        id<MTLBuffer> dst = mb->jfaGridB;

        while (step >= 1) {
            memcpy([mb->jfaBStep contents], &step, sizeof(int));
            [enc setComputePipelineState:mb->jfaStepPso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBuffer:mb->jfaBPx offset:0 atIndex:2];
            [enc setBuffer:mb->jfaBPy offset:0 atIndex:3];
            [enc setBuffer:mb->jfaBStep offset:0 atIndex:4];
            [enc setBuffer:mb->jfaBRes offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(res), static_cast<NSUInteger>(res), 1)
             threadsPerThreadgroup:MTLSizeMake(tpg2d, tpg2d, 1)];

            id<MTLBuffer> tmp = src;
            src = dst;
            dst = tmp;
            step /= 2;
        }

        [enc setComputePipelineState:mb->jfaEdgePso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:mb->jfaBEdges offset:0 atIndex:1];
        [enc setBuffer:mb->jfaBRes offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(res), static_cast<NSUInteger>(res), 1)
         threadsPerThreadgroup:MTLSizeMake(tpg2d, tpg2d, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    const int* rawEdges = static_cast<const int*>([mb->jfaBEdges contents]);
    const NSUInteger totalCells = cellCount;

    std::vector<std::pair<int, int>> edgeSet;
    edgeSet.reserve(static_cast<size_t>(pointCount) * 6u);

    for (NSUInteger i = 0; i < totalCells; i++) {
        for (int e = 0; e < 2; e++) {
            int a = rawEdges[i * 4 + e * 2 + 0];
            int b = rawEdges[i * 4 + e * 2 + 1];
            if (a >= 0 && b >= 0 && a != b)
                edgeSet.push_back({ a, b });
        }
    }

    std::sort(edgeSet.begin(), edgeSet.end());
    edgeSet.erase(std::unique(edgeSet.begin(), edgeSet.end()), edgeSet.end());

    int count = static_cast<int>(std::min(edgeSet.size(), static_cast<size_t>(maxEdges)));
    for (int i = 0; i < count; i++) {
        outEdgeA[i] = edgeSet[static_cast<size_t>(i)].first;
        outEdgeB[i] = edgeSet[static_cast<size_t>(i)].second;
    }
    *outEdgeCount = count;
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
    const NSUInteger tpg = MIN(maxTpg, 256UL);

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

namespace {

struct LaplaceParams3DHost {
    uint32_t nx, ny, nz;
    float sv, lv;
};

struct GradParams3DHost {
    uint32_t nx, ny, nz;
    float iDx, iDy, iDz;
};

struct NormParams3DHost {
    uint32_t n;
    float dMin, dMax;
    int32_t invert;
    float exponent;
};

struct BoundaryParamsHost {
    uint32_t nx, ny, nz;
};

/// Prefer pipeline threadExecutionWidth (same idea as mb_run_laplacian_iterations threadgroup sizing).
NSUInteger MbThreadsPerThreadgroup1D(id<MTLComputePipelineState> pso)
{
    NSUInteger tw = pso.threadExecutionWidth;
    if (tw < 1u)
        tw = 64u;
    NSUInteger cap = pso.maxTotalThreadsPerThreadgroup;
    if (cap < 1u)
        cap = tw;
    return MIN(tw, cap);
}

/// Encode dot(x,y) for length @p ndof (partials + hierarchical sum). After commit/wait, read with @ref MbReadDotScalarAfterWait.
static void MbEncodeDotXY(MBContext* mb, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> y, int ndof)
{
    const NSUInteger kTpg = mb->pcgDotTpg;
    *static_cast<int32_t*>([mb->femNDof contents]) = static_cast<int32_t>(ndof);

    [enc setComputePipelineState:mb->pcgDotPartialPso];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:mb->pcgPartials offset:0 atIndex:2];
    [enc setBuffer:mb->femNDof offset:0 atIndex:3];
    const NSUInteger nGrp = (static_cast<NSUInteger>(ndof) + kTpg - 1u) / kTpg;
    [enc dispatchThreads:MTLSizeMake(nGrp * kTpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTpg, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    id<MTLBuffer> curIn = mb->pcgPartials;
    id<MTLBuffer> curOut = mb->pcgPartials2;
    int nCur = static_cast<int>(nGrp);

    [enc setComputePipelineState:mb->pcgReduceLevelPso];
    while (nCur > 1) {
        *static_cast<int32_t*>([mb->pcgReduceCount contents]) = static_cast<int32_t>(nCur);
        [enc setBuffer:curIn offset:0 atIndex:0];
        [enc setBuffer:curOut offset:0 atIndex:1];
        [enc setBuffer:mb->pcgReduceCount offset:0 atIndex:2];
        const NSUInteger nOut = (static_cast<NSUInteger>(nCur) + kTpg - 1u) / kTpg;
        [enc dispatchThreads:MTLSizeMake(nOut * kTpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTpg, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        nCur = static_cast<int>(nOut);
        id<MTLBuffer> t = curIn;
        curIn = curOut;
        curOut = t;
    }
}

static float MbReadDotScalarAfterWait(MBContext* mb, int ndof)
{
    const NSUInteger kTpg = mb->pcgDotTpg;
    const NSUInteger nGrp = (static_cast<NSUInteger>(ndof) + kTpg - 1u) / kTpg;
    if (nGrp <= 1u)
        return static_cast<const float*>([mb->pcgPartials contents])[0];
    id<MTLBuffer> curIn = mb->pcgPartials;
    id<MTLBuffer> curOut = mb->pcgPartials2;
    int nCur = static_cast<int>(nGrp);
    while (nCur > 1) {
        const NSUInteger nOut = (static_cast<NSUInteger>(nCur) + kTpg - 1u) / kTpg;
        nCur = static_cast<int>(nOut);
        id<MTLBuffer> t = curIn;
        curIn = curOut;
        curOut = t;
    }
    return static_cast<const float*>([curIn contents])[0];
}

static void MbFemEncodeMatvecPenaltyUintToFloat(MBContext* mb, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> vBuf)
{
    memset([mb->femAv contents], 0, static_cast<size_t>(mb->femCachedNDof) * sizeof(uint32_t));

    [enc setComputePipelineState:mb->femMatVecPso];
    [enc setBuffer:mb->femKe offset:0 atIndex:0];
    [enc setBuffer:mb->femDofMap offset:0 atIndex:1];
    [enc setBuffer:mb->femRho offset:0 atIndex:2];
    [enc setBuffer:vBuf offset:0 atIndex:3];
    [enc setBuffer:mb->femAv offset:0 atIndex:4];
    [enc setBuffer:mb->femNElem offset:0 atIndex:5];
    NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->femMatVecPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(mb->femCachedNElem), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:mb->femFixedPenaltyPso];
    [enc setBuffer:mb->femAv offset:0 atIndex:0];
    [enc setBuffer:mb->femFixedMask offset:0 atIndex:1];
    [enc setBuffer:vBuf offset:0 atIndex:2];
    [enc setBuffer:mb->femPenalty offset:0 atIndex:3];
    [enc setBuffer:mb->femNDof offset:0 atIndex:4];
    tpg = MbThreadsPerThreadgroup1D(mb->femFixedPenaltyPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(mb->femCachedNDof), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:mb->pcgUintToFloatPso];
    [enc setBuffer:mb->femAv offset:0 atIndex:0];
    [enc setBuffer:mb->femAvFloat offset:0 atIndex:1];
    [enc setBuffer:mb->femNDof offset:0 atIndex:2];
    tpg = MbThreadsPerThreadgroup1D(mb->pcgUintToFloatPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(mb->femCachedNDof), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static void MbEncodeAxpy(
    MBContext* mb,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> out,
    id<MTLBuffer> x,
    id<MTLBuffer> y,
    id<MTLBuffer> aBuf,
    id<MTLBuffer> bBuf,
    int ndof)
{
    *static_cast<int32_t*>([mb->femNDof contents]) = static_cast<int32_t>(ndof);
    [enc setComputePipelineState:mb->pcgAxpyPso];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBuffer:aBuf offset:0 atIndex:3];
    [enc setBuffer:bBuf offset:0 atIndex:4];
    [enc setBuffer:mb->femNDof offset:0 atIndex:5];
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->pcgAxpyPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static int MgCmdWait(id<MTLCommandBuffer> cmd)
{
    if (cmd == nil)
        return -99;
    [cmd waitUntilCompleted];
    return (cmd.status == MTLCommandBufferStatusError) ? -99 : 0;
}

static void MgMatVecUniquePenalty(MBContext* mb, id<MTLComputeCommandEncoder> enc, int lev, id<MTLBuffer> vBuf, id<MTLBuffer> axAtomic)
{
    const int nd = mb->mgCachedNDof[lev];
    const int ne = mb->mgCachedNElem[lev];

    [enc setComputePipelineState:mb->mgZeroUintPso];
    [enc setBuffer:axAtomic offset:0 atIndex:0];
    [enc setBuffer:mb->mgNDofBuf[lev] offset:0 atIndex:1];
    NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgZeroUintPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nd), 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:mb->femMatVecUniquePso];
    [enc setBuffer:mb->mgKeUnique[lev] offset:0 atIndex:0];
    [enc setBuffer:mb->mgKeIdx[lev] offset:0 atIndex:1];
    [enc setBuffer:mb->mgDofMap[lev] offset:0 atIndex:2];
    [enc setBuffer:mb->mgRho[lev] offset:0 atIndex:3];
    [enc setBuffer:vBuf offset:0 atIndex:4];
    [enc setBuffer:axAtomic offset:0 atIndex:5];
    [enc setBuffer:mb->mgNElemBuf[lev] offset:0 atIndex:6];
    tpg = MbThreadsPerThreadgroup1D(mb->femMatVecUniquePso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ne), 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:mb->femFixedPenaltyPso];
    [enc setBuffer:axAtomic offset:0 atIndex:0];
    [enc setBuffer:mb->mgFixed[lev] offset:0 atIndex:1];
    [enc setBuffer:vBuf offset:0 atIndex:2];
    [enc setBuffer:mb->mgPenaltyBuf offset:0 atIndex:3];
    [enc setBuffer:mb->mgNDofBuf[lev] offset:0 atIndex:4];
    tpg = MbThreadsPerThreadgroup1D(mb->femFixedPenaltyPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nd), 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static void MgEncodeDotMG(MBContext* mb, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> y, int n)
{
    const NSUInteger kTpg = mb->mgDotTpg;
    *static_cast<int32_t*>([mb->mgScratchN contents]) = static_cast<int32_t>(n);
    [enc setThreadgroupMemoryLength:kTpg * sizeof(float) atIndex:0];
    [enc setComputePipelineState:mb->mgPcgDotPartialPso];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:mb->mgPcgPartials offset:0 atIndex:2];
    [enc setBuffer:mb->mgScratchN offset:0 atIndex:3];
    const NSUInteger nGrp = (static_cast<NSUInteger>(n) + kTpg - 1u) / kTpg;
    [enc dispatchThreads:MTLSizeMake(nGrp * kTpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTpg, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    id<MTLBuffer> curIn = mb->mgPcgPartials;
    id<MTLBuffer> curOut = mb->mgPcgPartials2;
    int nCur = static_cast<int>(nGrp);

    [enc setThreadgroupMemoryLength:kTpg * sizeof(float) atIndex:0];
    [enc setComputePipelineState:mb->mgPcgReduceLevelPso];
    while (nCur > 1) {
        *static_cast<int32_t*>([mb->mgPcgReduceCount contents]) = static_cast<int32_t>(nCur);
        [enc setBuffer:curIn offset:0 atIndex:0];
        [enc setBuffer:curOut offset:0 atIndex:1];
        [enc setBuffer:mb->mgPcgReduceCount offset:0 atIndex:2];
        const NSUInteger nOut = (static_cast<NSUInteger>(nCur) + kTpg - 1u) / kTpg;
        [enc dispatchThreads:MTLSizeMake(nOut * kTpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTpg, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        nCur = static_cast<int>(nOut);
        id<MTLBuffer> t = curIn;
        curIn = curOut;
        curOut = t;
    }
}

static float MgReadDotScalarMG(MBContext* mb, int n)
{
    const NSUInteger kTpg = mb->mgDotTpg;
    const NSUInteger nGrp = (static_cast<NSUInteger>(n) + kTpg - 1u) / kTpg;
    if (nGrp <= 1u)
        return static_cast<const float*>([mb->mgPcgPartials contents])[0];
    id<MTLBuffer> curIn = mb->mgPcgPartials;
    id<MTLBuffer> curOut = mb->mgPcgPartials2;
    int nCur = static_cast<int>(nGrp);
    while (nCur > 1) {
        const NSUInteger nOut = (static_cast<NSUInteger>(nCur) + kTpg - 1u) / kTpg;
        nCur = static_cast<int>(nOut);
        id<MTLBuffer> t = curIn;
        curIn = curOut;
        curOut = t;
    }
    return static_cast<const float*>([curIn contents])[0];
}

static int MgDotGpu(MBContext* mb, id<MTLBuffer> x, id<MTLBuffer> y, int n, float* outVal)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -99;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -99;
        MgEncodeDotMG(mb, enc, x, y, n);
        [enc endEncoding];
        [cmd commit];
        const int w = MgCmdWait(cmd);
        if (w != 0)
            return w;
        *outVal = MgReadDotScalarMG(mb, n);
    }
    return 0;
}

static void MgEncodeAxpyMg(
    MBContext* mb,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> out,
    id<MTLBuffer> x,
    id<MTLBuffer> y,
    id<MTLBuffer> aBuf,
    id<MTLBuffer> bBuf,
    int ndof)
{
    *static_cast<int32_t*>([mb->mgScratchN contents]) = static_cast<int32_t>(ndof);
    [enc setComputePipelineState:mb->mgPcgAxpyPso];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBuffer:aBuf offset:0 atIndex:3];
    [enc setBuffer:bBuf offset:0 atIndex:4];
    [enc setBuffer:mb->mgScratchN offset:0 atIndex:5];
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgPcgAxpyPso);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static int VCycleMB(
    MBContext* mb,
    int level,
    int numLevels,
    int nSmooth,
    id<MTLBuffer> b0_override,
    id<MTLBuffer> x0_override)
{
    const int nDofL = mb->mgCachedNDof[level];
    const int nElemL = mb->mgCachedNElem[level];
    (void)nElemL;
    id<MTLBuffer> bUse = (level == 0 && b0_override != nil) ? b0_override : mb->mgB[level];
    id<MTLBuffer> xUse = (level == 0 && x0_override != nil) ? x0_override : mb->mgX[level];

    auto runMatJac = [&]() -> int {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            MgMatVecUniquePenalty(mb, enc, level, xUse, mb->mgAx[level]);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:mb->mgJacobiPso];
            [enc setBuffer:xUse offset:0 atIndex:0];
            [enc setBuffer:bUse offset:0 atIndex:1];
            [enc setBuffer:mb->mgAx[level] offset:0 atIndex:2];
            [enc setBuffer:mb->mgDiag[level] offset:0 atIndex:3];
            [enc setBuffer:mb->mgOmega offset:0 atIndex:4];
            [enc setBuffer:mb->mgNDofBuf[level] offset:0 atIndex:5];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgJacobiPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofL), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            return MgCmdWait(cmd);
        }
    };

    for (int s = 0; s < nSmooth; s++) {
        const int e = runMatJac();
        if (e != 0)
            return e;
    }

    if (level == numLevels - 1) {
        for (int s = 0; s < 40; s++) {
            const int e = runMatJac();
            if (e != 0)
                return e;
        }
        return 0;
    }

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            MgMatVecUniquePenalty(mb, enc, level, xUse, mb->mgAx[level]);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:mb->mgResidualPso];
            [enc setBuffer:mb->mgR[level] offset:0 atIndex:0];
            [enc setBuffer:bUse offset:0 atIndex:1];
            [enc setBuffer:mb->mgAx[level] offset:0 atIndex:2];
            [enc setBuffer:mb->mgNDofBuf[level] offset:0 atIndex:3];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgResidualPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofL), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            const int w = MgCmdWait(cmd);
            if (w != 0)
                return w;
        }
    }

    const int nDofC = mb->mgCachedNDof[level + 1];
    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            [enc setComputePipelineState:mb->mgZeroUintPso];
            [enc setBuffer:mb->mgB[level + 1] offset:0 atIndex:0];
            [enc setBuffer:mb->mgNDofBuf[level + 1] offset:0 atIndex:1];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgZeroUintPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofC), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:mb->mgRestrictPso];
            [enc setBuffer:mb->mgR[level] offset:0 atIndex:0];
            [enc setBuffer:mb->mgB[level + 1] offset:0 atIndex:1];
            [enc setBuffer:mb->mgProlong[level + 1] offset:0 atIndex:2];
            [enc setBuffer:mb->mgProlongW[level + 1] offset:0 atIndex:3];
            [enc setBuffer:mb->mgNDofBuf[level] offset:0 atIndex:4];
            tpg = MbThreadsPerThreadgroup1D(mb->mgRestrictPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofL), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            const int w = MgCmdWait(cmd);
            if (w != 0)
                return w;
        }
    }

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            [enc setComputePipelineState:mb->mgZeroPso];
            [enc setBuffer:mb->mgX[level + 1] offset:0 atIndex:0];
            [enc setBuffer:mb->mgNDofBuf[level + 1] offset:0 atIndex:1];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgZeroPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofC), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            const int w = MgCmdWait(cmd);
            if (w != 0)
                return w;
        }
    }

    const int er = VCycleMB(mb, level + 1, numLevels, nSmooth, nil, nil);
    if (er != 0)
        return er;

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            [enc setComputePipelineState:mb->mgProlongatePso];
            [enc setBuffer:xUse offset:0 atIndex:0];
            [enc setBuffer:mb->mgX[level + 1] offset:0 atIndex:1];
            [enc setBuffer:mb->mgProlong[level + 1] offset:0 atIndex:2];
            [enc setBuffer:mb->mgProlongW[level + 1] offset:0 atIndex:3];
            [enc setBuffer:mb->mgNDofBuf[level] offset:0 atIndex:4];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgProlongatePso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nDofL), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            const int w = MgCmdWait(cmd);
            if (w != 0)
                return w;
        }
    }

    for (int s = 0; s < nSmooth; s++) {
        const int e = runMatJac();
        if (e != 0)
            return e;
    }
    return 0;
}

} // namespace

extern "C" {

int mb_laplace_jacobi_3d(
    void* ctx,
    float* inside,
    float* support,
    float* load,
    float* phi,
    int nx,
    int ny,
    int nz,
    float supportVal,
    float loadVal,
    int iterations)
{
    if (ctx == nullptr || inside == nullptr || support == nullptr || load == nullptr || phi == nullptr)
        return -1;
    if (nx <= 0 || ny <= 0 || nz <= 0 || iterations <= 0)
        return -1;

    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const NSUInteger nBytes = static_cast<NSUInteger>(n * sizeof(float));

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->laplaceJacobi3dPso == nil)
        return -1;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> bIn = [mb->device newBufferWithBytes:inside length:nBytes options:opts];
    id<MTLBuffer> bSup = [mb->device newBufferWithBytes:support length:nBytes options:opts];
    id<MTLBuffer> bLoa = [mb->device newBufferWithBytes:load length:nBytes options:opts];
    id<MTLBuffer> bufA = [mb->device newBufferWithLength:nBytes options:opts];
    id<MTLBuffer> bufB = [mb->device newBufferWithLength:nBytes options:opts];
    if (bIn == nil || bSup == nil || bLoa == nil || bufA == nil || bufB == nil)
        return -1;

    memcpy([bufA contents], phi, nBytes);

    LaplaceParams3DHost params{};
    params.nx = static_cast<uint32_t>(nx);
    params.ny = static_cast<uint32_t>(ny);
    params.nz = static_cast<uint32_t>(nz);
    params.sv = supportVal;
    params.lv = loadVal;
    id<MTLBuffer> bParams = [mb->device newBufferWithBytes:&params length:sizeof(params) options:opts];
    if (bParams == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->laplaceJacobi3dPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(n);

    for (int it = 0; it < iterations; it++) {
        const bool aIsSrc = (it % 2) == 0;
        id<MTLBuffer> src = aIsSrc ? bufA : bufB;
        id<MTLBuffer> dst = aIsSrc ? bufB : bufA;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -1;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -1;

            [enc setComputePipelineState:pso];
            [enc setBuffer:bIn offset:0 atIndex:0];
            [enc setBuffer:bSup offset:0 atIndex:1];
            [enc setBuffer:bLoa offset:0 atIndex:2];
            [enc setBuffer:src offset:0 atIndex:3];
            [enc setBuffer:dst offset:0 atIndex:4];
            [enc setBuffer:bParams offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    id<MTLBuffer> outBuf = (iterations % 2 == 1) ? bufB : bufA;
    memcpy(phi, [outBuf contents], nBytes);
    return 0;
}

int mb_gradient_magnitude_3d(
    void* ctx,
    float* phi,
    float* inside,
    float* gradOut,
    int nx,
    int ny,
    int nz,
    float invDx,
    float invDy,
    float invDz)
{
    if (ctx == nullptr || phi == nullptr || inside == nullptr || gradOut == nullptr)
        return -1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
        return -1;

    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const NSUInteger nBytes = static_cast<NSUInteger>(n * sizeof(float));

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->gradientMag3dPso == nil)
        return -1;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> bPhi = [mb->device newBufferWithBytes:phi length:nBytes options:opts];
    id<MTLBuffer> bIn = [mb->device newBufferWithBytes:inside length:nBytes options:opts];
    id<MTLBuffer> bGrad = [mb->device newBufferWithLength:nBytes options:opts];
    if (bPhi == nil || bIn == nil || bGrad == nil)
        return -1;
    memset([bGrad contents], 0, nBytes);

    GradParams3DHost gp{};
    gp.nx = static_cast<uint32_t>(nx);
    gp.ny = static_cast<uint32_t>(ny);
    gp.nz = static_cast<uint32_t>(nz);
    gp.iDx = invDx;
    gp.iDy = invDy;
    gp.iDz = invDz;
    id<MTLBuffer> bParams = [mb->device newBufferWithBytes:&gp length:sizeof(gp) options:opts];
    if (bParams == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->gradientMag3dPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(n);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bPhi offset:0 atIndex:0];
        [enc setBuffer:bIn offset:0 atIndex:1];
        [enc setBuffer:bGrad offset:0 atIndex:2];
        [enc setBuffer:bParams offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(gradOut, [bGrad contents], nBytes);
    return 0;
}

int mb_normalize_contrast_3d(
    void* ctx,
    float* dataInOut,
    float* inside,
    int nx,
    int ny,
    int nz,
    float domainMin,
    float domainMax,
    int invert,
    float exponent)
{
    if (ctx == nullptr || dataInOut == nullptr || inside == nullptr)
        return -1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
        return -1;

    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const NSUInteger nBytes = static_cast<NSUInteger>(n * sizeof(float));

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->normalizeContrast3dPso == nil)
        return -1;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> bData = [mb->device newBufferWithBytes:dataInOut length:nBytes options:opts];
    id<MTLBuffer> bIn = [mb->device newBufferWithBytes:inside length:nBytes options:opts];
    if (bData == nil || bIn == nil)
        return -1;

    NormParams3DHost np{};
    np.n = static_cast<uint32_t>(n);
    np.dMin = domainMin;
    np.dMax = domainMax;
    np.invert = invert;
    np.exponent = exponent;
    id<MTLBuffer> bParams = [mb->device newBufferWithBytes:&np length:sizeof(np) options:opts];
    if (bParams == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->normalizeContrast3dPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(n);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bData offset:0 atIndex:0];
        [enc setBuffer:bIn offset:0 atIndex:1];
        [enc setBuffer:bParams offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(dataInOut, [bData contents], nBytes);
    return 0;
}

int mb_zero_voxel_boundary(void* ctx, float* data, int nx, int ny, int nz)
{
    if (ctx == nullptr || data == nullptr)
        return -1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
        return -1;

    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const NSUInteger nBytes = static_cast<NSUInteger>(n * sizeof(float));

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->zeroVoxelBoundaryPso == nil)
        return -1;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> bData = [mb->device newBufferWithBytes:data length:nBytes options:opts];
    if (bData == nil)
        return -1;

    BoundaryParamsHost bp{};
    bp.nx = static_cast<uint32_t>(nx);
    bp.ny = static_cast<uint32_t>(ny);
    bp.nz = static_cast<uint32_t>(nz);
    id<MTLBuffer> bParams = [mb->device newBufferWithBytes:&bp length:sizeof(bp) options:opts];
    if (bParams == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->zeroVoxelBoundaryPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(n);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bData offset:0 atIndex:0];
        [enc setBuffer:bParams offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(data, [bData contents], nBytes);
    return 0;
}

int mb_run_laplacian_constrained(
    void* ctx,
    float* posX,
    float* posY,
    float* posZ,
    const int* adjFlat,
    const int* rowOffsets,
    int vertexCount,
    float strength,
    int iterations,
    const unsigned char* fixedMask)
{
    if (ctx == nullptr || vertexCount <= 0 || iterations <= 0)
        return -1;
    if (posX == nullptr || posY == nullptr || posZ == nullptr)
        return -1;
    if (adjFlat == nullptr || rowOffsets == nullptr || fixedMask == nullptr)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->laplacianConstrainedPso == nil)
        return -1;

    const NSUInteger v = static_cast<NSUInteger>(vertexCount);
    const NSUInteger fbytes = v * sizeof(float);
    const NSUInteger ubytes = v * sizeof(unsigned char);
    const int nnz = rowOffsets[vertexCount];
    if (nnz < 0)
        return -1;
    const NSUInteger adjBytes = static_cast<NSUInteger>(nnz) * sizeof(int);
    const NSUInteger offBytes = (v + 1u) * sizeof(int);

    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bAX = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bAY = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bAZ = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bBX = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bBY = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bBZ = [mb->device newBufferWithLength:fbytes options:opts];
    id<MTLBuffer> bAdj = [mb->device newBufferWithBytes:adjFlat length:adjBytes options:opts];
    id<MTLBuffer> bOff = [mb->device newBufferWithBytes:rowOffsets length:offBytes options:opts];
    id<MTLBuffer> bVc = [mb->device newBufferWithBytes:&vertexCount length:sizeof(int) options:opts];
    id<MTLBuffer> bStr = [mb->device newBufferWithBytes:&strength length:sizeof(float) options:opts];
    id<MTLBuffer> bFix = [mb->device newBufferWithBytes:fixedMask length:ubytes options:opts];

    if (bAX == nil || bAY == nil || bAZ == nil || bBX == nil || bBY == nil || bBZ == nil || bAdj == nil || bOff == nil
        || bVc == nil || bStr == nil || bFix == nil)
        return -1;

    memcpy([bAX contents], posX, fbytes);
    memcpy([bAY contents], posY, fbytes);
    memcpy([bAZ contents], posZ, fbytes);

    id<MTLComputePipelineState> pso = mb->laplacianConstrainedPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bAdj offset:0 atIndex:6];
        [enc setBuffer:bOff offset:0 atIndex:7];
        [enc setBuffer:bVc offset:0 atIndex:8];
        [enc setBuffer:bStr offset:0 atIndex:9];
        [enc setBuffer:bFix offset:0 atIndex:10];

        for (int it = 0; it < iterations; it++) {
            const bool aIsSrc = (it % 2) == 0;
            [enc setBuffer:(aIsSrc ? bAX : bBX) offset:0 atIndex:0];
            [enc setBuffer:(aIsSrc ? bAY : bBY) offset:0 atIndex:1];
            [enc setBuffer:(aIsSrc ? bAZ : bBZ) offset:0 atIndex:2];
            [enc setBuffer:(aIsSrc ? bBX : bAX) offset:0 atIndex:3];
            [enc setBuffer:(aIsSrc ? bBY : bAY) offset:0 atIndex:4];
            [enc setBuffer:(aIsSrc ? bBZ : bAZ) offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(v, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            if (it < iterations - 1)
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    const bool resultInB = (iterations % 2) == 1;
    memcpy(posX, [(resultInB ? bBX : bAX) contents], fbytes);
    memcpy(posY, [(resultInB ? bBY : bAY) contents], fbytes);
    memcpy(posZ, [(resultInB ? bBZ : bAZ) contents], fbytes);
    return 0;
}

int mb_fem_matvec(
    void* ctx,
    const float* Ke_flat,
    const int* dofMap,
    const float* rho,
    const float* v_in,
    float* Av_out,
    const unsigned char* fixedMask,
    float penalty,
    int nElem,
    int ndof)
{
    if (ctx == nullptr || Ke_flat == nullptr || dofMap == nullptr || rho == nullptr || v_in == nullptr || Av_out == nullptr)
        return -1;
    if (nElem <= 0 || ndof <= 0)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->femMatVecPso == nil)
        return -1;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    const NSUInteger keBytes = static_cast<NSUInteger>(nElem) * 24u * 24u * sizeof(float);
    const NSUInteger dmBytes = static_cast<NSUInteger>(nElem) * 24u * sizeof(int);
    const NSUInteger rhoBytes = static_cast<NSUInteger>(nElem) * sizeof(float);
    const NSUInteger dofBytes = static_cast<NSUInteger>(ndof) * sizeof(float);
    const NSUInteger avUIntBytes = static_cast<NSUInteger>(ndof) * sizeof(uint32_t);
    const NSUInteger nElemSize = sizeof(int);
    const NSUInteger ndofSize = sizeof(int);

    id<MTLBuffer> bKe = [mb->device newBufferWithBytes:Ke_flat length:keBytes options:opts];
    id<MTLBuffer> bDof = [mb->device newBufferWithBytes:dofMap length:dmBytes options:opts];
    id<MTLBuffer> bRho = [mb->device newBufferWithBytes:rho length:rhoBytes options:opts];
    id<MTLBuffer> bV = [mb->device newBufferWithBytes:v_in length:dofBytes options:opts];
    id<MTLBuffer> bAv = [mb->device newBufferWithLength:avUIntBytes options:opts];
    id<MTLBuffer> bN = [mb->device newBufferWithBytes:&nElem length:nElemSize options:opts];

    id<MTLBuffer> bFix = nil;
    id<MTLBuffer> bPen = nil;
    id<MTLBuffer> bNDof = nil;
    if (fixedMask != nullptr && mb->femFixedPenaltyPso != nil) {
        bFix = [mb->device newBufferWithBytes:fixedMask length:static_cast<NSUInteger>(ndof) * sizeof(unsigned char) options:opts];
        bPen = [mb->device newBufferWithBytes:&penalty length:sizeof(float) options:opts];
        bNDof = [mb->device newBufferWithBytes:&ndof length:ndofSize options:opts];
    }

    if (bKe == nil || bDof == nil || bRho == nil || bV == nil || bAv == nil || bN == nil)
        return -1;
    if (fixedMask != nullptr && (bFix == nil || bPen == nil || bNDof == nil))
        return -1;

    memset([bAv contents], 0, avUIntBytes);

    id<MTLComputePipelineState> pso = mb->femMatVecPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bKe offset:0 atIndex:0];
        [enc setBuffer:bDof offset:0 atIndex:1];
        [enc setBuffer:bRho offset:0 atIndex:2];
        [enc setBuffer:bV offset:0 atIndex:3];
        [enc setBuffer:bAv offset:0 atIndex:4];
        [enc setBuffer:bN offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nElem), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

        if (fixedMask != nullptr && mb->femFixedPenaltyPso != nil) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:mb->femFixedPenaltyPso];
            [enc setBuffer:bAv offset:0 atIndex:0];
            [enc setBuffer:bFix offset:0 atIndex:1];
            [enc setBuffer:bV offset:0 atIndex:2];
            [enc setBuffer:bPen offset:0 atIndex:3];
            [enc setBuffer:bNDof offset:0 atIndex:4];
            const NSUInteger tpgP = MbThreadsPerThreadgroup1D(mb->femFixedPenaltyPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpgP, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    auto* avu = static_cast<const uint32_t*>([bAv contents]);
    for (int i = 0; i < ndof; i++) {
        uint32_t u = avu[i];
        float f;
        std::memcpy(&f, &u, sizeof(float));
        Av_out[i] = f;
    }
    return 0;
}

int mb_fem_pcg_solve(
    void* ctx,
    const float* Ke_flat,
    const int* dofMap,
    const unsigned char* fixedMask,
    const float* rho,
    const float* diag,
    const float* f_rhs,
    float* u_inout,
    float penalty,
    int nElem,
    int ndof,
    int maxIter,
    float tolRel)
{
    if (ctx == nullptr || Ke_flat == nullptr || dofMap == nullptr || fixedMask == nullptr || rho == nullptr
        || diag == nullptr || f_rhs == nullptr || u_inout == nullptr)
        return -1;
    if (nElem <= 0 || ndof <= 0 || maxIter <= 0)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->femMatVecPso == nil || mb->femFixedPenaltyPso == nil || mb->pcgAxpyPso == nil
        || mb->pcgPrecondPso == nil || mb->pcgDotPartialPso == nil || mb->pcgReduceLevelPso == nil
        || mb->pcgCopyPso == nil || mb->pcgUintToFloatPso == nil)
        return -1;

    const bool rebindMesh = mb->femCachedNElem != nElem || mb->femCachedNDof != ndof;
    if (FemPcgEnsureBuffers(mb, nElem, ndof) != 0)
        return -1;

    const size_t keBytes = static_cast<size_t>(nElem) * 576u * sizeof(float);
    const size_t dmBytes = static_cast<size_t>(nElem) * 24u * sizeof(int32_t);
    const size_t dofBytes = static_cast<size_t>(ndof) * sizeof(float);

    if (rebindMesh) {
        memcpy([mb->femKe contents], Ke_flat, keBytes);
        memcpy([mb->femDofMap contents], dofMap, dmBytes);
        memcpy([mb->femFixedMask contents], fixedMask, static_cast<size_t>(ndof) * sizeof(unsigned char));
    }
    *static_cast<float*>([mb->femPenalty contents]) = penalty;
    *static_cast<int32_t*>([mb->femNElem contents]) = static_cast<int32_t>(nElem);
    *static_cast<int32_t*>([mb->femNDof contents]) = static_cast<int32_t>(ndof);

    memcpy([mb->femRho contents], rho, static_cast<size_t>(nElem) * sizeof(float));
    memcpy([mb->pcgDiag contents], diag, dofBytes);
    memcpy([mb->pcgF contents], f_rhs, dofBytes);
    memcpy([mb->pcgU contents], u_inout, dofBytes);

    float normB = 1.f;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        MbFemEncodeMatvecPenaltyUintToFloat(mb, enc, mb->pcgU);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        MbEncodeAxpy(mb, enc, mb->pcgR, mb->pcgF, mb->femAvFloat, mb->pcgOne, mb->pcgNegOne, ndof);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        MbEncodeDotXY(mb, enc, mb->pcgF, mb->pcgF, ndof);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    {
        const float ff = MbReadDotScalarAfterWait(mb, ndof);
        normB = std::sqrt(std::max(ff, 0.f));
        if (normB < 1e-30f)
            normB = 1.f;
    }

    float rzOld = 0.f;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        *static_cast<int32_t*>([mb->femNDof contents]) = static_cast<int32_t>(ndof);
        [enc setComputePipelineState:mb->pcgPrecondPso];
        [enc setBuffer:mb->pcgZ offset:0 atIndex:0];
        [enc setBuffer:mb->pcgR offset:0 atIndex:1];
        [enc setBuffer:mb->pcgDiag offset:0 atIndex:2];
        [enc setBuffer:mb->femNDof offset:0 atIndex:3];
        NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->pcgPrecondPso);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:mb->pcgCopyPso];
        [enc setBuffer:mb->pcgP offset:0 atIndex:0];
        [enc setBuffer:mb->pcgZ offset:0 atIndex:1];
        [enc setBuffer:mb->femNDof offset:0 atIndex:2];
        tpg = MbThreadsPerThreadgroup1D(mb->pcgCopyPso);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        MbEncodeDotXY(mb, enc, mb->pcgR, mb->pcgZ, ndof);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    rzOld = MbReadDotScalarAfterWait(mb, ndof);

    float betaGpu = 0.f;

    for (int it = 0; it < maxIter; ++it) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -1;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -1;

            if (it > 0) {
                *static_cast<float*>([mb->pcgBeta contents]) = betaGpu;
                MbEncodeAxpy(mb, enc, mb->pcgP, mb->pcgZ, mb->pcgP, mb->pcgOne, mb->pcgBeta, ndof);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            MbFemEncodeMatvecPenaltyUintToFloat(mb, enc, mb->pcgP);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            MbEncodeDotXY(mb, enc, mb->pcgP, mb->femAvFloat, ndof);

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const float denom = MbReadDotScalarAfterWait(mb, ndof);
        if (std::fabs(denom) < 1e-40f)
            break;

        const float alpha = rzOld / denom;
        *static_cast<float*>([mb->pcgAlpha contents]) = alpha;
        *static_cast<float*>([mb->pcgNegAlpha contents]) = -alpha;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -1;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -1;

            MbEncodeAxpy(mb, enc, mb->pcgU, mb->pcgU, mb->pcgP, mb->pcgOne, mb->pcgAlpha, ndof);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            MbEncodeAxpy(mb, enc, mb->pcgR, mb->pcgR, mb->femAvFloat, mb->pcgOne, mb->pcgNegAlpha, ndof);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            MbEncodeDotXY(mb, enc, mb->pcgR, mb->pcgR, ndof);

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const float nr2 = MbReadDotScalarAfterWait(mb, ndof);
        if (std::sqrt(std::max(nr2, 0.f)) < tolRel * normB)
            break;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -1;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -1;

            *static_cast<int32_t*>([mb->femNDof contents]) = static_cast<int32_t>(ndof);
            [enc setComputePipelineState:mb->pcgPrecondPso];
            [enc setBuffer:mb->pcgZ offset:0 atIndex:0];
            [enc setBuffer:mb->pcgR offset:0 atIndex:1];
            [enc setBuffer:mb->pcgDiag offset:0 atIndex:2];
            [enc setBuffer:mb->femNDof offset:0 atIndex:3];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->pcgPrecondPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            MbEncodeDotXY(mb, enc, mb->pcgR, mb->pcgZ, ndof);

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const float rzNew = MbReadDotScalarAfterWait(mb, ndof);
        betaGpu = rzNew / (rzOld + 1e-40f);
        rzOld = rzNew;
    }

    memcpy(u_inout, [mb->pcgU contents], dofBytes);
    return 0;
}

int mb_fem_matvec_unique(
    void* ctx,
    const float* Ke_unique,
    const int* keIdx,
    const int* dofMap,
    const float* rho,
    const float* v_in,
    float* Av_out,
    int numUnique,
    int nElem,
    int ndof)
{
    if (ctx == nullptr || Ke_unique == nullptr || keIdx == nullptr || dofMap == nullptr || rho == nullptr
        || v_in == nullptr || Av_out == nullptr)
        return -1;
    if (numUnique <= 0 || nElem <= 0 || ndof <= 0)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->femMatVecUniquePso == nil)
        return -1;

    if (mb->fmuCachedNElem != nElem || mb->fmuCachedNumUniq != numUnique) {
        ReleaseFmuCache(mb);
        MTLResourceOptions opts = MTLResourceStorageModeShared;
        const NSUInteger keB = static_cast<NSUInteger>(numUnique) * 576u * sizeof(float);
        const NSUInteger idxB = static_cast<NSUInteger>(nElem) * sizeof(int32_t);
        const NSUInteger dmB = static_cast<NSUInteger>(nElem) * 24u * sizeof(int32_t);
        const NSUInteger rhoB = static_cast<NSUInteger>(nElem) * sizeof(float);
        const NSUInteger vB = static_cast<NSUInteger>(ndof) * sizeof(float);
        const NSUInteger avB = static_cast<NSUInteger>(ndof) * sizeof(uint32_t);

        mb->fmuKe = [mb->device newBufferWithLength:keB options:opts];
        mb->fmuKeIdx = [mb->device newBufferWithLength:idxB options:opts];
        mb->fmuDofMap = [mb->device newBufferWithLength:dmB options:opts];
        mb->fmuRho = [mb->device newBufferWithLength:rhoB options:opts];
        mb->fmuV = [mb->device newBufferWithLength:vB options:opts];
        mb->fmuAv = [mb->device newBufferWithLength:avB options:opts];
        mb->fmuNElem = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];

        if (mb->fmuKe == nil || mb->fmuKeIdx == nil || mb->fmuDofMap == nil || mb->fmuRho == nil || mb->fmuV == nil
            || mb->fmuAv == nil || mb->fmuNElem == nil) {
            ReleaseFmuCache(mb);
            return -1;
        }
        memcpy([mb->fmuKe contents], Ke_unique, static_cast<size_t>(keB));
        memcpy([mb->fmuKeIdx contents], keIdx, static_cast<size_t>(idxB));
        memcpy([mb->fmuDofMap contents], dofMap, static_cast<size_t>(dmB));
        mb->fmuCachedNElem = nElem;
        mb->fmuCachedNumUniq = numUnique;
    } else {
        memcpy([mb->fmuKe contents], Ke_unique, static_cast<size_t>(numUnique) * 576u * sizeof(float));
        memcpy([mb->fmuKeIdx contents], keIdx, static_cast<size_t>(nElem) * sizeof(int32_t));
        memcpy([mb->fmuDofMap contents], dofMap, static_cast<size_t>(nElem) * 24u * sizeof(int32_t));
    }

    memcpy([mb->fmuRho contents], rho, static_cast<size_t>(nElem) * sizeof(float));
    memcpy([mb->fmuV contents], v_in, static_cast<size_t>(ndof) * sizeof(float));
    memset([mb->fmuAv contents], 0, static_cast<size_t>(ndof) * sizeof(uint32_t));

    *static_cast<int32_t*>([mb->fmuNElem contents]) = static_cast<int32_t>(nElem);

    const NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->femMatVecUniquePso);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:mb->femMatVecUniquePso];
        [enc setBuffer:mb->fmuKe offset:0 atIndex:0];
        [enc setBuffer:mb->fmuKeIdx offset:0 atIndex:1];
        [enc setBuffer:mb->fmuDofMap offset:0 atIndex:2];
        [enc setBuffer:mb->fmuRho offset:0 atIndex:3];
        [enc setBuffer:mb->fmuV offset:0 atIndex:4];
        [enc setBuffer:mb->fmuAv offset:0 atIndex:5];
        [enc setBuffer:mb->fmuNElem offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(nElem), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    auto* avu = static_cast<const uint32_t*>([mb->fmuAv contents]);
    for (int i = 0; i < ndof; i++) {
        uint32_t u = avu[i];
        float f;
        std::memcpy(&f, &u, sizeof(float));
        Av_out[i] = f;
    }
    return 0;
}

int mb_fem_mgpcg_solve(
    void* ctx,
    const float* const* mg_ke_unique,
    const int* const* mg_ke_idx,
    const int* const* mg_dof_map,
    const float* const* mg_diag,
    const unsigned char* const* mg_fixed,
    const int* const* mg_prolong,
    const float* const* mg_prolong_w,
    const float* const* mg_rho,
    const int* mg_nelem,
    const int* mg_ndof,
    const int* mg_num_unique,
    int numLevels,
    const float* f_rhs,
    float* u_inout,
    int maxIter,
    float tolRel,
    int nSmooth,
    float omegaJacobi,
    int* out_pcg_iters)
{
    if (ctx == nullptr || mg_ke_unique == nullptr || mg_ke_idx == nullptr || mg_dof_map == nullptr
        || mg_diag == nullptr || mg_fixed == nullptr || mg_rho == nullptr || mg_nelem == nullptr
        || mg_ndof == nullptr || mg_num_unique == nullptr || f_rhs == nullptr || u_inout == nullptr)
        return -1;
    if (numLevels < 2 || numLevels > MB_MG_MAX_LEVELS || maxIter <= 0)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->femMatVecUniquePso == nil || mb->femFixedPenaltyPso == nil
        || mb->mgJacobiPso == nil || mb->mgResidualPso == nil || mb->mgRestrictPso == nil
        || mb->mgProlongatePso == nil || mb->mgZeroPso == nil || mb->mgZeroUintPso == nil
        || mb->mgPcgAxpyPso == nil || mb->mgPcgDotPartialPso == nil || mb->mgPcgReduceLevelPso == nil)
        return -1;

    const int n0 = mg_nelem[0];
    if (mb->mgCachedNElem0 != n0 || mb->mgNumLevels != numLevels) {
        ReleaseMgHierarchy(mb);

        NSUInteger dotTpg = 256u;
        if (mb->mgPcgDotPartialPso != nil) {
            const NSUInteger m = mb->mgPcgDotPartialPso.maxTotalThreadsPerThreadgroup;
            if (m < dotTpg) {
                dotTpg = 1u;
                while (dotTpg * 2u <= m)
                    dotTpg <<= 1u;
            }
        }
        mb->mgDotTpg = dotTpg;

        MTLResourceOptions opts = MTLResourceStorageModeShared;
        const int ndof0 = mg_ndof[0];
        const int nPartCap = std::max(1, (ndof0 + static_cast<int>(dotTpg) - 1) / static_cast<int>(dotTpg));
        const NSUInteger partBytes = static_cast<NSUInteger>(nPartCap) * sizeof(float);
        const NSUInteger dof0Bytes = static_cast<NSUInteger>(ndof0) * sizeof(float);

        mb->mgPcgPartials = [mb->device newBufferWithLength:partBytes options:opts];
        mb->mgPcgPartials2 = [mb->device newBufferWithLength:partBytes options:opts];
        mb->mgPcgReduceCount = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
        mb->mgPcgP = [mb->device newBufferWithLength:dof0Bytes options:opts];
        mb->mgPcgZ = [mb->device newBufferWithLength:dof0Bytes options:opts];
        mb->mgVcRhs = [mb->device newBufferWithLength:dof0Bytes options:opts];
        mb->mgVcSol = [mb->device newBufferWithLength:dof0Bytes options:opts];
        mb->mgOmega = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgPenaltyBuf = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgScratchN = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
        mb->mgAxpyOne = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgAxpyZero = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgPcgAlpha = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgPcgNegAlpha = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgPcgBeta = [mb->device newBufferWithLength:sizeof(float) options:opts];
        mb->mgPcgNegOne = [mb->device newBufferWithLength:sizeof(float) options:opts];

        if (mb->mgPcgPartials == nil || mb->mgPcgPartials2 == nil || mb->mgPcgReduceCount == nil || mb->mgPcgP == nil
            || mb->mgPcgZ == nil || mb->mgVcRhs == nil || mb->mgVcSol == nil || mb->mgOmega == nil
            || mb->mgPenaltyBuf == nil || mb->mgScratchN == nil || mb->mgAxpyOne == nil || mb->mgAxpyZero == nil
            || mb->mgPcgAlpha == nil || mb->mgPcgNegAlpha == nil || mb->mgPcgBeta == nil || mb->mgPcgNegOne == nil) {
            ReleaseMgHierarchy(mb);
            return -1;
        }

        *static_cast<float*>([mb->mgPenaltyBuf contents]) = 1e12f;
        *static_cast<float*>([mb->mgAxpyOne contents]) = 1.f;
        *static_cast<float*>([mb->mgAxpyZero contents]) = 0.f;
        *static_cast<float*>([mb->mgPcgNegOne contents]) = -1.f;

        for (int l = 0; l < numLevels; l++) {
            const int ne = mg_nelem[l];
            const int nd = mg_ndof[l];
            const int nu = mg_num_unique[l];
            if (ne <= 0 || nd <= 0 || nu <= 0)
                return -1;
            if (l >= 1 && (mg_prolong == nullptr || mg_prolong[l] == nullptr || mg_prolong_w == nullptr
                    || mg_prolong_w[l] == nullptr))
                return -1;

            const size_t keB = static_cast<size_t>(nu) * 576u * sizeof(float);
            const size_t idxB = static_cast<size_t>(ne) * sizeof(int32_t);
            const size_t dmB = static_cast<size_t>(ne) * 24u * sizeof(int32_t);
            const size_t diagB = static_cast<size_t>(nd) * sizeof(float);
            const size_t fixB = static_cast<size_t>(nd) * sizeof(unsigned char);
            const size_t rhoB = static_cast<size_t>(ne) * sizeof(float);

            mb->mgKeUnique[l] = [mb->device newBufferWithLength:keB options:opts];
            mb->mgKeIdx[l] = [mb->device newBufferWithLength:idxB options:opts];
            mb->mgDofMap[l] = [mb->device newBufferWithLength:dmB options:opts];
            mb->mgDiag[l] = [mb->device newBufferWithLength:diagB options:opts];
            mb->mgFixed[l] = [mb->device newBufferWithLength:fixB options:opts];
            mb->mgNElemBuf[l] = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
            mb->mgNDofBuf[l] = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
            mb->mgNumUniqBuf[l] = [mb->device newBufferWithLength:sizeof(int32_t) options:opts];
            mb->mgX[l] = [mb->device newBufferWithLength:diagB options:opts];
            mb->mgB[l] = [mb->device newBufferWithLength:diagB options:opts];
            mb->mgR[l] = [mb->device newBufferWithLength:diagB options:opts];
            mb->mgAx[l] = [mb->device newBufferWithLength:static_cast<NSUInteger>(nd) * sizeof(uint32_t) options:opts];
            mb->mgRho[l] = [mb->device newBufferWithLength:rhoB options:opts];

            if (mb->mgKeUnique[l] == nil || mb->mgKeIdx[l] == nil || mb->mgDofMap[l] == nil || mb->mgDiag[l] == nil
                || mb->mgFixed[l] == nil || mb->mgNElemBuf[l] == nil || mb->mgNDofBuf[l] == nil
                || mb->mgNumUniqBuf[l] == nil || mb->mgX[l] == nil || mb->mgB[l] == nil || mb->mgR[l] == nil
                || mb->mgAx[l] == nil || mb->mgRho[l] == nil) {
                ReleaseMgHierarchy(mb);
                return -1;
            }

            memcpy([mb->mgKeUnique[l] contents], mg_ke_unique[l], keB);
            memcpy([mb->mgKeIdx[l] contents], mg_ke_idx[l], idxB);
            memcpy([mb->mgDofMap[l] contents], mg_dof_map[l], dmB);
            memcpy([mb->mgFixed[l] contents], mg_fixed[l], fixB);

            *static_cast<int32_t*>([mb->mgNElemBuf[l] contents]) = static_cast<int32_t>(ne);
            *static_cast<int32_t*>([mb->mgNDofBuf[l] contents]) = static_cast<int32_t>(nd);
            *static_cast<int32_t*>([mb->mgNumUniqBuf[l] contents]) = static_cast<int32_t>(nu);

            if (l == 0) {
                mb->mgProlong[l] = nil;
                mb->mgProlongW[l] = nil;
            } else {
                const size_t pB = static_cast<size_t>(mg_ndof[l - 1]) * 8u * sizeof(int32_t);
                const size_t pwB = static_cast<size_t>(mg_ndof[l - 1]) * 8u * sizeof(float);
                mb->mgProlong[l] = [mb->device newBufferWithLength:pB options:opts];
                mb->mgProlongW[l] = [mb->device newBufferWithLength:pwB options:opts];
                if (mb->mgProlong[l] == nil || mb->mgProlongW[l] == nil) {
                    ReleaseMgHierarchy(mb);
                    return -1;
                }
                memcpy([mb->mgProlong[l] contents], mg_prolong[l], pB);
                memcpy([mb->mgProlongW[l] contents], mg_prolong_w[l], pwB);
            }

            mb->mgCachedNElem[l] = ne;
            mb->mgCachedNDof[l] = nd;
            mb->mgCachedNumUnique[l] = nu;
        }

        mb->mgCachedNElem0 = n0;
        mb->mgNumLevels = numLevels;
    }

    for (int l = 0; l < numLevels; l++) {
        const int ne = mg_nelem[l];
        const int nd = mg_ndof[l];
        memcpy([mb->mgRho[l] contents], mg_rho[l], static_cast<size_t>(ne) * sizeof(float));
        memcpy([mb->mgDiag[l] contents], mg_diag[l], static_cast<size_t>(nd) * sizeof(float));
    }
    *static_cast<float*>([mb->mgOmega contents]) = omegaJacobi;
    memcpy([mb->mgB[0] contents], f_rhs, static_cast<size_t>(mg_ndof[0]) * sizeof(float));
    memcpy([mb->mgX[0] contents], u_inout, static_cast<size_t>(mg_ndof[0]) * sizeof(float));

    const int ndof0 = mb->mgCachedNDof[0];
    const size_t dof0z = static_cast<size_t>(ndof0) * sizeof(float);

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            MgMatVecUniquePenalty(mb, enc, 0, mb->mgX[0], mb->mgAx[0]);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:mb->mgResidualPso];
            [enc setBuffer:mb->mgR[0] offset:0 atIndex:0];
            [enc setBuffer:mb->mgB[0] offset:0 atIndex:1];
            [enc setBuffer:mb->mgAx[0] offset:0 atIndex:2];
            [enc setBuffer:mb->mgNDofBuf[0] offset:0 atIndex:3];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgResidualPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof0), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            const int w = MgCmdWait(cmd);
            if (w != 0)
                return w;
        }
    }

    float normB = 1.f;
    float ff = 0.f;
    int dr = MgDotGpu(mb, mb->mgB[0], mb->mgB[0], ndof0, &ff);
    if (dr != 0)
        return dr;
    normB = std::sqrt(std::max(ff, 0.f));
    if (normB < 1e-30f)
        normB = 1.f;

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            [enc setComputePipelineState:mb->mgZeroPso];
            [enc setBuffer:mb->mgVcSol offset:0 atIndex:0];
            [enc setBuffer:mb->mgNDofBuf[0] offset:0 atIndex:1];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgZeroPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof0), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            if (MgCmdWait(cmd) != 0)
                return -99;
        }
    }

    memcpy([mb->mgVcRhs contents], [mb->mgR[0] contents], dof0z);
    {
        const int vz = VCycleMB(mb, 0, numLevels, nSmooth, mb->mgVcRhs, mb->mgVcSol);
        if (vz != 0)
            return vz;
    }

    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
            if (cmd == nil)
                return -99;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil)
                return -99;
            [enc setComputePipelineState:mb->pcgCopyPso];
            [enc setBuffer:mb->mgPcgP offset:0 atIndex:0];
            [enc setBuffer:mb->mgVcSol offset:0 atIndex:1];
            [enc setBuffer:mb->mgNDofBuf[0] offset:0 atIndex:2];
            NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->pcgCopyPso);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof0), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            if (MgCmdWait(cmd) != 0)
                return -99;
        }
    }

    float rzOld = 0.f;
    dr = MgDotGpu(mb, mb->mgR[0], mb->mgPcgP, ndof0, &rzOld);
    if (dr != 0)
        return dr;

    int itUsed = 0;
    float betaGpu = 0.f;

    for (int iter = 0; iter < maxIter; iter++) {
        itUsed = iter + 1;
        if (iter > 0) {
            *static_cast<float*>([mb->mgPcgBeta contents]) = betaGpu;
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
                if (cmd == nil)
                    return -99;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil)
                    return -99;
                MgEncodeAxpyMg(mb, enc, mb->mgPcgP, mb->mgPcgZ, mb->mgPcgP, mb->mgAxpyOne, mb->mgPcgBeta, ndof0);
                [enc endEncoding];
                [cmd commit];
                if (MgCmdWait(cmd) != 0)
                    return -99;
            }
        }

        {
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
                if (cmd == nil)
                    return -99;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil)
                    return -99;
                MgMatVecUniquePenalty(mb, enc, 0, mb->mgPcgP, mb->mgAx[0]);
                [enc endEncoding];
                [cmd commit];
                if (MgCmdWait(cmd) != 0)
                    return -99;
            }
        }

        float denom = 0.f;
        dr = MgDotGpu(mb, mb->mgPcgP, mb->mgAx[0], ndof0, &denom);
        if (dr != 0)
            return dr;
        if (std::fabs(denom) < 1e-30f)
            break;

        const float alpha = rzOld / denom;
        *static_cast<float*>([mb->mgPcgAlpha contents]) = alpha;
        *static_cast<float*>([mb->mgPcgNegAlpha contents]) = -alpha;

        {
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
                if (cmd == nil)
                    return -99;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil)
                    return -99;
                MgEncodeAxpyMg(mb, enc, mb->mgX[0], mb->mgX[0], mb->mgPcgP, mb->mgAxpyOne, mb->mgPcgAlpha, ndof0);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                MgEncodeAxpyMg(mb, enc, mb->mgR[0], mb->mgR[0], mb->mgAx[0], mb->mgAxpyOne, mb->mgPcgNegAlpha, ndof0);
                [enc endEncoding];
                [cmd commit];
                if (MgCmdWait(cmd) != 0)
                    return -99;
            }
        }

        float nr2 = 0.f;
        dr = MgDotGpu(mb, mb->mgR[0], mb->mgR[0], ndof0, &nr2);
        if (dr != 0)
            return dr;
        if (std::sqrt(std::max(nr2, 0.f)) < tolRel * normB)
            break;

        {
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
                if (cmd == nil)
                    return -99;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil)
                    return -99;
                [enc setComputePipelineState:mb->mgZeroPso];
                [enc setBuffer:mb->mgPcgZ offset:0 atIndex:0];
                [enc setBuffer:mb->mgNDofBuf[0] offset:0 atIndex:1];
                NSUInteger tpg = MbThreadsPerThreadgroup1D(mb->mgZeroPso);
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(ndof0), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                if (MgCmdWait(cmd) != 0)
                    return -99;
            }
        }

        memcpy([mb->mgVcRhs contents], [mb->mgR[0] contents], dof0z);
        {
            const int vz = VCycleMB(mb, 0, numLevels, nSmooth, mb->mgVcRhs, mb->mgPcgZ);
            if (vz != 0)
                return vz;
        }

        float rzNew = 0.f;
        dr = MgDotGpu(mb, mb->mgR[0], mb->mgPcgZ, ndof0, &rzNew);
        if (dr != 0)
            return dr;
        betaGpu = rzNew / (rzOld + 1e-40f);
        rzOld = rzNew;
    }

    memcpy(u_inout, [mb->mgX[0] contents], dof0z);
    if (out_pcg_iters != nullptr)
        *out_pcg_iters = itUsed;
    return 0;
}

int mb_voxel_sample(
    void* ctx,
    const float* ptX,
    const float* ptY,
    const float* ptZ,
    const float* charge,
    float* gridOut,
    float bbMinX,
    float bbMinY,
    float bbMinZ,
    float dxCell,
    float dyCell,
    float dzCell,
    int nx,
    int ny,
    int nz,
    int nPoints,
    float range,
    int linearFalloff,
    int densitySampling)
{
    if (ctx == nullptr || gridOut == nullptr || nx <= 0 || ny <= 0 || nz <= 0 || range <= 0.f)
        return -1;
    if (nPoints > 0 && (ptX == nullptr || ptY == nullptr || ptZ == nullptr || charge == nullptr))
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->voxelSamplePso == nil)
        return -1;

    const int total = nx * ny * nz;
    const NSUInteger gridBytes = static_cast<NSUInteger>(total) * sizeof(float);
    MTLResourceOptions opts = MTLResourceStorageModeShared;

    float z0 = 0.f;
    id<MTLBuffer> bPx;
    id<MTLBuffer> bPy;
    id<MTLBuffer> bPz;
    id<MTLBuffer> bCh;
    if (nPoints > 0) {
        const NSUInteger pb = static_cast<NSUInteger>(nPoints) * sizeof(float);
        bPx = [mb->device newBufferWithBytes:ptX length:pb options:opts];
        bPy = [mb->device newBufferWithBytes:ptY length:pb options:opts];
        bPz = [mb->device newBufferWithBytes:ptZ length:pb options:opts];
        bCh = [mb->device newBufferWithBytes:charge length:pb options:opts];
    } else {
        bPx = [mb->device newBufferWithBytes:&z0 length:sizeof(float) options:opts];
        bPy = [mb->device newBufferWithBytes:&z0 length:sizeof(float) options:opts];
        bPz = [mb->device newBufferWithBytes:&z0 length:sizeof(float) options:opts];
        bCh = [mb->device newBufferWithBytes:&z0 length:sizeof(float) options:opts];
    }

    id<MTLBuffer> bGrid = [mb->device newBufferWithLength:gridBytes options:opts];
    float bbMin[3] = {bbMinX, bbMinY, bbMinZ};
    float cellSz[3] = {dxCell, dyCell, dzCell};
    int dims[3] = {nx, ny, nz};
    int np = nPoints;
    int flg = (linearFalloff ? 1 : 0) | (densitySampling ? 2 : 0);
    id<MTLBuffer> bBbMin = [mb->device newBufferWithBytes:bbMin length:sizeof(bbMin) options:opts];
    id<MTLBuffer> bCell = [mb->device newBufferWithBytes:cellSz length:sizeof(cellSz) options:opts];
    id<MTLBuffer> bDims = [mb->device newBufferWithBytes:dims length:sizeof(dims) options:opts];
    id<MTLBuffer> bNp = [mb->device newBufferWithBytes:&np length:sizeof(np) options:opts];
    id<MTLBuffer> bRange = [mb->device newBufferWithBytes:&range length:sizeof(range) options:opts];
    id<MTLBuffer> bFlags = [mb->device newBufferWithBytes:&flg length:sizeof(flg) options:opts];

    if (bPx == nil || bPy == nil || bPz == nil || bCh == nil || bGrid == nil || bBbMin == nil || bCell == nil
        || bDims == nil || bNp == nil || bRange == nil || bFlags == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->voxelSamplePso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(total);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bPx offset:0 atIndex:0];
        [enc setBuffer:bPy offset:0 atIndex:1];
        [enc setBuffer:bPz offset:0 atIndex:2];
        [enc setBuffer:bCh offset:0 atIndex:3];
        [enc setBuffer:bGrid offset:0 atIndex:4];
        [enc setBuffer:bBbMin offset:0 atIndex:5];
        [enc setBuffer:bCell offset:0 atIndex:6];
        [enc setBuffer:bDims offset:0 atIndex:7];
        [enc setBuffer:bNp offset:0 atIndex:8];
        [enc setBuffer:bRange offset:0 atIndex:9];
        [enc setBuffer:bFlags offset:0 atIndex:10];
        [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(gridOut, [bGrid contents], gridBytes);
    return 0;
}

int mb_proximity_blend(
    void* ctx,
    const float* gradNorm,
    const float* distSL,
    const float* inside,
    float* densityOut,
    const float* params,
    int nx,
    int ny,
    int nz)
{
    if (ctx == nullptr || gradNorm == nullptr || distSL == nullptr || inside == nullptr || densityOut == nullptr
        || params == nullptr)
        return -1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
        return -1;

    auto* mb = static_cast<MBContext*>(ctx);
    if (mb->queue == nil || mb->proximityBlendPso == nil)
        return -1;

    const int total = nx * ny * nz;
    const NSUInteger gridBytes = static_cast<NSUInteger>(total) * sizeof(float);
    const NSUInteger paramBytes = 24u * sizeof(float);
    MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> bGrad = [mb->device newBufferWithBytes:gradNorm length:gridBytes options:opts];
    id<MTLBuffer> bDist = [mb->device newBufferWithBytes:distSL length:gridBytes options:opts];
    id<MTLBuffer> bIn = [mb->device newBufferWithBytes:inside length:gridBytes options:opts];
    id<MTLBuffer> bOut = [mb->device newBufferWithLength:gridBytes options:opts];
    id<MTLBuffer> bPar = [mb->device newBufferWithBytes:params length:paramBytes options:opts];
    id<MTLBuffer> bTot = [mb->device newBufferWithBytes:&total length:sizeof(total) options:opts];
    int dims[3] = {nx, ny, nz};
    id<MTLBuffer> bDims = [mb->device newBufferWithBytes:dims length:sizeof(dims) options:opts];

    if (bGrad == nil || bDist == nil || bIn == nil || bOut == nil || bPar == nil || bTot == nil || bDims == nil)
        return -1;

    id<MTLComputePipelineState> pso = mb->proximityBlendPso;
    const NSUInteger tpg = MbThreadsPerThreadgroup1D(pso);
    const NSUInteger threadCount = static_cast<NSUInteger>(total);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [mb->queue commandBuffer];
        if (cmd == nil)
            return -1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil)
            return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bGrad offset:0 atIndex:0];
        [enc setBuffer:bDist offset:0 atIndex:1];
        [enc setBuffer:bIn offset:0 atIndex:2];
        [enc setBuffer:bOut offset:0 atIndex:3];
        [enc setBuffer:bPar offset:0 atIndex:4];
        [enc setBuffer:bTot offset:0 atIndex:5];
        [enc setBuffer:bDims offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(densityOut, [bOut contents], gridBytes);
    return 0;
}

} // extern "C"
