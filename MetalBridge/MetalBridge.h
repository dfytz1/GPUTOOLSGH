#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @return 0 on success, negative error code on failure.
int mb_create_context(void** outCtx);

void mb_destroy_context(void* ctx);

/// In-place benchmark on @p buffer (length @p elementCount).
int mb_run_benchmark(void* ctx, float* buffer, int elementCount, int innerIters, int outerIters);

/// One umbrella Laplacian smoothing step on topology-vertex positions (SoA). Buffers are host pointers; data is copied.
/// @p rowOffsets length @p vertexCount + 1.
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
    float strength);

/// Same Laplacian as @ref mb_run_laplacian but runs @p iterations steps on-GPU with one command-buffer submit (ping-pong buffers).
/// @p posX/Y/Z are read as initial state and overwritten with the final state.
int mb_run_laplacian_iterations(
    void* ctx,
    float* posX,
    float* posY,
    float* posZ,
    int* adjFlat,
    int* rowOffsets,
    int vertexCount,
    float strength,
    int iterations);

/// Closest point on triangle mesh (brute force per query). @p triIndices length @p triangleCount * 3 (vertex indices per triangle).
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
    int* outTriIndex);

/// For each query, closest target point (brute force). @p outIndex is index into target set (-1 if targetCount==0).
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
    int* outIndex);

/// Mark Delaunay triangles whose circumcircle contains (@p queryX, @p queryY). @p triFlat length @p triCount * 3.
int mb_delaunay_mark_bad_triangles(
    void* ctx,
    float* px,
    float* py,
    int vertexCount,
    int* triFlat,
    int triCount,
    float queryX,
    float queryY,
    int* outBad);

/// Fill directed edges with Euclidean 3D lengths; @p edgeCount must equal @p rowOffsets[vertexCount]. @p edgeWriteBase[v] is write offset (typically @p rowOffsets).
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
    float* edgeW);

#ifdef __cplusplus
}
#endif
