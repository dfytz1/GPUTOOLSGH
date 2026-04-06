using System.Runtime.InteropServices;

namespace GHGPUPlugin.NativeInterop;

public static class MetalBridge
{
    private const string LibName = "MetalBridge";

    [DllImport(LibName, EntryPoint = "mb_create_context", CallingConvention = CallingConvention.Cdecl)]
    public static extern int CreateContext(out IntPtr ctx);

    [DllImport(LibName, EntryPoint = "mb_destroy_context", CallingConvention = CallingConvention.Cdecl)]
    public static extern void DestroyContext(IntPtr ctx);

    [DllImport(LibName, EntryPoint = "mb_run_benchmark", CallingConvention = CallingConvention.Cdecl)]
    public static extern int RunBenchmark(
        IntPtr ctx,
        IntPtr buffer,
        int elementCount,
        int innerIters,
        int outerIters);

    [DllImport(LibName, EntryPoint = "mb_run_laplacian", CallingConvention = CallingConvention.Cdecl)]
    public static extern int RunLaplacian(
        IntPtr ctx,
        [In] float[] posXIn,
        [In] float[] posYIn,
        [In] float[] posZIn,
        [Out] float[] posXOut,
        [Out] float[] posYOut,
        [Out] float[] posZOut,
        [In] int[] adjFlat,
        [In] int[] rowOffsets,
        int vertexCount,
        float strength);

    /// <summary>Runs <paramref name="iterations"/> Laplacian steps in one GPU submit; overwrites SoA position buffers.</summary>
    [DllImport(LibName, EntryPoint = "mb_run_laplacian_iterations", CallingConvention = CallingConvention.Cdecl)]
    public static extern int RunLaplacianIterations(
        IntPtr ctx,
        [In, Out] float[] posX,
        [In, Out] float[] posY,
        [In, Out] float[] posZ,
        [In] int[] adjFlat,
        [In] int[] rowOffsets,
        int vertexCount,
        float strength,
        int iterations);

    [DllImport(LibName, EntryPoint = "mb_run_laplacian_constrained", CallingConvention = CallingConvention.Cdecl)]
    public static extern int RunLaplacianConstrained(
        IntPtr ctx,
        [In, Out] float[] posX,
        [In, Out] float[] posY,
        [In, Out] float[] posZ,
        [In] int[] adjFlat,
        [In] int[] rowOffsets,
        int vertexCount,
        float strength,
        int iterations,
        [In] byte[] fixedMask);

    [DllImport(LibName, EntryPoint = "mb_fem_matvec", CallingConvention = CallingConvention.Cdecl)]
    public static extern int FemMatVec(
        IntPtr ctx,
        [In] float[] Ke_flat,
        [In] int[] dofMap,
        [In] float[] rho,
        [In] float[] v_in,
        [Out] float[] Av_out,
        [In] byte[]? fixedMask,
        float penalty,
        int nElem,
        int ndof);

    [DllImport(LibName, EntryPoint = "mb_fem_pcg_solve", CallingConvention = CallingConvention.Cdecl)]
    public static extern int FemPcgSolve(
        IntPtr ctx,
        [In] float[] Ke_flat,
        [In] int[] dofMap,
        [In] byte[] fixedMask,
        [In] float[] rho,
        [In] float[] diag,
        [In] float[] f_rhs,
        [In, Out] float[] u_inout,
        float penalty,
        int nElem,
        int ndof,
        int maxIter,
        float tolRel);

    [DllImport(LibName, EntryPoint = "mb_voxel_sample", CallingConvention = CallingConvention.Cdecl)]
    public static extern int VoxelSample(
        IntPtr ctx,
        [In] float[] ptX,
        [In] float[] ptY,
        [In] float[] ptZ,
        [In] float[] charge,
        [Out] float[] gridOut,
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
        int densitySampling);

    [DllImport(LibName, EntryPoint = "mb_proximity_blend", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ProximityBlend(
        IntPtr ctx,
        [In] float[] gradNorm,
        [In] float[] distSL,
        [In] float[] inside,
        [Out] float[] densityOut,
        [In] float[] proximityParams,
        int nx,
        int ny,
        int nz);

    [DllImport(LibName, EntryPoint = "mb_closest_points_cloud", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ClosestPointsCloud(
        IntPtr ctx,
        [In] float[] qx,
        [In] float[] qy,
        [In] float[] qz,
        int queryCount,
        [In] float[] px,
        [In] float[] py,
        [In] float[] pz,
        int targetCount,
        [Out] float[] outCx,
        [Out] float[] outCy,
        [Out] float[] outCz,
        [Out] float[] outDistSq,
        [Out] int[] outIndex);

    [DllImport(LibName, EntryPoint = "mb_closest_points_mesh", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ClosestPointsMesh(
        IntPtr ctx,
        [In] float[] qx,
        [In] float[] qy,
        [In] float[] qz,
        int queryCount,
        [In] float[] vx,
        [In] float[] vy,
        [In] float[] vz,
        int vertexCount,
        [In] int[] triIndices,
        int triangleCount,
        [Out] float[] outCx,
        [Out] float[] outCy,
        [Out] float[] outCz,
        [Out] float[] outDistSq,
        [Out] int[] outTriIndex);

    [DllImport(LibName, EntryPoint = "mb_jfa_delaunay_2d", CallingConvention = CallingConvention.Cdecl)]
    public static extern int JfaDelaunay2D(
        IntPtr ctx,
        [In] float[] px,
        [In] float[] py,
        int pointCount,
        [Out] int[] outEdgeA,
        [Out] int[] outEdgeB,
        out int outEdgeCount,
        int maxEdges,
        int gridResolution);

    [DllImport(LibName, EntryPoint = "mb_build_weighted_edges_csr", CallingConvention = CallingConvention.Cdecl)]
    public static extern int BuildWeightedEdgesCsr(
        IntPtr ctx,
        [In] float[] vx,
        [In] float[] vy,
        [In] float[] vz,
        int vertexCount,
        [In] int[] rowOffsets,
        [In] int[] adjFlat,
        [In] int[] edgeWriteBase,
        int edgeCount,
        [Out] int[] edgeU,
        [Out] int[] edgeV,
        [Out] float[] edgeW);

    [DllImport(LibName, EntryPoint = "mb_laplace_jacobi_3d", CallingConvention = CallingConvention.Cdecl)]
    public static extern int LaplaceJacobi3D(
        IntPtr ctx,
        [In] float[] inside,
        [In] float[] support,
        [In] float[] load,
        [In, Out] float[] phi,
        int nx,
        int ny,
        int nz,
        float supportVal,
        float loadVal,
        int iterations);

    [DllImport(LibName, EntryPoint = "mb_gradient_magnitude_3d", CallingConvention = CallingConvention.Cdecl)]
    public static extern int GradientMagnitude3D(
        IntPtr ctx,
        [In] float[] phi,
        [In] float[] inside,
        [Out] float[] gradOut,
        int nx,
        int ny,
        int nz,
        float invDx,
        float invDy,
        float invDz);

    [DllImport(LibName, EntryPoint = "mb_normalize_contrast_3d", CallingConvention = CallingConvention.Cdecl)]
    public static extern int NormalizeContrast3D(
        IntPtr ctx,
        [In, Out] float[] dataInOut,
        [In] float[] inside,
        int nx,
        int ny,
        int nz,
        float domainMin,
        float domainMax,
        int invert,
        float exponent);

    [DllImport(LibName, EntryPoint = "mb_zero_voxel_boundary", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ZeroVoxelBoundary(
        IntPtr ctx,
        [In, Out] float[] data,
        int nx,
        int ny,
        int nz);
}
