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

    [DllImport(LibName, EntryPoint = "mb_delaunay_mark_bad_triangles", CallingConvention = CallingConvention.Cdecl)]
    public static extern int DelaunayMarkBadTriangles(
        IntPtr ctx,
        [In] float[] px,
        [In] float[] py,
        int vertexCount,
        [In] int[] triFlat,
        int triCount,
        float queryX,
        float queryY,
        [Out] int[] outBad);

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
}
