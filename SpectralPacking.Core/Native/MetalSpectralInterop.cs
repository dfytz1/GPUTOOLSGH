using System.Runtime.InteropServices;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.Core.Native;

/// <summary>P/Invoke extensions on MetalBridge for spectral packing (voxel columns, DF BFS).</summary>
public static class MetalSpectralInterop
{
    private const string LibName = "MetalBridge";

    [DllImport(LibName, EntryPoint = "mb_spectral_voxel_columns", CallingConvention = CallingConvention.Cdecl)]
    public static extern int SpectralVoxelColumns(
        IntPtr ctx,
        [In] float[] vx,
        [In] float[] vy,
        [In] float[] vz,
        int vertexCount,
        [In] int[] tri,
        int triCount,
        float bbMinX,
        float bbMinY,
        float bbMinZ,
        float dx,
        float dy,
        float dz,
        int nx,
        int ny,
        int nz,
        [Out] byte[] gridOut);

    [DllImport(LibName, EntryPoint = "mb_spectral_df_bfs", CallingConvention = CallingConvention.Cdecl)]
    public static extern int SpectralDfBfs(
        IntPtr ctx,
        [In] byte[] solid,
        [In, Out] float[] phi,
        int nx,
        int ny,
        int nz,
        float voxelSize);

    /// <returns>0 if native path ran successfully.</returns>
    public static int TryDistanceFieldBfs(IntPtr ctx, VoxelGrid solidBinary, VoxelGrid phiOut, float voxelSize)
    {
        if (ctx == IntPtr.Zero)
            return -1;
        int n = solidBinary.LinearSize;
        var solid = new byte[n];
        for (int i = 0; i < n; i++)
            solid[i] = solidBinary.Data[i] > 0.5f ? (byte)1 : (byte)0;
        var phi = phiOut.Data;
        return SpectralDfBfs(ctx, solid, phi, solidBinary.Width, solidBinary.Height, solidBinary.Depth, voxelSize);
    }

    /// <returns>0 on success.</returns>
    public static int TryVoxelColumns(
        IntPtr ctx,
        float[] vx, float[] vy, float[] vz,
        int[] tri,
        float bbMinX, float bbMinY, float bbMinZ,
        float dx, float dy, float dz,
        int nx, int ny, int nz,
        byte[] gridOut)
    {
        if (ctx == IntPtr.Zero)
            return -1;
        return SpectralVoxelColumns(ctx, vx, vy, vz, vx.Length, tri, tri.Length / 3, bbMinX, bbMinY, bbMinZ, dx, dy, dz, nx, ny, nz, gridOut);
    }
}
