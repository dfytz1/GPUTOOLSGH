using System.Numerics;
using SpectralPacking.Core.Disassembly;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Metrics;
using SpectralPacking.Core.Placement;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.Core.Packing;

public sealed class SpectralPackResult
{
    public List<int> PackedIndices { get; } = new();
    public List<Matrix4x4> Rotations { get; } = new();
    public List<Vector3> Translations { get; } = new();
    public List<int> UnpackedIndices { get; } = new();
    public double PackingDensity { get; set; }
    public VoxelGrid FinalOmega { get; set; } = null!;
    public int[] FinalOwner { get; set; } = Array.Empty<int>();
    public bool IsInterlockFree { get; set; }
}

public static class GreedyPacker
{
    public static SpectralPackResult Pack(
        IReadOnlyList<MeshTriangleSoup> meshes,
        AxisAlignedBox trayWorld,
        double voxelSize,
        int orientationCount,
        double gravityWeight,
        bool enableInterlockResolution,
        bool useGpuDistanceField,
        IntPtr metalCtx,
        bool useParallelOrientations,
        IFFTBackend fft,
        OrientationSamplingMode orientationMode,
        int refinementIterations)
    {
        if (meshes.Count == 0)
            throw new ArgumentException("No meshes.");

        double dx = voxelSize;
        int nx = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxX - trayWorld.MinX) / dx));
        int ny = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxY - trayWorld.MinY) / dx));
        int nz = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxZ - trayWorld.MinZ) / dx));

        var omega = VoxelGrid.CreateZero(nx, ny, nz);
        ConservativeVoxelizer.MarkTrayWalls(omega, 1f);
        int nCell = nx * ny * nz;
        var owner = new int[nCell];
        for (int i = 0; i < nCell; i++)
            owner[i] = omega.Data[i] > 0.5f ? -2 : -1;

        var phi = VoxelGrid.CreateZero(nx, ny, nz);
        RefreshPhi(omega, phi, (float)voxelSize, useGpuDistanceField, metalCtx);

        var order = Enumerable.Range(0, meshes.Count)
            .OrderByDescending(i => meshes[i].BoundingBox.Volume)
            .ToList();

        var orientations = OrientationSampler.Sample(orientationCount, orientationMode);

        var result = new SpectralPackResult
        {
            FinalOmega = omega,
            FinalOwner = owner
        };

        foreach (int objIdx in order)
        {
            var mesh = meshes[objIdx];
            var cand = FftPlacementSearch.FindBestPlacement(
                omega, phi, trayWorld, voxelSize, mesh, orientations, fft, gravityWeight, useParallelOrientations);
            if (cand == null)
            {
                result.UnpackedIndices.Add(objIdx);
                continue;
            }

            Vector3 t = cand.TranslationWorld;
            if (refinementIterations > 0)
                ContinuousRefinement.RefineAlongWorldZ(omega, trayWorld, voxelSize, cand, mesh, refinementIterations, ref t);

            FftPlacementSearch.StampCandidate(omega, cand, objIdx, owner);
            RefreshPhi(omega, phi, (float)voxelSize, useGpuDistanceField, metalCtx);

            result.PackedIndices.Add(objIdx);
            result.Rotations.Add(cand.Rotation);
            result.Translations.Add(t);
        }

        double volTray = (trayWorld.MaxX - trayWorld.MinX) * (trayWorld.MaxY - trayWorld.MinY) * (trayWorld.MaxZ - trayWorld.MinZ);
        double volSolid = 0;
        for (int i = 0; i < nCell; i++)
        {
            if (owner[i] >= 0)
                volSolid += dx * dx * dx;
        }

        result.PackingDensity = volTray > 1e-12 ? volSolid / volTray : 0;

        int objectCount = meshes.Count;
        var adj = DirectionalBlockingGraph.BuildAdjacency(nx, ny, nz, owner, objectCount, trayWorld, voxelSize);
        var sccs = DirectionalBlockingGraph.FindSccsTarjan(objectCount, adj);
        result.IsInterlockFree = !sccs.Any(c => c.Count > 1);

        if (enableInterlockResolution && !result.IsInterlockFree)
            ResolveInterlocking(
                meshes, trayWorld, voxelSize, result, fft, orientations, gravityWeight,
                useParallelOrientations, useGpuDistanceField, metalCtx, refinementIterations, objectCount);

        return result;
    }

    private static void RefreshPhi(VoxelGrid omega, VoxelGrid phi, float voxelSize, bool useGpu, IntPtr ctx)
    {
        if (useGpu && ctx != IntPtr.Zero)
            DistanceField.BuildFromSolidGpu(ctx, omega, phi, voxelSize);
        else
            DistanceField.BuildFromSolid(omega, phi, voxelSize);
    }

    private static void ResolveInterlocking(
        IReadOnlyList<MeshTriangleSoup> meshes,
        AxisAlignedBox trayWorld,
        double voxelSize,
        SpectralPackResult result,
        IFFTBackend fft,
        IReadOnlyList<Matrix4x4> orientations,
        double gravityWeight,
        bool useParallel,
        bool useGpuDf,
        IntPtr metalCtx,
        int refinementIterations,
        int objectCount)
    {
        int nx = result.FinalOmega.Width, ny = result.FinalOmega.Height, nz = result.FinalOmega.Depth;
        var adj = DirectionalBlockingGraph.BuildAdjacency(nx, ny, nz, result.FinalOwner, objectCount, trayWorld, voxelSize);
        var sccs = DirectionalBlockingGraph.FindSccsTarjan(objectCount, adj);
        var removeIds = new List<int>();
        foreach (var comp in sccs)
        {
            if (comp.Count < 2)
                continue;
            int best = comp.OrderBy(id => meshes[id].BoundingBox.Volume).First();
            removeIds.Add(best);
        }

        foreach (int rm in removeIds)
        {
            int pk = result.PackedIndices.IndexOf(rm);
            if (pk < 0)
                continue;
            result.PackedIndices.RemoveAt(pk);
            result.Rotations.RemoveAt(pk);
            result.Translations.RemoveAt(pk);
            ClearObjectVoxels(result.FinalOmega, result.FinalOwner, rm);
            if (!result.UnpackedIndices.Contains(rm))
                result.UnpackedIndices.Add(rm);
        }

        RebuildOmegaAndOwner(result, meshes, trayWorld, voxelSize, nx, ny, nz);
        var phi = VoxelGrid.CreateZero(nx, ny, nz);
        RefreshPhi(result.FinalOmega, phi, (float)voxelSize, useGpuDf, metalCtx);

        foreach (int rm in removeIds.OrderByDescending(id => meshes[id].BoundingBox.Volume))
        {
            var mesh = meshes[rm];
            var cand = FftPlacementSearch.FindBestPlacement(
                result.FinalOmega, phi, trayWorld, voxelSize, mesh, orientations, fft, gravityWeight, useParallel);
            if (cand == null)
                continue;

            Vector3 t = cand.TranslationWorld;
            if (refinementIterations > 0)
                ContinuousRefinement.RefineAlongWorldZ(result.FinalOmega, trayWorld, voxelSize, cand, mesh, refinementIterations, ref t);

            FftPlacementSearch.StampCandidate(result.FinalOmega, cand, rm, result.FinalOwner);
            RefreshPhi(result.FinalOmega, phi, (float)voxelSize, useGpuDf, metalCtx);
            result.PackedIndices.Add(rm);
            result.Rotations.Add(cand.Rotation);
            result.Translations.Add(t);
            result.UnpackedIndices.Remove(rm);
        }

        adj = DirectionalBlockingGraph.BuildAdjacency(nx, ny, nz, result.FinalOwner, objectCount, trayWorld, voxelSize);
        sccs = DirectionalBlockingGraph.FindSccsTarjan(objectCount, adj);
        result.IsInterlockFree = !sccs.Any(c => c.Count > 1);
    }

    private static void ClearObjectVoxels(VoxelGrid omega, int[] owner, int objectId)
    {
        for (int i = 0; i < owner.Length; i++)
        {
            if (owner[i] == objectId)
            {
                owner[i] = -1;
                omega.Data[i] = 0f;
            }
        }
    }

    private static void RebuildOmegaAndOwner(
        SpectralPackResult result,
        IReadOnlyList<MeshTriangleSoup> meshes,
        AxisAlignedBox tray,
        double voxelSize,
        int nx,
        int ny,
        int nz)
    {
        var omega = result.FinalOmega;
        var owner = result.FinalOwner;
        int n = omega.LinearSize;
        for (int i = 0; i < n; i++)
        {
            omega.Data[i] = 0f;
            owner[i] = -1;
        }

        ConservativeVoxelizer.MarkTrayWalls(omega, 1f);
        for (int i = 0; i < n; i++)
        {
            if (omega.Data[i] > 0.5f)
                owner[i] = -2;
        }

        for (int k = 0; k < result.PackedIndices.Count; k++)
        {
            int id = result.PackedIndices[k];
            var R = result.Rotations[k];
            var t = result.Translations[k];
            var meshR = meshes[id].RotatedAboutCentroid(
                R.M11, R.M12, R.M13,
                R.M21, R.M22, R.M23,
                R.M31, R.M32, R.M33);
            int nv = meshR.VertexCount;
            var wx = new double[nv];
            var wy = new double[nv];
            var wz = new double[nv];
            for (int i = 0; i < nv; i++)
            {
                wx[i] = meshR.Vx[i] + t.X;
                wy[i] = meshR.Vy[i] + t.Y;
                wz[i] = meshR.Vz[i] + t.Z;
            }

            var moved = new MeshTriangleSoup(wx, wy, wz, meshR.TriangleIndices);
            var mask = ConservativeVoxelizer.VoxelizeMesh(moved, tray, voxelSize, fillSixWalls: false);
            for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                if (mask[x, y, z] <= 0.5f)
                    continue;
                int idx = omega.Index(x, y, z);
                if (owner[idx] == -2)
                    continue;
                omega.Data[idx] = 1f;
                owner[idx] = id;
            }
        }
    }
}
