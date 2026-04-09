using System.Numerics;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Packing;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.Core.Placement;

/// <summary>Sub-voxel adjustment along tray vertical (Z) using occupancy re-tests.</summary>
public static class ContinuousRefinement
{
    public static void RefineAlongWorldZ(
        VoxelGrid omega,
        AxisAlignedBox trayWorld,
        double voxelSize,
        SpectralPlacementCandidate candidate,
        MeshTriangleSoup meshWorld,
        int iterations,
        ref Vector3 translationWorld)
    {
        if (iterations <= 0)
            return;

        double zMin = trayWorld.MinZ + 1e-6;
        double zMax = trayWorld.MaxZ - 1e-6;

        for (int it = 0; it < iterations; it++)
        {
            double step = voxelSize * 0.5;
            Vector3 up = new(0, 0, (float)step);
            Vector3 down = new(0, 0, -(float)step);

            if (TryShift(omega, trayWorld, voxelSize, candidate, meshWorld, translationWorld + up, zMin, zMax))
                translationWorld += up;
            else if (TryShift(omega, trayWorld, voxelSize, candidate, meshWorld, translationWorld + down, zMin, zMax))
                translationWorld += down;
            else
                break;
        }
    }

    private static bool TryShift(
        VoxelGrid omega,
        AxisAlignedBox trayWorld,
        double voxelSize,
        SpectralPlacementCandidate candidate,
        MeshTriangleSoup meshWorld,
        Vector3 newTranslation,
        double zMin,
        double zMax)
    {
        var R = candidate.Rotation;
        var meshT = meshWorld.RotatedAboutCentroid(
            R.M11, R.M12, R.M13,
            R.M21, R.M22, R.M23,
            R.M31, R.M32, R.M33);
        int n = meshT.VertexCount;
        var tx = new double[n];
        var ty = new double[n];
        var tz = new double[n];
        for (int i = 0; i < n; i++)
        {
            tx[i] = meshT.Vx[i] + newTranslation.X;
            ty[i] = meshT.Vy[i] + newTranslation.Y;
            tz[i] = meshT.Vz[i] + newTranslation.Z;
            if (tz[i] < zMin || tz[i] > zMax)
                return false;
        }

        var moved = new MeshTriangleSoup(tx, ty, tz, meshT.TriangleIndices);
        var probe = ConservativeVoxelizer.VoxelizeMesh(moved, trayWorld, voxelSize, fillSixWalls: false);
        int nx = omega.Width, ny = omega.Height, nz = omega.Depth;
        for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            if (probe[x, y, z] <= 0.5f)
                continue;
            if (omega[x, y, z] > 0.5f)
                return false;
        }

        return true;
    }
}
