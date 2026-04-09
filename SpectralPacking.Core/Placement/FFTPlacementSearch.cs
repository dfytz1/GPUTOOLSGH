using System.Numerics;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Metrics;
using SpectralPacking.Core.Packing;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.Core.Placement;

public static class FftPlacementSearch
{
    private const float CollisionEps = 1e-4f;

    /// <summary>Spectral placement (Algorithm 1): minimize ρ with ζ = 0 over orientations and lattice translations.</summary>
    public static SpectralPlacementCandidate? FindBestPlacement(
        VoxelGrid omega,
        VoxelGrid phi,
        AxisAlignedBox trayWorld,
        double voxelSize,
        MeshTriangleSoup meshWorld,
        IReadOnlyList<Matrix4x4> orientations,
        IFFTBackend fft,
        double gravityWeight,
        bool useParallel)
    {
        int nx = omega.Width, ny = omega.Height, nz = omega.Depth;

        object gate = new();
        float bestLocal = float.PositiveInfinity;
        SpectralPlacementCandidate? bestCand = null;

        void Body(int oi)
        {
            if (meshWorld.VertexCount == 0)
                return;

            Matrix4x4 R = orientations[oi];
            var meshR = meshWorld.RotatedAboutCentroid(
                R.M11, R.M12, R.M13,
                R.M21, R.M22, R.M23,
                R.M31, R.M32, R.M33);

            var bb = meshR.BoundingBox;
            double cornerX = trayWorld.MinX + Math.Floor((bb.MinX - trayWorld.MinX) / voxelSize) * voxelSize;
            double cornerY = trayWorld.MinY + Math.Floor((bb.MinY - trayWorld.MinY) / voxelSize) * voxelSize;
            double cornerZ = trayWorld.MinZ + Math.Floor((bb.MinZ - trayWorld.MinZ) / voxelSize) * voxelSize;

            int sx = Math.Max(1, (int)Math.Ceiling((bb.MaxX - cornerX) / voxelSize));
            int sy = Math.Max(1, (int)Math.Ceiling((bb.MaxY - cornerY) / voxelSize));
            int sz = Math.Max(1, (int)Math.Ceiling((bb.MaxZ - cornerZ) / voxelSize));

            if (sx > nx || sy > ny || sz > nz)
                return;

            var localTray = new AxisAlignedBox(cornerX, cornerY, cornerZ,
                cornerX + sx * voxelSize,
                cornerY + sy * voxelSize,
                cornerZ + sz * voxelSize);

            var localGrid = ConservativeVoxelizer.VoxelizeMesh(meshR, localTray, voxelSize, fillSixWalls: false);
            if (localGrid.Width != sx || localGrid.Height != sy || localGrid.Depth != sz)
                return;

            int ppx = FftGridDims.NextPow2(nx + sx - 1);
            int ppy = FftGridDims.NextPow2(ny + sy - 1);
            int ppz = FftGridDims.NextPow2(nz + sz - 1);
            int ppn = ppx * ppy * ppz;
            var padA = new float[ppn];
            var padO = new float[ppn];
            var padP = new float[ppn];
            CopyToPad(localGrid.Data, sx, sy, sz, padA, ppx, ppy, ppz);
            CopyToPad(omega.Data, nx, ny, nz, padO, ppx, ppy, ppz);
            CopyToPad(phi.Data, nx, ny, nz, padP, ppx, ppy, ppz);

            var z = new float[ppn];
            var r = new float[ppn];
            CollisionMetric.Compute(fft, padA, padO, ppx, ppy, ppz, z);
            ProximityMetric.Compute(fft, padA, padP, ppx, ppy, ppz, r);

            float bestOri = float.PositiveInfinity;
            int btx = 0, bty = 0, btz = 0;

            for (int tz = 0; tz <= nz - sz; tz++)
            for (int ty = 0; ty <= ny - sy; ty++)
            for (int tx = 0; tx <= nx - sx; tx++)
            {
                long idxL = (long)tx + (long)ty * ppx + (long)tz * ppx * ppy;
                if (idxL < 0 || idxL >= ppn)
                    continue;
                int idx = (int)idxL;
                if (z[idx] > CollisionEps)
                    continue;
                float g = (float)(gravityWeight * tz * voxelSize);
                float s = r[idx] + g;
                if (s < bestOri)
                {
                    bestOri = s;
                    btx = tx;
                    bty = ty;
                    btz = tz;
                }
            }

            if (bestOri >= float.PositiveInfinity / 2)
                return;

            var tw = ComputeTranslationWorld(meshR, trayWorld, voxelSize, btx, bty, btz);
            var loc = (float[])localGrid.Data.Clone();
            var cand = new SpectralPlacementCandidate
            {
                Rotation = R,
                Tx = btx,
                Ty = bty,
                Tz = btz,
                Sx = sx,
                Sy = sy,
                Sz = sz,
                LocalOccupancy = loc,
                TranslationWorld = tw,
                Score = bestOri
            };

            lock (gate)
            {
                if (bestOri < bestLocal)
                {
                    bestLocal = bestOri;
                    bestCand = cand;
                }
            }
        }

        if (useParallel)
            System.Threading.Tasks.Parallel.For(0, orientations.Count, Body);
        else
        {
            for (int oi = 0; oi < orientations.Count; oi++)
                Body(oi);
        }

        return bestCand;
    }

    public static void StampCandidate(VoxelGrid omega, SpectralPlacementCandidate c, int objectIndex, int[] owner)
    {
        int nx = omega.Width, ny = omega.Height, nz = omega.Depth;
        int sx = c.Sx, sy = c.Sy, sz = c.Sz;
        for (int lz = 0; lz < sz; lz++)
        for (int ly = 0; ly < sy; ly++)
        for (int lx = 0; lx < sx; lx++)
        {
            int li = lx + ly * sx + lz * sx * sy;
            if (c.LocalOccupancy[li] <= 0.5f)
                continue;
            int ix = c.Tx + lx;
            int iy = c.Ty + ly;
            int iz = c.Tz + lz;
            if ((uint)ix >= (uint)nx || (uint)iy >= (uint)ny || (uint)iz >= (uint)nz)
                continue;
            omega[ix, iy, iz] = 1f;
            int gi = omega.Index(ix, iy, iz);
            owner[gi] = objectIndex;
        }
    }

    private static Vector3 ComputeTranslationWorld(
        MeshTriangleSoup meshR,
        AxisAlignedBox trayWorld,
        double voxelSize,
        int tx, int ty, int tz)
    {
        if (meshR.VertexCount == 0)
            return Vector3.Zero;

        double brMinX = meshR.Vx[0], brMinY = meshR.Vy[0], brMinZ = meshR.Vz[0];
        for (int i = 1; i < meshR.VertexCount; i++)
        {
            brMinX = Math.Min(brMinX, meshR.Vx[i]);
            brMinY = Math.Min(brMinY, meshR.Vy[i]);
            brMinZ = Math.Min(brMinZ, meshR.Vz[i]);
        }

        double cornerX = trayWorld.MinX + tx * voxelSize;
        double cornerY = trayWorld.MinY + ty * voxelSize;
        double cornerZ = trayWorld.MinZ + tz * voxelSize;

        return new Vector3(
            (float)(cornerX - brMinX),
            (float)(cornerY - brMinY),
            (float)(cornerZ - brMinZ));
    }

    private static void CopyToPad(ReadOnlySpan<float> src, int nx, int ny, int nz, float[] dst, int px, int py, int pz)
    {
        Array.Clear(dst);
        for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            int si = x + y * nx + z * nx * ny;
            int di = x + y * px + z * px * py;
            dst[di] = src[si];
        }
    }
}
