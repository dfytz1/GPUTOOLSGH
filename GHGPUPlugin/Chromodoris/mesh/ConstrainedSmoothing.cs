/*
 * Laplacian smoothing with per-vertex locks from voxel support/load masks (Workflow A).
 */

using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GHGPUPlugin.Chromodoris.MeshTools
{
    /// <summary>
    /// Maps world points to voxel indices consistent with Build IsoSurface TransformMeshToBox (cell-centered or corner grid).
    /// </summary>
    internal static class VoxelWorldMapping
    {
        /// <summary>Normalized tx,ty,tz in [0,1] along box AABB.</summary>
        public static void WorldToUnit(Point3d w, BoundingBox bb, out double tx, out double ty, out double tz)
        {
            double sx = bb.Max.X - bb.Min.X;
            double sy = bb.Max.Y - bb.Min.Y;
            double sz = bb.Max.Z - bb.Min.Z;
            if (sx < 1e-30) sx = 1;
            if (sy < 1e-30) sy = 1;
            if (sz < 1e-30) sz = 1;
            tx = (w.X - bb.Min.X) / sx;
            ty = (w.Y - bb.Min.Y) / sy;
            tz = (w.Z - bb.Min.Z) / sz;
            tx = Rhino.RhinoMath.Clamp(tx, 0.0, 1.0);
            ty = Rhino.RhinoMath.Clamp(ty, 0.0, 1.0);
            tz = Rhino.RhinoMath.Clamp(tz, 0.0, 1.0);
        }

        /// <summary>Inverse of cell-centered iso map: tx = (vX+0.5)/nx.</summary>
        public static void UnitToVoxelCellCentered(double tx, double ty, double tz, int nx, int ny, int nz,
            out int ix, out int iy, out int iz)
        {
            ix = Math.Min(nx - 1, Math.Max(0, (int)Math.Floor(tx * nx)));
            iy = Math.Min(ny - 1, Math.Max(0, (int)Math.Floor(ty * ny)));
            iz = Math.Min(nz - 1, Math.Max(0, (int)Math.Floor(tz * nz)));
        }

        /// <summary>Inverse of corner-grid iso: tx = vX/denom.</summary>
        public static void UnitToVoxelCorner(double tx, double ty, double tz, int nx, int ny, int nz,
            out int ix, out int iy, out int iz)
        {
            double dx = Math.Max(1, nx - 1);
            double dy = Math.Max(1, ny - 1);
            double dz = Math.Max(1, nz - 1);
            ix = Math.Min(nx - 1, Math.Max(0, (int)Math.Floor(tx * dx + 0.5)));
            iy = Math.Min(ny - 1, Math.Max(0, (int)Math.Floor(ty * dy + 0.5)));
            iz = Math.Min(nz - 1, Math.Max(0, (int)Math.Floor(tz * dz + 0.5)));
        }
    }

    public sealed class ConstrainedVertexSmooth
    {
        private readonly int iterations;
        private readonly double step;
        private readonly Mesh mesh;
        private readonly bool[] constrained;
        private readonly List<int[]> neighbourVerts;
        private Point3f[] topoVertLocations;
        private readonly List<int[]> topoVertexIndices;

        /// <param name="constrainedPerTopology">Length = mesh.TopologyVertices.Count; true = do not move.</param>
        public ConstrainedVertexSmooth(Mesh mesh, double step, int iterations, bool[] constrainedPerTopology)
        {
            this.mesh = mesh;
            this.step = step;
            this.iterations = iterations;
            constrained = constrainedPerTopology ?? throw new ArgumentNullException(nameof(constrainedPerTopology));
            if (constrained.Length != mesh.TopologyVertices.Count)
                throw new ArgumentException("constrained length must match topology vertex count.");

            neighbourVerts = new List<int[]>();
            topoVertLocations = new Point3f[mesh.TopologyVertices.Count];
            topoVertexIndices = new List<int[]>();
        }

        public static bool[] BuildConstraints(Mesh mesh, Box box, int nx, int ny, int nz,
            float[,,] support, float[,,] load, bool cellCentered, bool fixSupport, bool fixLoad, int dilateRings)
        {
            int nTopo = mesh.TopologyVertices.Count;
            var flags = new bool[nTopo];
            if (!fixSupport && !fixLoad)
                return flags;

            BoundingBox bb = box.BoundingBox;
            if (!bb.IsValid)
                return flags;

            bool[,,] raw = new bool[nx, ny, nz];
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                    {
                        bool s = fixSupport && support[i, j, k] >= 0.5f;
                        bool l = fixLoad && load[i, j, k] >= 0.5f;
                        raw[i, j, k] = s || l;
                    }

            bool[,,] mask = DilateBool(raw, nx, ny, nz, Math.Max(0, dilateRings));

            for (int ti = 0; ti < nTopo; ti++)
            {
                int[] mvInds = mesh.TopologyVertices.MeshVertexIndices(ti);
                Point3d w = mesh.Vertices[mvInds[0]];

                VoxelWorldMapping.WorldToUnit(w, bb, out double tx, out double ty, out double tz);

                int ix, iy, iz;
                if (cellCentered)
                    VoxelWorldMapping.UnitToVoxelCellCentered(tx, ty, tz, nx, ny, nz, out ix, out iy, out iz);
                else
                    VoxelWorldMapping.UnitToVoxelCorner(tx, ty, tz, nx, ny, nz, out ix, out iy, out iz);

                flags[ti] = mask[ix, iy, iz];
            }

            return flags;
        }

        private static bool[,,] DilateBool(bool[,,] src, int nx, int ny, int nz, int rings)
        {
            if (rings <= 0)
                return src;

            bool[,,] cur = new bool[nx, ny, nz];
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        cur[i, j, k] = src[i, j, k];

            for (int r = 0; r < rings; r++)
            {
                var next = new bool[nx, ny, nz];
                for (int i = 0; i < nx; i++)
                {
                    for (int j = 0; j < ny; j++)
                    {
                        for (int k = 0; k < nz; k++)
                        {
                            if (cur[i, j, k])
                            {
                                next[i, j, k] = true;
                                continue;
                            }

                            bool any = false;
                            for (int di = -1; di <= 1 && !any; di++)
                                for (int dj = -1; dj <= 1 && !any; dj++)
                                    for (int dk = -1; dk <= 1 && !any; dk++)
                                    {
                                        if (di == 0 && dj == 0 && dk == 0) continue;
                                        int ii = i + di, jj = j + dj, kk = k + dk;
                                        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz && cur[ii, jj, kk])
                                            any = true;
                                    }
                            next[i, j, k] = any;
                        }
                    }
                }
                cur = next;
            }

            return cur;
        }

        public Mesh Compute()
        {
            for (int i = 0; i < mesh.TopologyVertices.Count; i++)
            {
                int[] mvInds = mesh.TopologyVertices.MeshVertexIndices(i);
                topoVertLocations[i] = mesh.Vertices[mvInds[0]];
                topoVertexIndices.Add(mvInds);
                neighbourVerts.Add(mesh.TopologyVertices.ConnectedTopologyVertices(i));
            }

            for (int it = 0; it < iterations; it++)
                SmoothMultiThread();

            Point3f[] mVerts = new Point3f[mesh.Vertices.Count];
            for (int i = 0; i < topoVertLocations.Length; i++)
            {
                Point3f loc = topoVertLocations[i];
                foreach (int vInd in topoVertexIndices[i])
                    mVerts[vInd] = loc;
            }

            Mesh newMesh = new Mesh();
            newMesh.Vertices.AddVertices(mVerts);
            newMesh.Faces.AddFaces(mesh.Faces);
            return newMesh;
        }

        private void SmoothMultiThread()
        {
            var options = new System.Threading.Tasks.ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
            System.Threading.Tasks.Parallel.ForEach(
                Enumerable.Range(0, mesh.TopologyVertices.Count),
                options,
                SmoothTopoIndex);
        }

        private void SmoothTopoIndex(int v)
        {
            if (constrained[v]) return;

            Point3d loc = topoVertLocations[v];
            int[] nvs = neighbourVerts[v];
            if (nvs.Length == 0) return;

            Point3d avg = new Point3d();
            foreach (int nv in nvs)
                avg += topoVertLocations[nv];
            avg /= nvs.Length;

            Vector3d pos = new Vector3d(loc) + (avg - loc) * step;
            topoVertLocations[v] = new Point3f((float)pos.X, (float)pos.Y, (float)pos.Z);
        }
    }
}
