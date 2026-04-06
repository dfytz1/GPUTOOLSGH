using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace GHGPUPlugin.Chromodoris.Topology
{
    /// <summary>
    /// Workflow A: voxel grid aligned to an axis-aligned box, Laplace field, gradient density.
    /// </summary>
    internal static class WorkflowAGrid
    {
        public static void BuildGridFromMesh(Mesh mesh, double voxelSize,
            out Box box, out int nx, out int ny, out int nz, out double dx, out double dy, out double dz)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            if (voxelSize <= 0) throw new ArgumentException("Voxel size must be positive.", nameof(voxelSize));

            BoundingBox bb = mesh.GetBoundingBox(true);
            if (!bb.IsValid) throw new InvalidOperationException("Mesh bounding box is invalid.");

            dx = dy = dz = voxelSize;
            double sx = bb.Max.X - bb.Min.X;
            double sy = bb.Max.Y - bb.Min.Y;
            double sz = bb.Max.Z - bb.Min.Z;

            nx = Math.Max(2, (int)Math.Ceiling(sx / dx));
            ny = Math.Max(2, (int)Math.Ceiling(sy / dy));
            nz = Math.Max(2, (int)Math.Ceiling(sz / dz));

            var plane = Plane.WorldXY;
            plane.Origin = bb.Min;
            var xInt = new Interval(0, nx * dx);
            var yInt = new Interval(0, ny * dy);
            var zInt = new Interval(0, nz * dz);
            box = new Box(plane, xInt, yInt, zInt);
        }

        public static Point3d CellCenterWorld(Box box, int i, int j, int k, int nx, int ny, int nz)
        {
            double tx = (i + 0.5) / nx;
            double ty = (j + 0.5) / ny;
            double tz = (k + 0.5) / nz;
            return box.PointAt(tx, ty, tz);
        }

        public static float[,,] VoxelizeMeshInside(Mesh mesh, Box box, int nx, int ny, int nz, float insideValue = 1f)
        {
            var data = new float[nx, ny, nz];
            if (!mesh.IsClosed)
                return data;
            if (insideValue == 0f)
                return data;

            Parallel.For(0, nx, i =>
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        Point3d c = CellCenterWorld(box, i, j, k, nx, ny, nz);
                        if (mesh.IsPointInside(c, Rhino.RhinoMath.SqrtEpsilon * 10, false))
                            data[i, j, k] = insideValue;
                    }
                }
            });

            return data;
        }

        public static float MinDistanceToGeometry(Point3d p, IEnumerable<GeometryBase> geometries)
        {
            double min = double.MaxValue;
            foreach (GeometryBase g in geometries)
            {
                if (g == null) continue;
                double d = DistanceToGeometry(p, g);
                if (d < min) min = d;
            }
            return min == double.MaxValue ? float.MaxValue : (float)min;
        }

        /// <summary>Closest point on a single piece of geometry (same cases as distance queries).</summary>
        public static bool TryGetClosestPoint(Point3d p, GeometryBase g, out Point3d closest)
        {
            closest = p;
            if (g == null) return false;
            switch (g)
            {
                case Point pt:
                    closest = pt.Location;
                    return true;
                case PointCloud pc:
                    {
                        double md = double.MaxValue;
                        Point3d best = p;
                        foreach (Point3d q in pc.GetPoints())
                        {
                            double d = p.DistanceTo(q);
                            if (d < md)
                            {
                                md = d;
                                best = q;
                            }
                        }
                        if (md >= double.MaxValue - 1) return false;
                        closest = best;
                        return true;
                    }
                case Curve cv:
                    if (cv.ClosestPoint(p, out double tc))
                    {
                        closest = cv.PointAt(tc);
                        return true;
                    }
                    return false;
                case Mesh msh:
                    {
                        int fi = msh.ClosestPoint(p, out Point3d mp, out _, double.MaxValue);
                        if (fi >= 0)
                        {
                            closest = mp;
                            return true;
                        }
                        return false;
                    }
                case Brep br:
                    if (br.ClosestPoint(p, out Point3d bp, out ComponentIndex ci, out double su, out double tu,
                            double.MaxValue, out _))
                    {
                        closest = bp;
                        return true;
                    }
                    return false;
                case Surface srf:
                    if (srf.ClosestPoint(p, out double uu, out double vv))
                    {
                        closest = srf.PointAt(uu, vv);
                        return true;
                    }
                    return false;
                default:
                    var bbox = g.GetBoundingBox(true);
                    closest = bbox.ClosestPoint(p);
                    return bbox.IsValid;
            }
        }

        /// <summary>Closest point among a set of geometries; false if none succeeded.</summary>
        public static bool TryGetClosestPoint(Point3d p, IEnumerable<GeometryBase> geometries, out Point3d closest, out double distance)
        {
            closest = p;
            distance = double.MaxValue;
            if (geometries == null) return false;
            bool any = false;
            foreach (GeometryBase g in geometries)
            {
                if (g == null) continue;
                if (!TryGetClosestPoint(p, g, out Point3d q)) continue;
                double d = p.DistanceTo(q);
                if (d < distance)
                {
                    distance = d;
                    closest = q;
                    any = true;
                }
            }
            return any;
        }

        private static double DistanceToGeometry(Point3d p, GeometryBase g)
        {
            if (!TryGetClosestPoint(p, g, out Point3d q)) return double.MaxValue;
            return p.DistanceTo(q);
        }

        public static float[,,] PaintProximityMask(float[,,] domainMask, Box box, int nx, int ny, int nz,
            IEnumerable<GeometryBase> geometries, double proximityWorld)
        {
            var mask = new float[nx, ny, nz];
            if (geometries == null) return mask;

            Parallel.For(0, nx, i =>
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (domainMask[i, j, k] < 0.5f) continue;
                        Point3d c = CellCenterWorld(box, i, j, k, nx, ny, nz);
                        if (MinDistanceToGeometry(c, geometries) <= proximityWorld)
                            mask[i, j, k] = 1f;
                    }
                }
            });

            return mask;
        }

        /// <summary>
        /// Jacobi iteration for Laplace equation. Support fixed to supportValue, load fixed to loadValue.
        /// Outside domain: 0. Ghost neighbors outside domain contribute 0.
        /// </summary>
        public static float[,,] SolveLaplace(float[,,] domainMask, float[,,] supportMask, float[,,] loadMask,
            int nx, int ny, int nz, int iterations, float supportValue, float loadValue)
        {
            var phi = new float[nx, ny, nz];
            var phiNext = new float[nx, ny, nz];

            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                    {
                        if (supportMask[i, j, k] >= 0.5f) phi[i, j, k] = supportValue;
                        else if (loadMask[i, j, k] >= 0.5f) phi[i, j, k] = loadValue;
                        else if (domainMask[i, j, k] >= 0.5f) phi[i, j, k] = 0.5f * (supportValue + loadValue);
                        else phi[i, j, k] = 0f;
                    }

            for (int it = 0; it < iterations; it++)
            {
                for (int i = 0; i < nx; i++)
                {
                    for (int j = 0; j < ny; j++)
                    {
                        for (int k = 0; k < nz; k++)
                        {
                            if (domainMask[i, j, k] < 0.5f)
                            {
                                phiNext[i, j, k] = 0f;
                                continue;
                            }

                            if (supportMask[i, j, k] >= 0.5f)
                            {
                                phiNext[i, j, k] = supportValue;
                                continue;
                            }

                            if (loadMask[i, j, k] >= 0.5f)
                            {
                                phiNext[i, j, k] = loadValue;
                                continue;
                            }

                            float sum = 0f;
                            int n = 0;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i - 1, j, k); n++;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i + 1, j, k); n++;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i, j - 1, k); n++;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i, j + 1, k); n++;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i, j, k - 1); n++;
                            sum += NeighborPhi(phi, domainMask, nx, ny, nz, i, j, k + 1); n++;
                            phiNext[i, j, k] = sum / 6f;
                        }
                    }
                }

                var tmp = phi;
                phi = phiNext;
                phiNext = tmp;
            }

            return phi;
        }

        private static float NeighborPhi(float[,,] phi, float[,,] domain, int nx, int ny, int nz, int i, int j, int k)
        {
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0f;
            if (domain[i, j, k] < 0.5f) return 0f;
            return phi[i, j, k];
        }

        /// <summary>
        /// Central-difference gradient magnitude in index space; scaled by physical spacing later.
        /// </summary>
        public static float[,,] GradientMagnitude(float[,,] phi, float[,,] domainMask, int nx, int ny, int nz,
            double cellSizeX, double cellSizeY, double cellSizeZ)
        {
            var g = new float[nx, ny, nz];
            double inv2dx = 1.0 / (2.0 * cellSizeX);
            double inv2dy = 1.0 / (2.0 * cellSizeY);
            double inv2dz = 1.0 / (2.0 * cellSizeZ);

            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (domainMask[i, j, k] < 0.5f) continue;

                        double pxi = i > 0 ? phi[i - 1, j, k] : phi[i, j, k];
                        double pxa = i < nx - 1 ? phi[i + 1, j, k] : phi[i, j, k];
                        double pyi = j > 0 ? phi[i, j - 1, k] : phi[i, j, k];
                        double pya = j < ny - 1 ? phi[i, j + 1, k] : phi[i, j, k];
                        double pzi = k > 0 ? phi[i, j, k - 1] : phi[i, j, k];
                        double pza = k < nz - 1 ? phi[i, j, k + 1] : phi[i, j, k];

                        double gx = (pxa - pxi) * inv2dx;
                        double gy = (pya - pyi) * inv2dy;
                        double gz = (pza - pzi) * inv2dz;
                        g[i, j, k] = (float)Math.Sqrt(gx * gx + gy * gy + gz * gz);
                    }
                }
            }

            return g;
        }

        public static void NormalizeToUnitInterval(float[,,] data, float[,,] domainMask, int nx, int ny, int nz, bool invert)
        {
            float max = 0f;
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        if (domainMask[i, j, k] >= 0.5f)
                            max = Math.Max(max, data[i, j, k]);

            if (max < 1e-20f) max = 1f;

            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (domainMask[i, j, k] < 0.5f)
                        {
                            data[i, j, k] = 0f;
                            continue;
                        }

                        float v = data[i, j, k] / max;
                        data[i, j, k] = invert ? (1f - v) : v;
                    }
                }
            }
        }

        public static void ApplyContrast(float[,,] data, float[,,] domainMask, int nx, int ny, int nz, double exponent)
        {
            if (Math.Abs(exponent - 1.0) < 1e-9) return;
            float exp = (float)exponent;
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        if (domainMask[i, j, k] >= 0.5f)
                            data[i, j, k] = (float)Math.Pow(Math.Max(0, Math.Min(1, data[i, j, k])), exp);
        }

        /// <summary>
        /// Sets the outermost index layer to 0 (same rule as Close Voxel Data). Mutates <paramref name="data"/>.
        /// </summary>
        internal static void ZeroVoxelBoundaryInPlace(float[,,] data)
        {
            int lx = data.GetLength(0);
            int ly = data.GetLength(1);
            int lz = data.GetLength(2);
            for (int x = 0; x < lx; x++)
            {
                for (int y = 0; y < ly; y++)
                {
                    for (int z = 0; z < lz; z++)
                    {
                        if (x == 0 || y == 0 || z == 0 || x == lx - 1 || y == ly - 1 || z == lz - 1)
                            data[x, y, z] = 0f;
                    }
                }
            }
        }
    }
}
