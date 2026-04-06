using Rhino.Geometry;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace GHGPUPlugin.Chromodoris.Topology
{
    /// <summary>
    /// Euclidean proximity to support/load voxels (separable squared EDT, anisotropic world spacing) and optional box-center falloff.
    /// </summary>
    internal static class ProximityDensityBlend
    {
        /// <summary>
        /// Per voxel: world distance to nearest S∪L cell center. Uses separable EDT with dx²,dy²,dz² weights (correct for non-cubic voxels).
        /// Non-seeds use a finite sentinel instead of ∞ so the 1D Felzenszwalb pass never hits ∞−∞ (which becomes NaN and breaks multi-source EDT).
        /// </summary>
        public static float[,,] MinDistanceToSupportLoadWorld(Box box, float[,,] inside, float[,,] support, float[,,] load,
            int nx, int ny, int nz)
        {
            var dist = new float[nx, ny, nz];
            int n = nx * ny * nz;
            var ad = new double[n];
            var bd = new double[n];

            double dx = box.X.Length / nx;
            double dy = box.Y.Length / ny;
            double dz = box.Z.Length / nz;
            double wx = dx * dx;
            double wy = dy * dy;
            double wz = dz * dz;

            // Upper bound on squared world distance between any two cell centers; sentinel must dominate this.
            double maxSpanSq = wx * (nx - 1) * (nx - 1) + wy * (ny - 1) * (ny - 1) + wz * (nz - 1) * (nz - 1);
            double sentinelSq = maxSpanSq * 1e6 + 1.0;

            bool anySeed = false;
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        int id = I(i, j, k, nx, ny);
                        if (inside[i, j, k] < 0.5f)
                        {
                            ad[id] = sentinelSq;
                            continue;
                        }

                        if (support[i, j, k] >= 0.5f || load[i, j, k] >= 0.5f)
                        {
                            ad[id] = 0.0;
                            anySeed = true;
                        }
                        else
                            ad[id] = sentinelSq;
                    }
                }
            }

            if (!anySeed)
            {
                for (int i = 0; i < nx; i++)
                    for (int j = 0; j < ny; j++)
                        for (int k = 0; k < nz; k++)
                            dist[i, j, k] = inside[i, j, k] < 0.5f ? 0f : float.MaxValue;
                return dist;
            }

            int lineLen = Math.Max(nx, Math.Max(ny, nz));
            using var tlsIn = new ThreadLocal<double[]>(() => new double[lineLen], trackAllValues: false);
            using var tlsOut = new ThreadLocal<double[]>(() => new double[lineLen], trackAllValues: false);

            // Pass X: ad -> bd
            Parallel.For(0, ny * nz, t =>
            {
                double[] lineIn = tlsIn.Value;
                double[] lineOut = tlsOut.Value;
                int j = t / nz;
                int k = t % nz;
                for (int i = 0; i < nx; i++)
                    lineIn[i] = ad[I(i, j, k, nx, ny)];
                Edt1dSquaredWeighted(lineIn, lineOut, nx, wx);
                for (int i = 0; i < nx; i++)
                    bd[I(i, j, k, nx, ny)] = lineOut[i];
            });

            // Pass Y: bd -> ad
            Parallel.For(0, nx * nz, t =>
            {
                double[] lineIn = tlsIn.Value;
                double[] lineOut = tlsOut.Value;
                int i = t / nz;
                int k = t % nz;
                for (int j = 0; j < ny; j++)
                    lineIn[j] = bd[I(i, j, k, nx, ny)];
                Edt1dSquaredWeighted(lineIn, lineOut, ny, wy);
                for (int j = 0; j < ny; j++)
                    ad[I(i, j, k, nx, ny)] = lineOut[j];
            });

            // Pass Z: ad -> bd
            Parallel.For(0, nx * ny, t =>
            {
                double[] lineIn = tlsIn.Value;
                double[] lineOut = tlsOut.Value;
                int i = t / ny;
                int j = t % ny;
                for (int k = 0; k < nz; k++)
                    lineIn[k] = ad[I(i, j, k, nx, ny)];
                Edt1dSquaredWeighted(lineIn, lineOut, nz, wz);
                for (int k = 0; k < nz; k++)
                    bd[I(i, j, k, nx, ny)] = lineOut[k];
            });

            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (inside[i, j, k] < 0.5f)
                        {
                            dist[i, j, k] = 0f;
                            continue;
                        }

                        double sq = bd[I(i, j, k, nx, ny)];
                        if (double.IsNaN(sq) || sq >= sentinelSq * 0.5)
                            dist[i, j, k] = float.MaxValue;
                        else
                            dist[i, j, k] = (float)Math.Sqrt(Math.Max(0.0, sq));
                    }
                }
            }

            return dist;
        }

        private static int I(int i, int j, int k, int nx, int ny) => i + nx * (j + ny * k);

        /// <summary>d[q] = min_p fv[p] + w·(q−p)² — anisotropic axis step w = (cell size)².</summary>
        private static void Edt1dSquaredWeighted(double[] fv, double[] d, int n, double w)
        {
            if (n <= 0) return;
            if (w < 1e-30) w = 1.0;

            bool any = false;
            for (int i = 0; i < n; i++)
            {
                if (fv[i] < double.PositiveInfinity * 0.5)
                {
                    any = true;
                    break;
                }
            }

            if (!any)
            {
                for (int i = 0; i < n; i++)
                    d[i] = fv[i];
                return;
            }

            var v = new int[n];
            var z = new double[n + 1];
            int k = 0;
            v[0] = 0;
            z[0] = double.NegativeInfinity;
            z[1] = double.PositiveInfinity;

            for (int q = 1; q < n; q++)
            {
                double denom = 2.0 * w * (q - v[k]);
                if (Math.Abs(denom) < 1e-30)
                    denom = denom >= 0 ? 1e-30 : -1e-30;
                double s = ((fv[q] + w * q * q) - (fv[v[k]] + w * v[k] * v[k])) / denom;
                while (k > 0 && s <= z[k])
                {
                    k--;
                    denom = 2.0 * w * (q - v[k]);
                    if (Math.Abs(denom) < 1e-30)
                        denom = denom >= 0 ? 1e-30 : -1e-30;
                    s = ((fv[q] + w * q * q) - (fv[v[k]] + w * v[k] * v[k])) / denom;
                }

                k++;
                v[k] = q;
                z[k] = s;
                z[k + 1] = double.PositiveInfinity;
            }

            k = 0;
            for (int q = 0; q < n; q++)
            {
                while (k + 1 <= n && z[k + 1] < q)
                    k++;
                int vk = v[k];
                double t = q - vk;
                d[q] = fv[vk] + w * t * t;
            }
        }

        /// <summary>
        /// Proximity 0…1: high near S∪L (if includeSl) and (optionally) near box center. SL uses exp(-d/Rsl).
        /// </summary>
        public static void FillProximityField(float[,,] proximity, float[,,] inside, float[,,] dSlWorld,
            Box box, int nx, int ny, int nz, bool includeSl, double slRadiusWorld, double centerWeight, double centerRadiusWorld)
        {
            BoundingBox bb = box.BoundingBox;
            Point3d boxCenter = bb.Center;
            bool useCenter = centerWeight > 1e-12 && centerRadiusWorld > 1e-12;
            bool useSl = includeSl && slRadiusWorld > 1e-12;

            Parallel.For(0, nx * ny * nz, t =>
            {
                int k = t % nz;
                int j = (t / nz) % ny;
                int i = t / (nz * ny);

                if (inside[i, j, k] < 0.5f)
                {
                    proximity[i, j, k] = 0f;
                    return;
                }

                float pSl = 0f;
                if (useSl)
                {
                    double d = dSlWorld[i, j, k];
                    if (d >= float.MaxValue * 0.5f)
                        pSl = 0f;
                    else
                        pSl = (float)Math.Exp(-d / slRadiusWorld);
                }

                float pC = 0f;
                if (useCenter)
                {
                    Point3d c = WorkflowAGrid.CellCenterWorld(box, i, j, k, nx, ny, nz);
                    double dc = c.DistanceTo(boxCenter);
                    pC = (float)Math.Exp(-dc / centerRadiusWorld);
                }

                float p = pSl;
                if (useCenter)
                    p = Math.Max(p, (float)(centerWeight * pC));

                proximity[i, j, k] = p;
            });
        }

        public static void Blend(float[,,] laplaceDensity, float[,,] proximityNorm, float[,,] inside,
            int nx, int ny, int nz, double mix, float[,,] output)
        {
            double m = Rhino.RhinoMath.Clamp(mix, 0.0, 1.0);
            Parallel.For(0, nx * ny * nz, t =>
            {
                int k = t % nz;
                int j = (t / nz) % ny;
                int i = t / (nz * ny);

                if (inside[i, j, k] < 0.5f)
                {
                    output[i, j, k] = 0f;
                    return;
                }

                output[i, j, k] = (float)((1.0 - m) * laplaceDensity[i, j, k] + m * proximityNorm[i, j, k]);
            });
        }
    }
}
