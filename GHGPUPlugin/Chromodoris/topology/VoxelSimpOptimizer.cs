using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris.Topology
{
    /// <summary>
    /// SIMP compliance minimization on a voxel hex8 mesh (matrix-free PCG + optimality criteria).
    /// Coarse solve: optional stride downsamples masks; stiffness uses physical brick sizes per element.
    /// Output density is trilinearly upsampled to the original (fine) mask resolution for smooth isosurfaces.
    /// </summary>
    internal static class VoxelSimpOptimizer
    {
        internal const double Penalty = 1e12;

        public sealed class Result
        {
            public float[,,] DensityPhys;
            public double Compliance;
            public int IterationsUsed;
            public string Message;
        }

        public static Result Run(
            float[,,] insideFine,
            float[,,] supportFine,
            float[,,] loadFine,
            double dxFine, double dyFine, double dzFine,
            Vector3d forceTotal,
            double volumeFraction,
            int maxOuterIter,
            int maxPcgIter,
            double simpP,
            double moveLimit,
            double emin,
            double nu,
            int maxElements,
            int solveStride,
            bool useGpuMatVec = true)
        {
            int nx = insideFine.GetLength(0);
            int ny = insideFine.GetLength(1);
            int nz = insideFine.GetLength(2);

            var res = new Result
            {
                DensityPhys = new float[nx, ny, nz],
                Message = ""
            };

            if (volumeFraction <= 0 || volumeFraction > 1)
            {
                res.Message = "VolumeFraction must be in (0,1].";
                return res;
            }

            int S = Math.Max(1, solveStride);

            float[,,] inside, support, load;
            int nxc, nyc, nzc;
            if (S == 1)
            {
                inside = insideFine;
                support = supportFine;
                load = loadFine;
                nxc = nx;
                nyc = ny;
                nzc = nz;
            }
            else
            {
                DownsampleMasks(insideFine, supportFine, loadFine, nx, ny, nz, S,
                    out inside, out support, out load, out nxc, out nyc, out nzc);
            }

            int nElem = 0;
            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                        if (inside[i, j, k] >= 0.5f) nElem++;

            if (nElem == 0)
            {
                res.Message = "No voxels inside the design domain.";
                return res;
            }

            if (nElem > maxElements)
            {
                res.Message = $"Too many coarse voxels ({nElem} > {maxElements}). Raise SolveStride or MaxElements.";
                return res;
            }

            var nodeUsed = new bool[nxc + 1, nyc + 1, nzc + 1];
            var supN = new bool[nxc + 1, nyc + 1, nzc + 1];
            var loadN = new bool[nxc + 1, nyc + 1, nzc + 1];

            for (int i = 0; i < nxc; i++)
            {
                for (int j = 0; j < nyc; j++)
                {
                    for (int k = 0; k < nzc; k++)
                    {
                        if (inside[i, j, k] < 0.5f) continue;
                        for (int di = 0; di <= 1; di++)
                            for (int dj = 0; dj <= 1; dj++)
                                for (int dk = 0; dk <= 1; dk++)
                                {
                                    int I = i + di, J = j + dj, K = k + dk;
                                    nodeUsed[I, J, K] = true;
                                    if (support[i, j, k] >= 0.5f) supN[I, J, K] = true;
                                    if (load[i, j, k] >= 0.5f) loadN[I, J, K] = true;
                                }
                    }
                }
            }

            var nid = new int[nxc + 1, nyc + 1, nzc + 1];
            for (int i = 0; i <= nxc; i++)
                for (int j = 0; j <= nyc; j++)
                    for (int k = 0; k <= nzc; k++)
                        nid[i, j, k] = -1;

            int nNodes = 0;
            for (int i = 0; i <= nxc; i++)
                for (int j = 0; j <= nyc; j++)
                    for (int k = 0; k <= nzc; k++)
                        if (nodeUsed[i, j, k])
                            nid[i, j, k] = nNodes++;

            int ndof = 3 * nNodes;
            var fixedDof = new bool[ndof];
            for (int i = 0; i <= nxc; i++)
                for (int j = 0; j <= nyc; j++)
                    for (int k = 0; k <= nzc; k++)
                    {
                        if (!supN[i, j, k]) continue;
                        int id = nid[i, j, k];
                        if (id < 0) continue;
                        fixedDof[id * 3 + 0] = true;
                        fixedDof[id * 3 + 1] = true;
                        fixedDof[id * 3 + 2] = true;
                    }

            int nLoadNodes = 0;
            for (int i = 0; i <= nxc; i++)
                for (int j = 0; j <= nyc; j++)
                    for (int k = 0; k <= nzc; k++)
                        if (loadN[i, j, k]) nLoadNodes++;

            var f = new double[ndof];
            if (nLoadNodes > 0)
            {
                double fx = forceTotal.X / nLoadNodes;
                double fy = forceTotal.Y / nLoadNodes;
                double fz = forceTotal.Z / nLoadNodes;
                for (int i = 0; i <= nxc; i++)
                    for (int j = 0; j <= nyc; j++)
                        for (int k = 0; k <= nzc; k++)
                        {
                            if (!loadN[i, j, k]) continue;
                            int id = nid[i, j, k];
                            if (id < 0) continue;
                            f[id * 3 + 0] += fx;
                            f[id * 3 + 1] += fy;
                            f[id * 3 + 2] += fz;
                        }
            }
            else
            {
                res.Message = "No load voxels: paint LoadMask where the external load is applied.";
                return res;
            }

            int nFixedDof = 0;
            for (int i = 0; i < ndof; i++) if (fixedDof[i]) nFixedDof++;
            if (nFixedDof < 6)
            {
                res.Message = "Too few fixed DOFs. Enlarge SupportMask so all rigid-body modes are restrained.";
                return res;
            }

            var ex = new int[nElem];
            var ey = new int[nElem];
            var ez = new int[nElem];
            var elWx = new int[nElem];
            var elWy = new int[nElem];
            var elWz = new int[nElem];
            var passive = new bool[nElem];
            var dofMap = new int[nElem, 24];
            int ei = 0;
            int nDesign = 0;

            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                    {
                        if (inside[i, j, k] < 0.5f) continue;
                        ex[ei] = i;
                        ey[ei] = j;
                        ez[ei] = k;
                        BlockSpan(i, j, k, S, nx, ny, nz,
                            out int wx, out int wy, out int wz);
                        elWx[ei] = wx;
                        elWy[ei] = wy;
                        elWz[ei] = wz;

                        bool pas = support[i, j, k] >= 0.5f || load[i, j, k] >= 0.5f;
                        passive[ei] = pas;
                        if (!pas) nDesign++;

                        int[] cn = {
                            nid[i, j, k], nid[i + 1, j, k], nid[i + 1, j + 1, k], nid[i, j + 1, k],
                            nid[i, j, k + 1], nid[i + 1, j, k + 1], nid[i + 1, j + 1, k + 1], nid[i, j + 1, k + 1]
                        };
                        for (int n = 0; n < 8; n++)
                        {
                            int g = cn[n] * 3;
                            dofMap[ei, n * 3 + 0] = g + 0;
                            dofMap[ei, n * 3 + 1] = g + 1;
                            dofMap[ei, n * 3 + 2] = g + 2;
                        }
                        ei++;
                    }

            if (nDesign == 0)
            {
                res.Message = "No design voxels: need interior cells that are neither support nor load.";
                return res;
            }

            double volTarget = volumeFraction * nDesign;

            var keCache = new Dictionary<(int, int, int), double[,]>();
            double[][,] K0e = new double[nElem][,];
            for (int e = 0; e < nElem; e++)
            {
                var key = (elWx[e], elWy[e], elWz[e]);
                if (!keCache.TryGetValue(key, out double[,] K0))
                {
                    K0 = new double[24, 24];
                    Hex8BrickKe.BuildUnitKe(nu,
                        elWx[e] * dxFine, elWy[e] * dyFine, elWz[e] * dzFine, K0);
                    keCache[key] = K0;
                }
                K0e[e] = K0;
            }

            var x = new double[nElem];
            var xNew = new double[nElem];
            var dc = new double[nElem];
            var ue = new double[24];
            var u = new double[ndof];
            var y = new double[ndof];
            var r = new double[ndof];
            var zvec = new double[ndof];
            var pvec = new double[ndof];
            var Ap = new double[ndof];
            var diag = new double[ndof];

            for (int e = 0; e < nElem; e++)
                x[e] = passive[e] ? 1.0 : volumeFraction;

            double eminClamped = Math.Max(1e-9, Math.Min(emin, 0.5));

            IntPtr gpuCtx = IntPtr.Zero;
            bool useGpu = useGpuMatVec && MetalSharedContext.TryGetContext(out gpuCtx);
            float[] Ke_flat = null;
            int[] dofMapFlat = null;
            float[] rhoFlat = null;
            float[] vFlat = null;
            float[] AvFlat = null;

            if (useGpu)
            {
                Ke_flat = new float[nElem * 24 * 24];
                dofMapFlat = new int[nElem * 24];
                rhoFlat = new float[nElem];
                vFlat = new float[ndof];
                AvFlat = new float[ndof];

                for (int e = 0; e < nElem; e++)
                {
                    double[,] K0 = K0e[e];
                    int baseK = e * 24 * 24;
                    for (int a = 0; a < 24; a++)
                        for (int b = 0; b < 24; b++)
                            Ke_flat[baseK + a * 24 + b] = (float)K0[a, b];
                    for (int a = 0; a < 24; a++)
                        dofMapFlat[e * 24 + a] = dofMap[e, a];
                }
            }

            Action<double[], double[]> matVecFn = (vecIn, vecOut) =>
            {
                if (useGpu && Ke_flat != null && dofMapFlat != null && rhoFlat != null && vFlat != null && AvFlat != null)
                {
                    for (int e = 0; e < nElem; e++)
                        rhoFlat[e] = (float)StiffnessInterp(x[e], passive[e], eminClamped, simpP);
                    for (int i = 0; i < ndof; i++)
                        vFlat[i] = (float)vecIn[i];

                    int code = MetalBridge.FemMatVec(
                        gpuCtx, Ke_flat, dofMapFlat, rhoFlat, vFlat, AvFlat, nElem, ndof);

                    if (code == 0)
                    {
                        for (int i = 0; i < ndof; i++)
                            vecOut[i] = AvFlat[i];
                        for (int i = 0; i < ndof; i++)
                            if (fixedDof[i])
                                vecOut[i] += Penalty * vecIn[i];
                        return;
                    }
                }

                MatVec(nElem, dofMap, fixedDof, K0e, x, passive, eminClamped, simpP, vecIn, vecOut);
            };

            for (int outer = 0; outer < maxOuterIter; outer++)
            {
                BuildDiagonal(nElem, dofMap, fixedDof, K0e, x, passive, eminClamped, simpP, diag);

                if (outer == 0)
                    Array.Clear(u, 0, ndof);

                double tolRel = outer * 3 < maxOuterIter * 2 ? 1e-5 : 1e-7;
                PcgSolve(nElem, dofMap, fixedDof, K0e, x, passive, eminClamped, simpP, diag, f, u, y, r, zvec, pvec, Ap, matVecFn, maxPcgIter, tolRel);

                double C = Dot(f, u);

                for (int e = 0; e < nElem; e++)
                {
                    double[,] K0 = K0e[e];
                    for (int a = 0; a < 24; a++)
                        ue[a] = u[dofMap[e, a]];
                    double ce = Hex8BrickKe.ElementEnergy(K0, ue);
                    if (!passive[e])
                    {
                        double dxrho = simpP * Math.Pow(x[e], simpP - 1.0) * (1.0 - eminClamped);
                        dc[e] = -dxrho * ce;
                    }
                    else dc[e] = 0;
                }

                res.Compliance = C;
                OcUpdate(nElem, x, dc, passive, volTarget, moveLimit, 0.001, xNew);

                double change = 0;
                for (int e = 0; e < nElem; e++)
                {
                    change = Math.Max(change, Math.Abs(xNew[e] - x[e]));
                    x[e] = xNew[e];
                }

                res.IterationsUsed = outer + 1;
                if (change < 0.01 && outer > 5) break;
            }

            var rhoCoarse = new float[nxc, nyc, nzc];
            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                        rhoCoarse[i, j, k] = 0f;

            for (int e = 0; e < nElem; e++)
            {
                double rho = StiffnessInterp(x[e], passive[e], eminClamped, simpP);
                rhoCoarse[ex[e], ey[e], ez[e]] = (float)rho;
            }

            if (S == 1)
            {
                for (int i = 0; i < nx; i++)
                    for (int j = 0; j < ny; j++)
                        for (int k = 0; k < nz; k++)
                            res.DensityPhys[i, j, k] = rhoCoarse[i, j, k];
            }
            else
            {
                UpsampleTrilinear(rhoCoarse, nxc, nyc, nzc, nx, ny, nz, insideFine, res.DensityPhys);
            }

            res.Message = "OK";
            return res;
        }

        /// <summary>Fine cell counts along each axis for coarse cell (ic,jc,kc) with stride S.</summary>
        private static void BlockSpan(int ic, int jc, int kc, int S, int nx, int ny, int nz,
            out int wx, out int wy, out int wz)
        {
            if (S <= 1)
            {
                wx = wy = wz = 1;
                return;
            }

            int i0 = ic * S, i1ex = Math.Min((ic + 1) * S, nx);
            wx = i1ex - i0;
            int j0 = jc * S, j1ex = Math.Min((jc + 1) * S, ny);
            wy = j1ex - j0;
            int k0 = kc * S, k1ex = Math.Min((kc + 1) * S, nz);
            wz = k1ex - k0;
        }

        private static void DownsampleMasks(float[,,] inside, float[,,] sup, float[,,] load,
            int nx, int ny, int nz, int S,
            out float[,,] insideC, out float[,,] supC, out float[,,] loadC,
            out int nxc, out int nyc, out int nzc)
        {
            nxc = (nx + S - 1) / S;
            nyc = (ny + S - 1) / S;
            nzc = (nz + S - 1) / S;
            insideC = new float[nxc, nyc, nzc];
            supC = new float[nxc, nyc, nzc];
            loadC = new float[nxc, nyc, nzc];

            for (int ic = 0; ic < nxc; ic++)
            {
                int i0 = ic * S, i1 = Math.Min((ic + 1) * S, nx);
                for (int jc = 0; jc < nyc; jc++)
                {
                    int j0 = jc * S, j1 = Math.Min((jc + 1) * S, ny);
                    for (int kc = 0; kc < nzc; kc++)
                    {
                        int k0 = kc * S, k1 = Math.Min((kc + 1) * S, nz);
                        bool ins = false, s = false, l = false;
                        for (int i = i0; i < i1 && !ins; i++)
                            for (int j = j0; j < j1 && !ins; j++)
                                for (int k = k0; k < k1 && !ins; k++)
                                    if (inside[i, j, k] >= 0.5f) ins = true;
                        for (int i = i0; i < i1 && !s; i++)
                            for (int j = j0; j < j1 && !s; j++)
                                for (int k = k0; k < k1 && !s; k++)
                                    if (sup[i, j, k] >= 0.5f) s = true;
                        for (int i = i0; i < i1 && !l; i++)
                            for (int j = j0; j < j1 && !l; j++)
                                for (int k = k0; k < k1 && !l; k++)
                                    if (load[i, j, k] >= 0.5f) l = true;
                        insideC[ic, jc, kc] = ins ? 1f : 0f;
                        supC[ic, jc, kc] = s ? 1f : 0f;
                        loadC[ic, jc, kc] = l ? 1f : 0f;
                    }
                }
            }
        }

        private static void UpsampleTrilinear(float[,,] coarse, int nxc, int nyc, int nzc,
            int nx, int ny, int nz, float[,,] insideFine, float[,,] outFine)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (insideFine[i, j, k] < 0.5f)
                        {
                            outFine[i, j, k] = 0f;
                            continue;
                        }

                        double fx = (i + 0.5) / nx * nxc - 0.5;
                        double fy = (j + 0.5) / ny * nyc - 0.5;
                        double fz = (k + 0.5) / nz * nzc - 0.5;
                        outFine[i, j, k] = (float)TriSample(coarse, fx, fy, fz, nxc, nyc, nzc);
                    }
                }
            }
        }

        private static double TriSample(float[,,] c, double x, double y, double z, int nxc, int nyc, int nzc)
        {
            x = Clamp(x, 0, Math.Max(0, nxc - 1));
            y = Clamp(y, 0, Math.Max(0, nyc - 1));
            z = Clamp(z, 0, Math.Max(0, nzc - 1));

            int x0 = (int)Math.Floor(x);
            int y0 = (int)Math.Floor(y);
            int z0 = (int)Math.Floor(z);
            int x1 = Math.Min(x0 + 1, nxc - 1);
            int y1 = Math.Min(y0 + 1, nyc - 1);
            int z1 = Math.Min(z0 + 1, nzc - 1);

            double tx = x - x0;
            double ty = y - y0;
            double tz = z - z0;

            double c000 = c[x0, y0, z0];
            double c100 = c[x1, y0, z0];
            double c010 = c[x0, y1, z0];
            double c110 = c[x1, y1, z0];
            double c001 = c[x0, y0, z1];
            double c101 = c[x1, y0, z1];
            double c011 = c[x0, y1, z1];
            double c111 = c[x1, y1, z1];

            double c00 = c000 * (1 - tx) + c100 * tx;
            double c10 = c010 * (1 - tx) + c110 * tx;
            double c01 = c001 * (1 - tx) + c101 * tx;
            double c11 = c011 * (1 - tx) + c111 * tx;

            double c0 = c00 * (1 - ty) + c10 * ty;
            double c1 = c01 * (1 - ty) + c11 * ty;

            return c0 * (1 - tz) + c1 * tz;
        }

        private static double Clamp(double v, double a, double b)
        {
            if (v < a) return a;
            if (v > b) return b;
            return v;
        }

        private static double StiffnessInterp(double xd, bool passive, double emin, double p)
        {
            if (passive) return 1.0;
            return emin + Math.Pow(xd, p) * (1.0 - emin);
        }

        private static void MatVec(int nElem, int[,] dofMap, bool[] fixedDof, double[][,] K0e, double[] x,
            bool[] passive, double emin, double p, double[] v, double[] Av)
        {
            Array.Clear(Av, 0, Av.Length);
            for (int e = 0; e < nElem; e++)
            {
                double rho = StiffnessInterp(x[e], passive[e], emin, p);
                double[,] K0 = K0e[e];
                for (int a = 0; a < 24; a++)
                {
                    double s = 0;
                    for (int b = 0; b < 24; b++)
                        s += K0[a, b] * v[dofMap[e, b]];
                    Av[dofMap[e, a]] += rho * s;
                }
            }
            for (int i = 0; i < fixedDof.Length; i++)
                if (fixedDof[i])
                    Av[i] += Penalty * v[i];
        }

        private static void BuildDiagonal(int nElem, int[,] dofMap, bool[] fixedDof, double[][,] K0e, double[] x,
            bool[] passive, double emin, double p, double[] diag)
        {
            Array.Clear(diag, 0, diag.Length);
            for (int e = 0; e < nElem; e++)
            {
                double rho = StiffnessInterp(x[e], passive[e], emin, p);
                double[,] K0 = K0e[e];
                for (int a = 0; a < 24; a++)
                {
                    int g = dofMap[e, a];
                    diag[g] += rho * K0[a, a];
                }
            }
            for (int i = 0; i < fixedDof.Length; i++)
                if (fixedDof[i])
                    diag[i] += Penalty;
        }

        private static void PcgSolve(int nElem, int[,] dofMap, bool[] fixedDof, double[][,] K0e, double[] x,
            bool[] passive, double emin, double p, double[] diag, double[] b, double[] u,
            double[] y, double[] r, double[] zvec, double[] pvec, double[] Ap,
            Action<double[], double[]> matVecFn,
            int maxIter, double tolRel)
        {
            int n = b.Length;
            matVecFn(u, y);
            for (int i = 0; i < n; i++)
                r[i] = b[i] - y[i];

            double normB = 0;
            for (int i = 0; i < n; i++)
                normB += b[i] * b[i];
            normB = Math.Sqrt(normB);
            if (normB < 1e-30) normB = 1;

            for (int i = 0; i < n; i++)
                zvec[i] = diag[i] > 1e-30 ? r[i] / diag[i] : r[i];
            Array.Copy(zvec, pvec, n);

            double rzOld = Dot(r, zvec);

            for (int it = 0; it < maxIter; it++)
            {
                matVecFn(pvec, Ap);
                double denom = Dot(pvec, Ap);
                if (Math.Abs(denom) < 1e-40) break;
                double alpha = rzOld / denom;
                for (int i = 0; i < n; i++)
                {
                    u[i] += alpha * pvec[i];
                    r[i] -= alpha * Ap[i];
                }
                double nr = Norm(r);
                if (nr < tolRel * normB) break;

                for (int i = 0; i < n; i++)
                    zvec[i] = diag[i] > 1e-30 ? r[i] / diag[i] : r[i];
                double rzNew = Dot(r, zvec);
                double beta = rzNew / (rzOld + 1e-40);
                for (int i = 0; i < n; i++)
                    pvec[i] = zvec[i] + beta * pvec[i];
                rzOld = rzNew;
            }
        }

        private static double Dot(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
                s += a[i] * b[i];
            return s;
        }

        private static double Norm(double[] a)
        {
            return Math.Sqrt(Dot(a, a));
        }

        private static void OcUpdate(int nElem, double[] x, double[] dc, bool[] passive, double volTarget,
            double move, double xmin, double[] xNew)
        {
            double l1 = 1e-12, l2 = 1e12;
            for (int bis = 0; bis < 80; bis++)
            {
                double lmid = 0.5 * (l1 + l2);
                double sumDesign = 0;
                for (int e = 0; e < nElem; e++)
                {
                    if (passive[e])
                    {
                        xNew[e] = 1;
                        continue;
                    }
                    double step = Math.Sqrt(Math.Max(1e-30, -dc[e] / (lmid + 1e-30)));
                    double raw = x[e] * step;
                    raw = Math.Max(xmin, Math.Max(x[e] - move, Math.Min(1, Math.Min(x[e] + move, raw))));
                    xNew[e] = raw;
                    sumDesign += xNew[e];
                }

                if (sumDesign > volTarget) l1 = lmid;
                else l2 = lmid;
                if ((l2 - l1) / (l1 + l2 + 1e-30) < 1e-4) break;
            }

            double lmidF = 0.5 * (l1 + l2);
            for (int e = 0; e < nElem; e++)
            {
                if (passive[e]) { xNew[e] = 1; continue; }
                double step = Math.Sqrt(Math.Max(1e-30, -dc[e] / (lmidF + 1e-30)));
                double raw = x[e] * step;
                xNew[e] = Math.Max(xmin, Math.Max(x[e] - move, Math.Min(1, Math.Min(x[e] + move, raw))));
            }
        }
    }
}
