using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace GHGPUPlugin.Chromodoris.Topology
{
    public sealed class MgLevel
    {
        public int NxC, NyC, NzC;
        public int NElem, NNodes, NDof;

        public float[] KeUnique;
        public int[] KeIdx;
        public int[] DofMap;
        public float[] Diag;
        public bool[] FixedDof;
        public byte[] FixedMask;

        public int[] ProlongCoarse;
        public float[] ProlongWeights;

        public int[][] CoarseToFineElems;

        public double[][,] K0e;
        public bool[] Passive;
        public int[,] DofMapCpu;
        public int[] ExC, EyC, EzC;
    }

    /// <summary>Pins multigrid host arrays for native <c>mb_fem_mgpcg_solve</c>.</summary>
    public sealed class MgPinnedData : IDisposable
    {
        private readonly List<GCHandle> _handles = new();

        public IntPtr[] KeUnique { get; }
        public IntPtr[] KeIdx { get; }
        public IntPtr[] DofMap { get; }
        public IntPtr[] Diag { get; }
        public IntPtr[] Fixed { get; }
        public IntPtr[] Prolong { get; }
        public IntPtr[] ProlongW { get; }
        public int[] NElem { get; }
        public int[] NDof { get; }
        public int[] NumUnique { get; }

        public MgPinnedData(IReadOnlyList<MgLevel> levels)
        {
            int L = levels.Count;
            KeUnique = new IntPtr[L];
            KeIdx = new IntPtr[L];
            DofMap = new IntPtr[L];
            Diag = new IntPtr[L];
            Fixed = new IntPtr[L];
            Prolong = new IntPtr[L];
            ProlongW = new IntPtr[L];
            NElem = new int[L];
            NDof = new int[L];
            NumUnique = new int[L];

            for (int l = 0; l < L; l++)
            {
                var lv = levels[l];
                NElem[l] = lv.NElem;
                NDof[l] = lv.NDof;
                NumUnique[l] = lv.KeUnique.Length / (24 * 24);

                KeUnique[l] = Pin(lv.KeUnique);
                KeIdx[l] = Pin(lv.KeIdx);
                DofMap[l] = Pin(lv.DofMap);
                Diag[l] = Pin(lv.Diag);
                Fixed[l] = Pin(lv.FixedMask);

                if (l == 0)
                {
                    Prolong[l] = IntPtr.Zero;
                    ProlongW[l] = IntPtr.Zero;
                }
                else
                {
                    Prolong[l] = Pin(lv.ProlongCoarse);
                    ProlongW[l] = Pin(lv.ProlongWeights);
                }
            }
        }

        private IntPtr Pin<T>(T[] arr) where T : unmanaged
        {
            var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
            _handles.Add(h);
            return h.AddrOfPinnedObject();
        }

        public void Dispose()
        {
            foreach (var h in _handles)
                if (h.IsAllocated) h.Free();
            _handles.Clear();
        }
    }

    public static class MgGrid
    {
        private const double FixedPenalty = 1e12;

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

        /// <summary>Build geometric multigrid levels (finest first). Stops when min dimension &lt; 4 or maxLevels reached.</summary>
        public static List<MgLevel> BuildHierarchy(
            float[,,] inside,
            float[,,] support,
            float[,,] load,
            int nxFine,
            int nyFine,
            int nzFine,
            int coarseStride,
            double dxFine,
            double dyFine,
            double dzFine,
            double nu,
            int maxLevels = 4)
        {
            var levels = new List<MgLevel>();
            float[,,] ins = inside;
            float[,,] sup = support;
            float[,,] lo = load;
            int strideEff = Math.Max(1, coarseStride);

            for (int li = 0; li < maxLevels; li++)
            {
                var lv = AssembleLevel(ins, sup, lo, nxFine, nyFine, nzFine, strideEff, dxFine, dyFine, dzFine, nu);
                levels.Add(lv);

                int nxc = lv.NxC, nyc = lv.NyC, nzc = lv.NzC;
                if (Math.Min(nxc, Math.Min(nyc, nzc)) < 4)
                    break;

                int nxc2 = Math.Max(1, (nxc + 1) / 2);
                int nyc2 = Math.Max(1, (nyc + 1) / 2);
                int nzc2 = Math.Max(1, (nzc + 1) / 2);
                if (nxc2 == nxc && nyc2 == nyc && nzc2 == nzc)
                    break;

                ins = CoarsenMask(ins, nxc, nyc, nzc, nxc2, nyc2, nzc2);
                sup = CoarsenMask(sup, nxc, nyc, nzc, nxc2, nyc2, nzc2);
                lo = CoarsenMask(lo, nxc, nyc, nzc, nxc2, nyc2, nzc2);
                strideEff *= 2;
            }

            if (levels.Count >= 2)
            {
                for (int l = 0; l < levels.Count - 1; l++)
                    BuildProlongation(levels[l], levels[l + 1]);
            }

            return levels;
        }

        private static float[,,] CoarsenMask(float[,,] src, int nxf, int nyf, int nzf, int nxc, int nyc, int nzc)
        {
            var dst = new float[nxc, nyc, nzc];
            for (int ic = 0; ic < nxc; ic++)
            {
                int i0 = ic * 2, i1 = Math.Min(i0 + 2, nxf);
                for (int jc = 0; jc < nyc; jc++)
                {
                    int j0 = jc * 2, j1 = Math.Min(j0 + 2, nyf);
                    for (int kc = 0; kc < nzc; kc++)
                    {
                        int k0 = kc * 2, k1 = Math.Min(k0 + 2, nzf);
                        bool any = false;
                        for (int i = i0; i < i1 && !any; i++)
                            for (int j = j0; j < j1 && !any; j++)
                                for (int k = k0; k < k1 && !any; k++)
                                    if (src[i, j, k] >= 0.5f) any = true;
                        dst[ic, jc, kc] = any ? 1f : 0f;
                    }
                }
            }
            return dst;
        }

        private static MgLevel AssembleLevel(
            float[,,] inside,
            float[,,] support,
            float[,,] load,
            int nxFine,
            int nyFine,
            int nzFine,
            int strideEff,
            double dxFine,
            double dyFine,
            double dzFine,
            double nu)
        {
            int nxc = inside.GetLength(0);
            int nyc = inside.GetLength(1);
            int nzc = inside.GetLength(2);

            int nElem = 0;
            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                        if (inside[i, j, k] >= 0.5f) nElem++;

            var nodeUsed = new bool[nxc + 1, nyc + 1, nzc + 1];
            var supN = new bool[nxc + 1, nyc + 1, nzc + 1];
            var loadN = new bool[nxc + 1, nyc + 1, nzc + 1];

            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
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

            var ex = new int[nElem];
            var ey = new int[nElem];
            var ez = new int[nElem];
            var elWx = new int[nElem];
            var elWy = new int[nElem];
            var elWz = new int[nElem];
            var passive = new bool[nElem];
            var dofMap = new int[nElem, 24];
            int ei = 0;

            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                    {
                        if (inside[i, j, k] < 0.5f) continue;
                        ex[ei] = i;
                        ey[ei] = j;
                        ez[ei] = k;
                        BlockSpan(i, j, k, strideEff, nxFine, nyFine, nzFine, out int wx, out int wy, out int wz);
                        elWx[ei] = wx;
                        elWy[ei] = wy;
                        elWz[ei] = wz;
                        passive[ei] = support[i, j, k] >= 0.5f || load[i, j, k] >= 0.5f;

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

            var keCache = new Dictionary<(int, int, int), double[,]>();
            var K0e = new double[nElem][,];
            var elemKeIdx = new int[nElem];
            var uniqueList = new List<double[,]>();
            var keyToIdx = new Dictionary<(int, int, int), int>();

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
                if (!keyToIdx.TryGetValue(key, out int ui))
                {
                    ui = uniqueList.Count;
                    keyToIdx[key] = ui;
                    uniqueList.Add(K0);
                }
                elemKeIdx[e] = ui;
            }

            int numUnique = uniqueList.Count;
            var KeUnique = new float[numUnique * 24 * 24];
            for (int u = 0; u < numUnique; u++)
                for (int a = 0; a < 24; a++)
                    for (int b = 0; b < 24; b++)
                        KeUnique[u * 576 + a * 24 + b] = (float)uniqueList[u][a, b];

            var dofMapFlat = new int[nElem * 24];
            for (int e = 0; e < nElem; e++)
                for (int a = 0; a < 24; a++)
                    dofMapFlat[e * 24 + a] = dofMap[e, a];

            var diag = new float[ndof];
            var fixedMask = new byte[ndof];
            for (int i = 0; i < ndof; i++)
                fixedMask[i] = fixedDof[i] ? (byte)1 : (byte)0;

            var lv = new MgLevel
            {
                NxC = nxc,
                NyC = nyc,
                NzC = nzc,
                NElem = nElem,
                NNodes = nNodes,
                NDof = ndof,
                KeUnique = KeUnique,
                KeIdx = elemKeIdx,
                DofMap = dofMapFlat,
                Diag = diag,
                FixedDof = fixedDof,
                FixedMask = fixedMask,
                ProlongCoarse = Array.Empty<int>(),
                ProlongWeights = Array.Empty<float>(),
                CoarseToFineElems = Array.Empty<int[]>(),
                K0e = K0e,
                Passive = passive,
                DofMapCpu = dofMap,
                ExC = ex,
                EyC = ey,
                EzC = ez
            };

            return lv;
        }

        private static void BuildProlongation(MgLevel fine, MgLevel coarse)
        {
            int nxf = fine.NxC, nyf = fine.NyC, nzf = fine.NzC;
            int nxc = coarse.NxC, nyc = coarse.NyC, nzc = coarse.NzC;

            var fineNid = new int[nxf + 1, nyf + 1, nzf + 1];
            for (int i = 0; i <= nxf; i++)
                for (int j = 0; j <= nyf; j++)
                    for (int k = 0; k <= nzf; k++)
                        fineNid[i, j, k] = -1;

            var fineNodePos = new (int i, int j, int k)[fine.NNodes];
            for (int q = 0; q < fine.NNodes; q++)
                fineNodePos[q] = (-1, -1, -1);

            for (int e = 0; e < fine.NElem; e++)
            {
                int i = fine.ExC[e], j = fine.EyC[e], k = fine.EzC[e];
                int[] cn = {
                    fine.DofMapCpu[e, 0] / 3, fine.DofMapCpu[e, 3] / 3, fine.DofMapCpu[e, 6] / 3, fine.DofMapCpu[e, 9] / 3,
                    fine.DofMapCpu[e, 12] / 3, fine.DofMapCpu[e, 15] / 3, fine.DofMapCpu[e, 18] / 3, fine.DofMapCpu[e, 21] / 3
                };
                int[] I = { i, i + 1, i + 1, i, i, i + 1, i + 1, i };
                int[] J = { j, j, j + 1, j + 1, j, j, j + 1, j + 1 };
                int[] K = { k, k, k, k, k + 1, k + 1, k + 1, k + 1 };
                for (int n = 0; n < 8; n++)
                {
                    int gi = I[n], gj = J[n], gk = K[n];
                    int nid = cn[n];
                    fineNid[gi, gj, gk] = nid;
                    fineNodePos[nid] = (gi, gj, gk);
                }
            }

            var coarseNid = new int[nxc + 1, nyc + 1, nzc + 1];
            for (int i = 0; i <= nxc; i++)
                for (int j = 0; j <= nyc; j++)
                    for (int k = 0; k <= nzc; k++)
                        coarseNid[i, j, k] = -1;

            for (int e = 0; e < coarse.NElem; e++)
            {
                int i = coarse.ExC[e], j = coarse.EyC[e], k = coarse.EzC[e];
                int[] cn = {
                    coarse.DofMapCpu[e, 0] / 3, coarse.DofMapCpu[e, 3] / 3, coarse.DofMapCpu[e, 6] / 3, coarse.DofMapCpu[e, 9] / 3,
                    coarse.DofMapCpu[e, 12] / 3, coarse.DofMapCpu[e, 15] / 3, coarse.DofMapCpu[e, 18] / 3, coarse.DofMapCpu[e, 21] / 3
                };
                int[] I = { i, i + 1, i + 1, i, i, i + 1, i + 1, i };
                int[] J = { j, j, j + 1, j + 1, j, j, j + 1, j + 1 };
                int[] K = { k, k, k, k, k + 1, k + 1, k + 1, k + 1 };
                for (int n = 0; n < 8; n++)
                    coarseNid[I[n], J[n], K[n]] = cn[n];
            }

            int fdof = fine.NDof;
            var pc = new int[fdof * 8];
            var pw = new float[fdof * 8];
            for (int i = 0; i < pc.Length; i++) pc[i] = -1;

            for (int fn = 0; fn < fine.NNodes; fn++)
            {
                var (If, Jf, Kf) = fineNodePos[fn];
                if (If < 0) continue;

                double uc = If * 0.5;
                double vc = Jf * 0.5;
                double wc = Kf * 0.5;
                int I0 = (int)Math.Floor(uc);
                int J0 = (int)Math.Floor(vc);
                int K0 = (int)Math.Floor(wc);
                double fx = uc - I0;
                double fy = vc - J0;
                double fz = wc - K0;

                double w000 = (1 - fx) * (1 - fy) * (1 - fz);
                double w100 = fx * (1 - fy) * (1 - fz);
                double w010 = (1 - fx) * fy * (1 - fz);
                double w110 = fx * fy * (1 - fz);
                double w001 = (1 - fx) * (1 - fy) * fz;
                double w101 = fx * (1 - fy) * fz;
                double w011 = (1 - fx) * fy * fz;
                double w111 = fx * fy * fz;

                int[] cI = { I0, I0 + 1, I0, I0 + 1, I0, I0 + 1, I0, I0 + 1 };
                int[] cJ = { J0, J0, J0 + 1, J0 + 1, J0, J0, J0 + 1, J0 + 1 };
                int[] cK = { K0, K0, K0, K0, K0 + 1, K0 + 1, K0 + 1, K0 + 1 };
                double[] ww = { w000, w100, w010, w110, w001, w101, w011, w111 };

                for (int c = 0; c < 3; c++)
                {
                    int fd = fn * 3 + c;
                    int base8 = fd * 8;
                    for (int k = 0; k < 8; k++)
                    {
                        int ci = cI[k], cj = cJ[k], ck = cK[k];
                        if (ci < 0 || ci > nxc || cj < 0 || cj > nyc || ck < 0 || ck > nzc)
                        {
                            pc[base8 + k] = -1;
                            pw[base8 + k] = 0f;
                            continue;
                        }
                        int cnid = coarseNid[ci, cj, ck];
                        if (cnid < 0)
                        {
                            pc[base8 + k] = -1;
                            pw[base8 + k] = 0f;
                        }
                        else
                        {
                            pc[base8 + k] = cnid * 3 + c;
                            pw[base8 + k] = (float)ww[k];
                        }
                    }
                }
            }

            fine.ProlongCoarse = pc;
            fine.ProlongWeights = pw;

            var elemMap = new int[nxc, nyc, nzc];
            for (int i = 0; i < nxc; i++)
                for (int j = 0; j < nyc; j++)
                    for (int k = 0; k < nzc; k++)
                        elemMap[i, j, k] = -1;
            for (int e = 0; e < coarse.NElem; e++)
                elemMap[coarse.ExC[e], coarse.EyC[e], coarse.EzC[e]] = e;

            coarse.CoarseToFineElems = new int[coarse.NElem][];
            for (int e = 0; e < coarse.NElem; e++)
                coarse.CoarseToFineElems[e] = Array.Empty<int>();

            var buckets = new List<int>[coarse.NElem];
            for (int e = 0; e < coarse.NElem; e++)
                buckets[e] = new List<int>();

            for (int ef = 0; ef < fine.NElem; ef++)
            {
                int ic = fine.ExC[ef] / 2;
                int jc = fine.EyC[ef] / 2;
                int kc = fine.EzC[ef] / 2;
                if (ic >= nxc || jc >= nyc || kc >= nzc) continue;
                int ec = elemMap[ic, jc, kc];
                if (ec >= 0)
                    buckets[ec].Add(ef);
            }

            for (int e = 0; e < coarse.NElem; e++)
                coarse.CoarseToFineElems[e] = buckets[e].ToArray();
        }

        public static void RebuildDiag(MgLevel lv, double[] rho, bool[] passive, double emin, double p)
        {
            Array.Clear(lv.Diag, 0, lv.NDof);
            for (int e = 0; e < lv.NElem; e++)
            {
                double r = passive[e] ? 1.0 : emin + Math.Pow(rho[e], p) * (1.0 - emin);
                for (int a = 0; a < 24; a++)
                {
                    int g = lv.DofMapCpu[e, a];
                    lv.Diag[g] += (float)(r * lv.K0e[e][a, a]);
                }
            }
            for (int i = 0; i < lv.NDof; i++)
                if (lv.FixedDof[i])
                    lv.Diag[i] += (float)FixedPenalty;
        }

        public static void PropagateRho(MgLevel fine, MgLevel coarse, double[] rhoFine, double[] rhoCoarse)
        {
            for (int ec = 0; ec < coarse.NElem; ec++)
            {
                var ch = coarse.CoarseToFineElems[ec];
                if (ch == null || ch.Length == 0)
                {
                    rhoCoarse[ec] = 1.0;
                    continue;
                }
                double maxRho = 0;
                foreach (int ef in ch)
                    maxRho = Math.Max(maxRho, rhoFine[ef]);
                rhoCoarse[ec] = maxRho;
            }
        }
    }
}
