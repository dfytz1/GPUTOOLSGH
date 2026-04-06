using GHGPUPlugin.Chromodoris.Topology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Diagnostics;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Laplace |∇φ| density blended with a proximity field (support/load + optional box center).
    /// </summary>
    public class LaplaceProximityDensityComponent : GH_Component
    {
        public LaplaceProximityDensityComponent()
          : base("Laplace + Proximity Density GPU", "LaplaceProxGPU",
              "Same Laplace field as Laplace Field Density, plus a blend toward exp(-d/R) from supports/loads " +
              "and optional exp(-d/Rc) from the box center. Rsl≤0 disables SL term (center only); Rsl=0 auto (≈15% box diagonal).",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("InsideMask", "I", "Domain mask.", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupportMask", "S", "Support voxels (Dirichlet Vs).", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Load voxels (Dirichlet Vl).", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Iterations", "N", "Jacobi iterations for Laplace.", GH_ParamAccess.item, 400);
            pManager.AddNumberParameter("SupportPotential", "Vs", "Fixed scalar on support voxels.", GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("LoadPotential", "Vl", "Fixed scalar on load voxels.", GH_ParamAccess.item, 1.0);
            pManager.AddBooleanParameter("InvertLaplace", "Inv", "Invert normalized |∇φ| before blend.", GH_ParamAccess.item, false);
            pManager.AddNumberParameter("ContrastExponent", "E", "Exponent on final blended density (1 = linear).", GH_ParamAccess.item, 1.0);
            pManager.AddNumberParameter("ProximityMix", "Pmx", "0 = pure |∇φ|; 1 = pure proximity field.", GH_ParamAccess.item, 0.35);
            pManager.AddNumberParameter("SupportLoadRadius", "Rsl",
                "World units: exp falloff from nearest S∪L. 0 = auto (~15% diagonal). <0 = omit SL term (use center only).",
                GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("CenterWeight", "Wc", "Weight for box-center proximity (0 = off that term).", GH_ParamAccess.item, 0.25);
            pManager.AddNumberParameter("CenterRadius", "Rc", "World falloff from box center; 0 = disable center term.", GH_ParamAccess.item, 0.0);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Use Metal for Laplace, gradient, and normalize/contrast passes. CPU fallback if unavailable.", GH_ParamAccess.item, true);
            pManager[12].Optional = true;
            pManager[13].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Potential", "Phi", "float[x,y,z] — Laplace field.", GH_ParamAccess.item);
            pManager.AddGenericParameter("Density", "D", "float[x,y,z] — blended density for Build IsoSurface.", GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Passthrough for Build IsoSurface.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Box box = new Box();
            float[,,] inside = null, support = null, load = null;
            int iterations = 400;
            double vs = 0, vl = 1, contrast = 1, pmx = 0.35, rslInput = 0, wc = 0.25, rc = 0;
            bool inv = false;

            if (!DA.GetData(0, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask")) return;
            DA.GetData(4, ref iterations);
            DA.GetData(5, ref vs);
            DA.GetData(6, ref vl);
            DA.GetData(7, ref inv);
            DA.GetData(8, ref contrast);
            DA.GetData(9, ref pmx);
            DA.GetData(10, ref rslInput);
            DA.GetData(11, ref wc);
            DA.GetData(12, ref rc);
            bool useGpu = true;
            DA.GetData(13, ref useGpu);
            NativeLoader.EnsureLoaded();

            int nx = inside.GetLength(0), ny = inside.GetLength(1), nz = inside.GetLength(2);
            if (support.GetLength(0) != nx || load.GetLength(0) != nx ||
                support.GetLength(1) != ny || load.GetLength(1) != ny ||
                support.GetLength(2) != nz || load.GetLength(2) != nz)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "InsideMask, SupportMask, and LoadMask must match dimensions.");
                return;
            }

            if (iterations < 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
                return;
            }

            double dx = box.X.Length / nx;
            double dy = box.Y.Length / ny;
            double dz = box.Z.Length / nz;

            int total = nx * ny * nz;
            float iDx = (float)(1.0 / dx), iDy = (float)(1.0 / dy), iDz = (float)(1.0 / dz);
            float[] fIn = VoxelGpuHelper.Flatten(inside);
            float[] fSup = VoxelGpuHelper.Flatten(support);
            float[] fLoa = VoxelGpuHelper.Flatten(load);
            float[] fPhi = new float[total];
            var sw = Stopwatch.StartNew();

            bool gpuPhi = useGpu && VoxelGpuHelper.TryLaplaceGpu(
                this, fIn, fSup, fLoa, fPhi, nx, ny, nz, (float)vs, (float)vl, iterations);
            float[,,] phi = gpuPhi
                ? VoxelGpuHelper.Unflatten(fPhi, nx, ny, nz)
                : WorkflowAGrid.SolveLaplace(inside, support, load, nx, ny, nz, iterations, (float)vs, (float)vl);

            float[] fGrad = new float[total];
            bool gpuGrad = useGpu && gpuPhi && VoxelGpuHelper.TryGradientGpu(
                this, fPhi, fIn, fGrad, nx, ny, nz, iDx, iDy, iDz);
            if (!gpuGrad)
                fGrad = VoxelGpuHelper.Flatten(WorkflowAGrid.GradientMagnitude(phi, inside, nx, ny, nz, dx, dy, dz));

            sw.Stop();
            if (gpuPhi)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    $"GPU Laplace+Gradient+normalize ({sw.ElapsedMilliseconds} ms)");

            double mix = Rhino.RhinoMath.Clamp(pmx, 0.0, 1.0);
            float[,,] density;

            if (mix < 1e-12)
            {
                float[] fDen = (float[])fGrad.Clone();
                VoxelGpuHelper.DomainMinMax(fDen, fIn, total, out float dMin, out float dMax);
                bool gpuNorm = useGpu && VoxelGpuHelper.TryNormalizeGpu(
                    this, fDen, fIn, nx, ny, nz, dMin, dMax, inv, (float)contrast);
                if (gpuNorm)
                    density = VoxelGpuHelper.Unflatten(fDen, nx, ny, nz);
                else
                {
                    density = VoxelGpuHelper.Unflatten((float[])fGrad.Clone(), nx, ny, nz);
                    WorkflowAGrid.NormalizeToUnitInterval(density, inside, nx, ny, nz, inv);
                    WorkflowAGrid.ApplyContrast(density, inside, nx, ny, nz, contrast);
                }
            }
            else
            {
                float[] fLap = (float[])fGrad.Clone();
                VoxelGpuHelper.DomainMinMax(fLap, fIn, total, out float lMin, out float lMax);
                bool gpuLapNorm = useGpu && VoxelGpuHelper.TryNormalizeGpu(
                    this, fLap, fIn, nx, ny, nz, lMin, lMax, inv, 1f);
                float[,,] laplaceD;
                if (gpuLapNorm)
                    laplaceD = VoxelGpuHelper.Unflatten(fLap, nx, ny, nz);
                else
                {
                    laplaceD = VoxelGpuHelper.Unflatten((float[])fGrad.Clone(), nx, ny, nz);
                    WorkflowAGrid.NormalizeToUnitInterval(laplaceD, inside, nx, ny, nz, inv);
                }

                float[] fGradNorm = gpuLapNorm ? fLap : VoxelGpuHelper.Flatten(laplaceD);

                BoundingBox bb = box.BoundingBox;
                double sx = bb.Max.X - bb.Min.X, sy = bb.Max.Y - bb.Min.Y, sz = bb.Max.Z - bb.Min.Z;
                double diag = Math.Sqrt(sx * sx + sy * sy + sz * sz);
                bool includeSl = rslInput >= 0;
                double rslWorld = rslInput > 0 ? rslInput : (0.15 * diag);
                if (!includeSl)
                    rslWorld = 1.0;

                float[,,] dSl = ProximityDensityBlend.MinDistanceToSupportLoadWorld(box, inside, support, load, nx, ny, nz);
                float[] fDist = VoxelGpuHelper.Flatten(dSl);

                float pMax = MaxRawProximity(dSl, inside, box, nx, ny, nz, includeSl, rslWorld, wc, rc);
                if (pMax < 1e-30f)
                    pMax = 1f;
                float invProxMax = 1f / pMax;

                var proxParams = new float[24];
                FillProximityBlendParams(proxParams, mix, contrast, invProxMax, includeSl, rslWorld, wc, rc, box, nx, ny, nz);

                float[,,]? densityMix = null;
                bool gpuBlend = false;
                if (useGpu && MetalSharedContext.TryGetContext(out IntPtr ctxPb))
                {
                    var densityFlat = new float[total];
                    int cBlend = MetalBridge.ProximityBlend(ctxPb, fGradNorm, fDist, fIn, densityFlat, proxParams, nx, ny, nz);
                    if (cBlend == 0)
                    {
                        densityMix = VoxelGpuHelper.Unflatten(densityFlat, nx, ny, nz);
                        gpuBlend = true;
                    }
                }

                if (!gpuBlend)
                {
                    densityMix = new float[nx, ny, nz];
                    var proximity = new float[nx, ny, nz];
                    ProximityDensityBlend.FillProximityField(proximity, inside, dSl, box, nx, ny, nz, includeSl, rslWorld, wc, rc);

                    var proxNorm = (float[,,])proximity.Clone();
                    float[] fProxN = VoxelGpuHelper.Flatten(proxNorm);
                    VoxelGpuHelper.DomainMinMax(fProxN, fIn, total, out float pMin, out float pMax2);
                    bool gpuProxNorm = useGpu && VoxelGpuHelper.TryNormalizeGpu(
                        this, fProxN, fIn, nx, ny, nz, pMin, pMax2, false, 1f);
                    float[,,] proxNormArr;
                    if (gpuProxNorm)
                        proxNormArr = VoxelGpuHelper.Unflatten(fProxN, nx, ny, nz);
                    else
                    {
                        WorkflowAGrid.NormalizeToUnitInterval(proxNorm, inside, nx, ny, nz, false);
                        proxNormArr = proxNorm;
                    }

                    ProximityDensityBlend.Blend(laplaceD, proxNormArr, inside, nx, ny, nz, mix, densityMix);

                    float[] fDenOut = VoxelGpuHelper.Flatten(densityMix);
                    bool gpuContrast = useGpu && VoxelGpuHelper.TryNormalizeGpu(
                        this, fDenOut, fIn, nx, ny, nz, 0f, 1f, false, (float)contrast);
                    if (gpuContrast)
                        densityMix = VoxelGpuHelper.Unflatten(fDenOut, nx, ny, nz);
                    else
                        WorkflowAGrid.ApplyContrast(densityMix, inside, nx, ny, nz, contrast);
                }

                density = densityMix!;
            }

            DA.SetData(0, new GH_ObjectWrapper(phi));
            DA.SetData(1, new GH_ObjectWrapper(density));
            DA.SetData(2, box);

            if (mix > 1e-12 && wc <= 1e-12 && rslInput < 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Rsl<0 disables S∪L term and Wc=0 disables center — proximity field is zero.");
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.LaplaceProximity;

        public override Guid ComponentGuid => new Guid("833129d1-704a-45b6-8da7-d96b7fa0e8fb");

        private static float MaxRawProximity(float[,,] dSl, float[,,] inside, Box box, int nx, int ny, int nz,
            bool includeSl, double rslWorld, double wc, double rc)
        {
            float pMax = 0f;
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        if (inside[ix, iy, iz] < 0.5f)
                            continue;
                        float p = RawProximityAt(dSl[ix, iy, iz], inside, ix, iy, iz, box, nx, ny, nz, includeSl, rslWorld, wc, rc);
                        if (p > pMax)
                            pMax = p;
                    }
                }
            }

            return pMax;
        }

        private static float RawProximityAt(
            float dSlVal,
            float[,,] inside,
            int ix,
            int iy,
            int iz,
            Box box,
            int nx,
            int ny,
            int nz,
            bool includeSl,
            double rslWorld,
            double wc,
            double rc)
        {
            if (inside[ix, iy, iz] < 0.5f)
                return 0f;
            float pSl = 0f;
            if (includeSl && rslWorld > 1e-12)
            {
                if (dSlVal < float.MaxValue * 0.5f)
                    pSl = (float)Math.Exp(-dSlVal / rslWorld);
            }

            float p = pSl;
            if (wc > 1e-12 && rc > 1e-12)
            {
                Point3d c = WorkflowAGrid.CellCenterWorld(box, ix, iy, iz, nx, ny, nz);
                Point3d boxCenter = box.BoundingBox.Center;
                double dc = c.DistanceTo(boxCenter);
                float pC = (float)Math.Exp(-dc / rc);
                p = Math.Max(p, (float)(wc * pC));
            }

            return p;
        }

        private static void FillProximityBlendParams(float[] p, double mixVal, double contrastExp, float invProxMax,
            bool includeSl, double rslWorld, double wc, double rc, Box box, int nx, int ny, int nz)
        {
            p[0] = (float)mixVal;
            p[1] = (float)contrastExp;
            p[2] = invProxMax;
            p[3] = includeSl && rslWorld > 1e-12 ? (float)(1.0 / rslWorld) : 0f;
            p[4] = (float)wc;
            p[5] = wc > 1e-12 && rc > 1e-12 ? (float)(1.0 / rc) : 0f;
            p[6] = includeSl ? 1f : 0f;
            p[7] = wc > 1e-12 && rc > 1e-12 ? 1f : 0f;
            p[8] = float.MaxValue * 0.25f;
            Point3d o = box.PointAt(0, 0, 0);
            Vector3d ex = (box.PointAt(1, 0, 0) - o) / Math.Max(1, nx);
            Vector3d ey = (box.PointAt(0, 1, 0) - o) / Math.Max(1, ny);
            Vector3d ez = (box.PointAt(0, 0, 1) - o) / Math.Max(1, nz);
            p[9] = (float)o.X;
            p[10] = (float)o.Y;
            p[11] = (float)o.Z;
            p[12] = (float)ex.X;
            p[13] = (float)ex.Y;
            p[14] = (float)ex.Z;
            p[15] = (float)ey.X;
            p[16] = (float)ey.Y;
            p[17] = (float)ey.Z;
            p[18] = (float)ez.X;
            p[19] = (float)ez.Y;
            p[20] = (float)ez.Z;
            Point3d bc = box.BoundingBox.Center;
            p[21] = (float)bc.X;
            p[22] = (float)bc.Y;
            p[23] = (float)bc.Z;
        }
    }
}
