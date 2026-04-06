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
    /// Workflow A — step 3: Laplace field on voxels, then |∇φ| as density for isosurfacing.
    /// </summary>
    public class LaplaceDensityFieldComponent : GH_Component
    {
        public LaplaceDensityFieldComponent()
          : base("Laplace Field Density GPU", "LaplaceDenGPU",
              "Solves a Laplace field (support=0, load=1) inside the domain and builds a normalized density from |∇φ|. Wire B and D to Build IsoSurface.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("InsideMask", "I", "Domain mask.", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupportMask", "S", "Support voxels (Dirichlet 0).", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Load voxels (Dirichlet 1).", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Iterations", "N", "Jacobi iterations for Laplace (e.g. 200–800).", GH_ParamAccess.item, 400);
            pManager.AddNumberParameter("SupportPotential", "Vs", "Fixed scalar on support voxels.", GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("LoadPotential", "Vl", "Fixed scalar on load voxels.", GH_ParamAccess.item, 1.0);
            pManager.AddBooleanParameter("InvertDensity", "Inv", "If true, material proxy = 1 − normalized |∇φ|.", GH_ParamAccess.item, false);
            pManager.AddNumberParameter("ContrastExponent", "E", "Exponent on normalized density (1 = linear, above 1 = sharper).", GH_ParamAccess.item, 1.0);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Use Metal GPU (M-chip). CPU fallback if unavailable.", GH_ParamAccess.item, true);
            pManager[9].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Potential", "Phi", "float[x,y,z] — scalar field after iteration.", GH_ParamAccess.item);
            pManager.AddGenericParameter("Density", "D", "float[x,y,z] — normalized 0..1, suitable for Build IsoSurface iso threshold.", GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Passthrough of the input box — use for Build IsoSurface so B matches D.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Box box = new Box();
            float[,,] inside = null;
            float[,,] support = null;
            float[,,] load = null;
            int iterations = 400;
            double vs = 0;
            double vl = 1;
            bool invert = false;
            double contrast = 1.0;

            if (!DA.GetData(0, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask (Voxel Paint output S)")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask (Voxel Paint output L)")) return;
            DA.GetData(4, ref iterations);
            DA.GetData(5, ref vs);
            DA.GetData(6, ref vl);
            DA.GetData(7, ref invert);
            DA.GetData(8, ref contrast);
            bool useGpu = true;
            DA.GetData(9, ref useGpu);
            NativeLoader.EnsureLoaded();

            if (inside == null || support == null || load == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "One or more masks are null.");
                return;
            }

            int nx = inside.GetLength(0);
            int ny = inside.GetLength(1);
            int nz = inside.GetLength(2);
            if (support.GetLength(0) != nx || load.GetLength(0) != nx ||
                support.GetLength(1) != ny || load.GetLength(1) != ny ||
                support.GetLength(2) != nz || load.GetLength(2) != nz)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "InsideMask, SupportMask, and LoadMask must have identical dimensions.");
                return;
            }

            if (iterations < 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
                return;
            }

            if (Math.Abs(vl - vs) < 1e-12)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SupportPotential and LoadPotential should differ.");
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
            if (!gpuPhi)
                fPhi = VoxelGpuHelper.Flatten(WorkflowAGrid.SolveLaplace(
                    inside, support, load, nx, ny, nz, iterations, (float)vs, (float)vl));

            float[] fGrad = new float[total];
            bool gpuGrad = useGpu && gpuPhi && VoxelGpuHelper.TryGradientGpu(
                this, fPhi, fIn, fGrad, nx, ny, nz, iDx, iDy, iDz);
            if (!gpuGrad)
                fGrad = VoxelGpuHelper.Flatten(WorkflowAGrid.GradientMagnitude(
                    VoxelGpuHelper.Unflatten(fPhi, nx, ny, nz), inside, nx, ny, nz, dx, dy, dz));

            float[] fDen = (float[])fGrad.Clone();
            VoxelGpuHelper.DomainMinMax(fDen, fIn, total, out float dMin, out float dMax);
            bool gpuNorm = useGpu && gpuGrad && VoxelGpuHelper.TryNormalizeGpu(
                this, fDen, fIn, nx, ny, nz, dMin, dMax, invert, contrast);
            if (!gpuNorm)
            {
                var densArr = VoxelGpuHelper.Unflatten(fDen, nx, ny, nz);
                WorkflowAGrid.NormalizeToUnitInterval(densArr, inside, nx, ny, nz, invert);
                WorkflowAGrid.ApplyContrast(densArr, inside, nx, ny, nz, contrast);
                fDen = VoxelGpuHelper.Flatten(densArr);
            }

            sw.Stop();
            if (gpuPhi)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    $"GPU Laplace+Gradient+Normalize ({sw.ElapsedMilliseconds} ms)");

            float[,,] phi = VoxelGpuHelper.Unflatten(fPhi, nx, ny, nz);
            float[,,] density = VoxelGpuHelper.Unflatten(fDen, nx, ny, nz);

            DA.SetData(0, new GH_ObjectWrapper(phi));
            DA.SetData(1, new GH_ObjectWrapper(density));
            DA.SetData(2, box);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.LaplaceField;

        public override Guid ComponentGuid => new Guid("756ff6d3-bf54-408d-83df-3355f73c42d2");
    }
}
