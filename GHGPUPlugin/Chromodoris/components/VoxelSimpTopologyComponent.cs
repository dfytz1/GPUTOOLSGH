using GHGPUPlugin.Chromodoris.Topology;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Phase-1 SIMP: minimize linear compliance on a voxel hex mesh (no density filter yet).
    /// </summary>
    /// <remarks>CPU-only SIMP / PCG; shipped as part of GHGPUPlugin (MetalGH).</remarks>
    public class VoxelSimpTopologyComponent : GH_Component
    {
        public VoxelSimpTopologyComponent()
          : base("Voxel SIMP Topology GPU", "VoxelSIMPGPU",
              "SIMP on a coarse voxel stride (fast), then trilinear upsample to mask resolution for smooth iso. " +
              "Same masks as Laplace workflow. Not validated FEA. Use SolveStride 2–4 for speed.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("InsideMask", "I", "Domain mask from Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupportMask", "S", "Fixed-displacement voxels (same as Voxel Paint S).", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Loaded voxels; total force is distributed over their nodes.", GH_ParamAccess.item);
            pManager.AddNumberParameter("VolumeFraction", "Vf", "Target mean SIMP design variable on free voxels (0–1).", GH_ParamAccess.item, 0.3);
            pManager.AddIntegerParameter("OuterIterations", "It", "SIMP / OC iterations.", GH_ParamAccess.item, 30);
            pManager.AddIntegerParameter("PcgIterations", "Pcg", "Max PCG steps per linear solve.", GH_ParamAccess.item, 800);
            pManager.AddNumberParameter("SimpPenalty", "P", "SIMP exponent (typically 3).", GH_ParamAccess.item, 3.0);
            pManager.AddNumberParameter("MoveLimit", "M", "Max change of design variable per iteration.", GH_ParamAccess.item, 0.2);
            pManager.AddNumberParameter("VoidStiffness", "Emin", "Relative void stiffness Emin/E0 (e.g. 1e-6).", GH_ParamAccess.item, 1e-6);
            pManager.AddNumberParameter("Poisson", "Nu", "Poisson's ratio.", GH_ParamAccess.item, 0.3);
            pManager.AddIntegerParameter("MaxElements", "Max", "Abort if solid voxel count exceeds this (speed).", GH_ParamAccess.item, 40000);
            pManager.AddNumberParameter("ForceX", "Fx", "Total force X component (model units).", GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("ForceY", "Fy", "Total force Y component.", GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("ForceZ", "Fz", "Total force Z component.", GH_ParamAccess.item, -1.0);
            pManager.AddIntegerParameter("SolveStride", "Str",
                "Coarse solve: 1 = full grid; 2+ = merge S×S×S fine cells (faster, upsampled ρ for iso).",
                GH_ParamAccess.item, 2);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Density", "R",
                "float[x,y,z] at mask resolution — upsampled from coarse SIMP. Build IsoSurface (Cc=True); IsoValue ~0.2–0.5.",
                GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Passthrough for Build IsoSurface.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Compliance", "C", "Last linear compliance f·u (relative units).", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Box box = new Box();
            float[,,] inside = null, support = null, load = null;
            double vf = 0.3;
            int outer = 30, pcg = 800, maxEl = 40000, solveStride = 2;
            double simpP = 3, move = 0.2, emin = 1e-6, nu = 0.3;
            double fx = 0, fy = 0, fz = -1;

            if (!DA.GetData(0, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask")) return;
            DA.GetData(4, ref vf);
            DA.GetData(5, ref outer);
            DA.GetData(6, ref pcg);
            DA.GetData(7, ref simpP);
            DA.GetData(8, ref move);
            DA.GetData(9, ref emin);
            DA.GetData(10, ref nu);
            DA.GetData(11, ref maxEl);
            DA.GetData(12, ref fx);
            DA.GetData(13, ref fy);
            DA.GetData(14, ref fz);
            DA.GetData(15, ref solveStride);

            int nx = inside.GetLength(0), ny = inside.GetLength(1), nz = inside.GetLength(2);
            if (support.GetLength(0) != nx || load.GetLength(0) != nx)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mask dimensions must match.");
                return;
            }

            if (vf <= 0 || vf > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VolumeFraction must be in (0,1].");
                return;
            }

            if (outer < 1 || pcg < 10)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Invalid iteration counts.");
                return;
            }

            if (solveStride < 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SolveStride must be >= 1.");
                return;
            }

            double dx = box.X.Length / nx;
            double dy = box.Y.Length / ny;
            double dz = box.Z.Length / nz;

            var force = new Vector3d(fx, fy, fz);
            if (force.Length < 1e-20)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Force vector is (near) zero.");
                return;
            }

            VoxelSimpOptimizer.Result res;
            try
            {
                res = VoxelSimpOptimizer.Run(
                    inside, support, load, dx, dy, dz, force, vf,
                    outer, pcg, simpP, move, emin, nu, maxEl, solveStride);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
                return;
            }

            if (res.Message != "OK")
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, res.Message);
                return;
            }

            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                "Fast path: elastic SIMP on coarse stride, ρ upsampled to mask res — not sign-off FEA. No density filter.");

            DA.SetData(0, new GH_ObjectWrapper(res.DensityPhys));
            DA.SetData(1, box);
            DA.SetData(2, res.Compliance);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => null;

        public override Guid ComponentGuid => new Guid("30340f6b-7086-453c-9b94-53dca394329c");
    }
}
