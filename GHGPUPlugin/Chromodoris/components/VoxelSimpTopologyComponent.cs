using GHGPUPlugin.Chromodoris.Topology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Phase-1 SIMP: minimize linear compliance on a voxel hex mesh (no density filter yet).
    /// </summary>
    /// <remarks>SIMP / PCG with optional Metal MatVec; shipped as part of GHGPUPlugin (MetalGH).</remarks>
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
            pManager.AddPointParameter("LoadPoints", "LP", "World-space force application points (parallel to LoadVectors).", GH_ParamAccess.list);
            pManager.AddVectorParameter("LoadVectors", "LV", "Force vectors; with empty LP, summed and spread over LoadMask nodes.", GH_ParamAccess.list);
            pManager.AddNumberParameter("YoungModulus", "E", "Material stiffness. 1=normalised; use 210000 for steel (MPa+mm).", GH_ParamAccess.item, 1.0);
            pManager.AddPointParameter("SupportPoints", "SP", "Per-point directional supports (nearest mesh node); additive with SupportMask.", GH_ParamAccess.list);
            pManager.AddVectorParameter("SupportDirs", "SD",
                "Parallel to SupportPoints: abs(X,Y,Z) greater than 0.5 fixes that global DOF (e.g. 1,1,1 pin, 0,0,1 Z roller).",
                GH_ParamAccess.list);
            pManager.AddIntegerParameter("SolveStride", "Str",
                "Coarse solve: 1 = full grid; 2+ = merge S×S×S fine cells (faster, upsampled ρ for iso).",
                GH_ParamAccess.item, 2);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Use Metal for PCG MatVec (stiffness × vector); CPU fallback if unavailable.", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("RecordHistory", "Rec",
                "If true, store upsampled float[x,y,z] density per outer iteration in DensityHistory (more memory).", GH_ParamAccess.item, false);
            pManager.AddNumberParameter("FilterRadius", "Fr",
                "Sensitivity filter radius in element units (0 = off, 1.5 recommended).", GH_ParamAccess.item, 1.5);
            pManager.AddBooleanParameter("EnforceConnectivity", "Conn",
                "After OC, bridge support–load if no rho-above-0.05 path (from iteration 4 onward).", GH_ParamAccess.item, true);
            pManager[14].Optional = true;
            pManager[15].Optional = true;
            pManager[16].Optional = true;
            pManager[18].Optional = true;
            pManager[19].Optional = true;
            pManager[20].Optional = true;
            pManager[21].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Density", "R",
                "float[x,y,z] at mask resolution — upsampled from coarse SIMP. Build IsoSurface (Cc=True); IsoValue ~0.2–0.5.",
                GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Passthrough for Build IsoSurface.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Compliance", "C", "Last linear compliance f·u (relative units).", GH_ParamAccess.item);
            pManager.AddGenericParameter("DensityHistory", "ρHist",
                "Upsampled density grid per outer iteration (same as R); branch = iteration; one wrapped float[,,] per branch when RecordHistory is true.",
                GH_ParamAccess.tree);
            pManager.AddNumberParameter("IterationCompliance", "CIt",
                "Compliance f·u after each outer iteration (convergence plot).", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // Do not call DA.SetData before reading inputs — that can break Grasshopper's solution
            // propagation (inputs appear dead, especially on GPU solves). Set safe outputs only on exits.
            void FallbackOutputs(Box bbox)
            {
                DA.SetData(0, null);
                DA.SetData(1, bbox);
                DA.SetData(2, 0.0);
                DA.SetDataTree(3, new GH_Structure<IGH_Goo>());
                DA.SetDataList(4, new List<GH_Number>());
            }

            Box box = new Box();
            float[,,] inside = null, support = null, load = null;
            double vf = 0.3;
            int outer = 30, pcg = 800, maxEl = 40000, solveStride = 2;
            bool useGpu = true;
            bool recordHistory = false;
            double filterRadius = 1.5;
            bool enforceConn = true;
            double simpP = 3, move = 0.2, emin = 1e-6, nu = 0.3;
            var loadPts = new List<Point3d>();
            var loadVecs = new List<Vector3d>();
            double youngE = 1.0;
            var supPts = new List<Point3d>();
            var supDirs = new List<Vector3d>();

            if (!DA.GetData(0, ref box))
            {
                FallbackOutputs(new Box());
                return;
            }

            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask"))
            {
                FallbackOutputs(box);
                return;
            }
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask"))
            {
                FallbackOutputs(box);
                return;
            }
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask"))
            {
                FallbackOutputs(box);
                return;
            }

            static int ReadIntCoerce(IGH_DataAccess da, int index, int fallback)
            {
                IGH_Goo goo = null;
                if (!da.GetData(index, ref goo) || goo == null)
                    return fallback;

                if (goo is GH_Integer gi)
                    return gi.Value;
                if (goo is GH_Number gn)
                    return (int)Math.Round(gn.Value);

                object sv = goo.ScriptVariable();
                if (sv is int i)
                    return i;
                if (sv is double d)
                    return (int)Math.Round(d);
                if (sv is float f)
                    return (int)Math.Round(f);

                return fallback;
            }

            DA.GetData(4, ref vf);
            outer = ReadIntCoerce(DA, 5, outer);
            pcg = ReadIntCoerce(DA, 6, pcg);
            DA.GetData(7, ref simpP);
            DA.GetData(8, ref move);
            DA.GetData(9, ref emin);
            DA.GetData(10, ref nu);
            maxEl = ReadIntCoerce(DA, 11, maxEl);
            DA.GetDataList(12, loadPts);
            DA.GetDataList(13, loadVecs);
            DA.GetData(14, ref youngE);
            DA.GetDataList(15, supPts);
            DA.GetDataList(16, supDirs);
            solveStride = ReadIntCoerce(DA, 17, solveStride);
            DA.GetData(18, ref useGpu);
            DA.GetData(19, ref recordHistory);
            DA.GetData(20, ref filterRadius);
            DA.GetData(21, ref enforceConn);

            if (useGpu)
            {
                try
                {
                    NativeLoader.EnsureLoaded();
                    if (!NativeLoader.IsMetalAvailable)
                    {
                        useGpu = false;
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                            "GPU native library unavailable, using CPU: " + (NativeLoader.LoadError ?? "MetalBridge.dylib not loaded."));
                    }
                }
                catch (Exception ex)
                {
                    useGpu = false;
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        "GPU native library unavailable, using CPU: " + ex.Message);
                }
            }

            int nx = inside.GetLength(0), ny = inside.GetLength(1), nz = inside.GetLength(2);
            if (support.GetLength(0) != nx || load.GetLength(0) != nx)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mask dimensions must match.");
                FallbackOutputs(box);
                return;
            }

            if (vf <= 0 || vf > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VolumeFraction must be in (0,1].");
                FallbackOutputs(box);
                return;
            }

            if (outer < 1 || pcg < 10)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Invalid iteration counts.");
                FallbackOutputs(box);
                return;
            }

            if (solveStride < 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SolveStride must be >= 1.");
                FallbackOutputs(box);
                return;
            }

            bool hasLp = loadPts.Count > 0;
            if (hasLp && loadPts.Count != loadVecs.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "LoadPoints and LoadVectors must have the same count.");
                FallbackOutputs(box);
                return;
            }

            if (supPts.Count > 0 && supPts.Count != supDirs.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SupportPoints and SupportDirs must have the same count.");
                FallbackOutputs(box);
                return;
            }

            var sumLv = Vector3d.Zero;
            foreach (Vector3d v in loadVecs)
                sumLv += v;
            if (!hasLp && sumLv.Length < 1e-20)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No loads defined.");
                FallbackOutputs(box);
                return;
            }

            double dx = box.X.Length / nx;
            double dy = box.Y.Length / ny;
            double dz = box.Z.Length / nz;

            VoxelSimpOptimizer.Result res;
            try
            {
                res = VoxelSimpOptimizer.Run(
                    inside, support, load, dx, dy, dz,
                    box, loadPts, loadVecs, youngE, supPts, supDirs,
                    vf,
                    outer, pcg, simpP, move, emin, nu, maxEl, solveStride, useGpu, recordHistory, filterRadius,
                    penaltyContinuation: false, enforceConnectivity: enforceConn);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
                FallbackOutputs(box);
                return;
            }

            if (!string.IsNullOrEmpty(res.Message) && res.Message.StartsWith("GPU_FALLBACK:", StringComparison.Ordinal))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, res.Message);
                // Continue: solve succeeded via CPU fallback.
            }
            else if (!string.IsNullOrEmpty(res.Message) && res.Message.StartsWith("GPU_REMARK:", StringComparison.Ordinal))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.Message);
                // Continue: solve succeeded via CPU fallback.
            }
            else if (res.Message != "OK")
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, res.Message);
                FallbackOutputs(box);
                return;
            }

            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                "Fast path: elastic SIMP on coarse stride, ρ upsampled to mask res — not sign-off FEA. No density filter.");

            if (!string.IsNullOrWhiteSpace(res.DiagMessage))
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.DiagMessage);

            if (!string.IsNullOrWhiteSpace(res.GpuDiagPreSolve))
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.GpuDiagPreSolve);

            var histTree = new GH_Structure<IGH_Goo>();
            if (res.DensityHistory != null)
            {
                for (int i = 0; i < res.DensityHistory.Count; i++)
                    histTree.Append(new GH_ObjectWrapper(res.DensityHistory[i]), new GH_Path(i));
            }

            var cIt = new List<GH_Number>(res.IterationCompliance.Count);
            foreach (double c in res.IterationCompliance)
                cIt.Add(new GH_Number(c));

            DA.SetData(0, new GH_ObjectWrapper(res.DensityPhys));
            DA.SetData(1, box);
            DA.SetData(2, res.Compliance);
            DA.SetDataTree(3, histTree);
            DA.SetDataList(4, cIt);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => null;

        public override Guid ComponentGuid => new Guid("30340f6b-7086-453c-9b94-53dca394329c");
    }
}
