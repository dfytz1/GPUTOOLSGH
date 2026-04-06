/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace GHGPUPlugin.Chromodoris
{
    public class VoxelSampleCustomComponent : GH_Component
    {
        public VoxelSampleCustomComponent()
          : base("Sample Voxels GPU (Custom)", "VoxelSampleCustomGPU",
              "Construct and sample a voxel grid from a point cloud using a custom box and explicit resolution.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Points", "P", "Points to sample.", GH_ParamAccess.list);
            pManager.AddNumberParameter("Charges", "C", "Charge values corresponding to each point. Leave empty for uniform charge of 1.", GH_ParamAccess.list);
            pManager.AddBoxParameter("BoundingBox", "B", "The box defining the voxel grid region.", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResolutionX", "X", "Number of grid cells in the X direction.", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResolutionY", "Y", "Number of grid cells in the Y direction.", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResolutionZ", "Z", "Number of grid cells in the Z direction.", GH_ParamAccess.item);
            pManager.AddNumberParameter("EffectiveRange", "R", "The maximum search range for voxel sampling.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("DensitySampling", "D", "Toggle point density affecting the point values (bulge mode).", GH_ParamAccess.item, false);
            pManager.AddBooleanParameter("LinearFalloff", "L", "Toggle falloff from exponential to linear.", GH_ParamAccess.item, true);
            pManager.AddNumberParameter("TimeoutSeconds", "T", "Maximum allowed computation time in seconds. Set to 0 to disable.", GH_ParamAccess.item, 0.0);
            pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal voxel sampling when available (axis-aligned box only).", GH_ParamAccess.item, true);
            pManager[1].Optional = true;
            pManager[7].Optional = true;
            pManager[8].Optional = true;
            pManager[9].Optional = true;
            pManager[10].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "The box representing the voxel grid extents.", GH_ParamAccess.item);
            pManager.AddGenericParameter("VoxelData", "D", "Voxel data as float[x,y,z] — pass to Build IsoSurface.", GH_ParamAccess.item);
        }

        private static bool IsAxisAlignedWorldBox(Box b)
        {
            return b.Plane.ZAxis.IsParallelTo(Vector3d.ZAxis) != 0
                && b.Plane.XAxis.IsParallelTo(Vector3d.XAxis) != 0;
        }

        private static void NormalizeChargesList(List<double> ch, int nPts)
        {
            if (ch.Count == 1)
            {
                double v = ch[0];
                ch.Clear();
                for (int i = 0; i < nPts; i++)
                    ch.Add(v);
            }
            else if (ch.Count == 0 || ch.Count < nPts)
            {
                ch.Clear();
                for (int i = 0; i < nPts; i++)
                    ch.Add(1.0);
            }
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Point3d> points = new List<Point3d>();
            List<double> charges = new List<double>();
            Box box = new Box();
            int xr = 0, yr = 0, zr = 0;
            double range = 0;
            bool bulge = false;
            bool linear = true;
            double timeoutSeconds = 0;
            bool useGpu = true;

            if (!DA.GetDataList(0, points)) return;
            DA.GetDataList(1, charges);
            if (!DA.GetData(2, ref box)) return;
            if (!DA.GetData(3, ref xr)) return;
            if (!DA.GetData(4, ref yr)) return;
            if (!DA.GetData(5, ref zr)) return;
            if (!DA.GetData(6, ref range)) return;
            DA.GetData(7, ref bulge);
            DA.GetData(8, ref linear);
            DA.GetData(9, ref timeoutSeconds);
            DA.GetData(10, ref useGpu);
            NativeLoader.EnsureLoaded();

            if (points.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No points provided.");
                return;
            }

            if (range <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "EffectiveRange must be greater than 0.");
                return;
            }

            if (xr <= 1 || yr <= 1 || zr <= 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Resolution values must be greater than 1.");
                return;
            }

            if (charges.Count != 0 && charges.Count != 1 && charges.Count != points.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "Charges count should be 0, 1, or equal to the number of points. Using uniform charge.");
            }

            VoxelSampler sampler = new VoxelSampler(points, charges, box, xr, yr, zr, range, bulge, linear);
            sampler.Initialize();

            bool gpuDone = false;
            if (useGpu && timeoutSeconds <= 0 && MetalSharedContext.TryGetContext(out IntPtr ctx))
            {
                try
                {
                    Box sb = sampler.Box;
                    if (IsAxisAlignedWorldBox(sb))
                    {
                        var chg = new List<double>(charges);
                        NormalizeChargesList(chg, points.Count);
                        int nx = sampler.xRes;
                        int ny = sampler.yRes;
                        int nz = sampler.zRes;
                        BoundingBox bb = sb.BoundingBox;
                        float minX = (float)bb.Min.X;
                        float minY = (float)bb.Min.Y;
                        float minZ = (float)bb.Min.Z;
                        float sx = (float)(bb.Max.X - bb.Min.X);
                        float sy = (float)(bb.Max.Y - bb.Min.Y);
                        float sz = (float)(bb.Max.Z - bb.Min.Z);
                        float dx = sx / Math.Max(1, nx);
                        float dy = sy / Math.Max(1, ny);
                        float dz = sz / Math.Max(1, nz);

                        int n = points.Count;
                        var ptX = new float[n];
                        var ptY = new float[n];
                        var ptZ = new float[n];
                        var chf = new float[n];
                        for (int i = 0; i < n; i++)
                        {
                            ptX[i] = (float)points[i].X;
                            ptY[i] = (float)points[i].Y;
                            ptZ[i] = (float)points[i].Z;
                            chf[i] = (float)chg[i];
                        }

                        int total = nx * ny * nz;
                        var grid = new float[total];
                        int code = MetalBridge.VoxelSample(ctx, ptX, ptY, ptZ, chf, grid,
                            minX, minY, minZ, dx, dy, dz, nx, ny, nz, n, (float)range, linear ? 1 : 0, bulge ? 1 : 0);
                        if (code == 0)
                        {
                            DA.SetData(0, sb);
                            DA.SetData(1, new GH_ObjectWrapper(VoxelGpuHelper.Unflatten(grid, nx, ny, nz)));
                            gpuDone = true;
                        }
                    }
                }
                catch (Exception)
                {
                }
            }

            if (!gpuDone)
            {
                if (timeoutSeconds > 0)
                {
                    var task = Task.Run(() => sampler.ExecuteMultiThread());

                    bool completed = task.Wait(TimeSpan.FromSeconds(timeoutSeconds));

                    if (!completed)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                            string.Format("Computation exceeded {0:0.##}s timeout and was aborted. " +
                                "Try reducing resolution, EffectiveRange, or raising the timeout.", timeoutSeconds));
                        return;
                    }
                }
                else
                {
                    sampler.ExecuteMultiThread();
                }

                DA.SetData(0, sampler.Box);
                DA.SetData(1, sampler.Data);
            }
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.SampleCustom;

        public override Guid ComponentGuid => new Guid("5a31f82c-12d1-4ce1-9a6d-662beb67971b");
    }
}
