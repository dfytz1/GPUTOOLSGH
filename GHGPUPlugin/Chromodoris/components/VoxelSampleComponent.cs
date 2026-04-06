/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace GHGPUPlugin.Chromodoris
{
    public class VoxelSampleComponent : GH_Component
    {
        public VoxelSampleComponent()
          : base("Sample Voxels", "VoxelSample",
              "Construct and sample a voxel grid from a point cloud and optional charges.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Points", "P", "Points to sample.", GH_ParamAccess.list);
            pManager.AddNumberParameter("Charges", "C", "Charge values corresponding to each point. Leave empty for uniform charge of 1.", GH_ParamAccess.list);
            pManager.AddNumberParameter("VoxelSize", "S", "Size of each voxel cell.", GH_ParamAccess.item);
            pManager.AddNumberParameter("EffectiveRange", "R", "The maximum search range for voxel sampling.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("DensitySampling", "D", "Toggle point density affecting the point values (bulge mode).", GH_ParamAccess.item, false);
            pManager.AddBooleanParameter("LinearFalloff", "L", "Toggle falloff from exponential to linear.", GH_ParamAccess.item, true);
            pManager.AddNumberParameter("TimeoutSeconds", "T", "Maximum allowed computation time in seconds. Set to 0 to disable.", GH_ParamAccess.item, 0.0);
            pManager[1].Optional = true;
            pManager[4].Optional = true;
            pManager[5].Optional = true;
            pManager[6].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "The generated box representing the voxel grid extents.", GH_ParamAccess.item);
            pManager.AddGenericParameter("VoxelData", "D", "Voxel data as float[x,y,z] — pass to Build IsoSurface.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Point3d> points = new List<Point3d>();
            List<double> charges = new List<double>();
            double cellSize = 0;
            double range = 0;
            bool bulge = false;
            bool linear = true;
            double timeoutSeconds = 0;

            if (!DA.GetDataList(0, points)) return;
            DA.GetDataList(1, charges);
            if (!DA.GetData(2, ref cellSize)) return;
            if (!DA.GetData(3, ref range)) return;
            DA.GetData(4, ref bulge);
            DA.GetData(5, ref linear);
            DA.GetData(6, ref timeoutSeconds);

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

            if (cellSize <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelSize must be greater than 0.");
                return;
            }

            if (charges.Count != 0 && charges.Count != 1 && charges.Count != points.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "Charges count should be 0, 1, or equal to the number of points. Using uniform charge.");
            }

            VoxelSampler sampler = new VoxelSampler(points, charges, cellSize, range, bulge, linear);
            sampler.Initialize();

            bool useInverse = points.Count < sampler.xRes * sampler.yRes * sampler.zRes / 2;

            if (timeoutSeconds > 0)
            {
                var task = Task.Run(() =>
                {
                    if (useInverse)
                        sampler.ExecuteInverseMultiThread();
                    else
                        sampler.ExecuteMultiThread();
                });

                bool completed = task.Wait(TimeSpan.FromSeconds(timeoutSeconds));

                if (!completed)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        string.Format("Computation exceeded {0:0.##}s timeout and was aborted. " +
                            "Try increasing VoxelSize, reducing EffectiveRange, or raising the timeout.", timeoutSeconds));
                    return;
                }
            }
            else
            {
                if (useInverse)
                    sampler.ExecuteInverseMultiThread();
                else
                    sampler.ExecuteMultiThread();
            }

            DA.SetData(0, sampler.Box);
            DA.SetData(1, sampler.Data);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.Sample;

        public override Guid ComponentGuid => new Guid("e6e6b720-ea99-40fc-8d10-8e48f17baf4c");
    }
}
