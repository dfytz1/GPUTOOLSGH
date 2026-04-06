/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using GHGPUPlugin.Chromodoris.Topology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using System;
using System.Diagnostics;

namespace GHGPUPlugin.Chromodoris
{
    public class CloseVoxelGridComponent : GH_Component
    {
        public CloseVoxelGridComponent()
          : base("Close Voxel Data GPU", "CloseVoxelsGPU",
              "Caps voxel boundary values to zero, ensuring the resulting isosurface is a closed volume.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("VoxelData", "D", "The voxel data (float[x,y,z]) to close.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Use Metal GPU to zero boundary voxels. CPU fallback if unavailable.", GH_ParamAccess.item, true);
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("VoxelData", "D", "The closed voxel data — boundary cells are set to zero.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 0, this, out float[,,] inputData, "Voxel data"))
                return;

            bool useGpu = true;
            DA.GetData(1, ref useGpu);
            NativeLoader.EnsureLoaded();

            int nx = inputData.GetLength(0), ny = inputData.GetLength(1), nz = inputData.GetLength(2);
            float[,,] result = (float[,,])inputData.Clone();
            bool gpuOk = false;
            var sw = Stopwatch.StartNew();

            if (useGpu)
            {
                float[] flat = VoxelGpuHelper.Flatten(result);
                if (VoxelGpuHelper.TryZeroBoundaryGpu(this, flat, nx, ny, nz))
                {
                    result = VoxelGpuHelper.Unflatten(flat, nx, ny, nz);
                    gpuOk = true;
                }
            }

            if (!gpuOk)
                WorkflowAGrid.ZeroVoxelBoundaryInPlace(result);

            sw.Stop();
            if (gpuOk)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    $"GPU boundary zero ({sw.ElapsedMilliseconds} ms)");

            DA.SetData(0, new GH_ObjectWrapper(result));
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.CloseVolume;

        public override Guid ComponentGuid => new Guid("8f87ef6e-8a4e-4c1e-a75a-a6cd753701af");
    }
}
