/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using GHGPUPlugin.Chromodoris.Topology;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using System;

namespace GHGPUPlugin.Chromodoris
{
    public class CloseVoxelGridComponent : GH_Component
    {
        public CloseVoxelGridComponent()
          : base("Close Voxel Data", "CloseVoxels",
              "Caps voxel boundary values to zero, ensuring the resulting isosurface is a closed volume.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("VoxelData", "D", "The voxel data (float[x,y,z]) to close.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("VoxelData", "D", "The closed voxel data — boundary cells are set to zero.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 0, this, out float[,,] inputData, "Voxel data"))
                return;

            float[,,] result = (float[,,])inputData.Clone();
            WorkflowAGrid.ZeroVoxelBoundaryInPlace(result);

            DA.SetData(0, new GH_ObjectWrapper(result));
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.CloseVolume;

        public override Guid ComponentGuid => new Guid("8f87ef6e-8a4e-4c1e-a75a-a6cd753701af");
    }
}
