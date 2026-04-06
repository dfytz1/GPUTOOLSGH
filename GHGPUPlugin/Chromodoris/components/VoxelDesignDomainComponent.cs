using GHGPUPlugin.Chromodoris.Topology;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Workflow A — step 1: voxelize a closed design mesh into a domain mask (float[,,]).
    /// </summary>
    public class VoxelDesignDomainComponent : GH_Component
    {
        public VoxelDesignDomainComponent()
          : base("Voxel Design Domain GPU", "VoxDomainGPU",
              "Turns a closed solid mesh into a regular voxel mask (Inside value in solid, 0 outside). Use with Laplace Field Density.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("DesignMesh", "M", "Closed solid mesh defining the material design volume.", GH_ParamAccess.item);
            pManager.AddNumberParameter("VoxelSize", "S", "Voxel edge length in model units.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Inside", "In", "Value inside the mesh (0 outside). Match Voxel Boolean Tb for ∩/−; Union uses max.", GH_ParamAccess.item, 1.0);
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Axis-aligned box matching the voxel grid. Wire to Build IsoSurface and other Workflow A components.", GH_ParamAccess.item);
            pManager.AddGenericParameter("InsideMask", "I", "float[x,y,z] — Inside value in mesh, 0 outside. Wire only to grid inputs (not mesh slots).", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Mesh mesh = null;
            double voxelSize = 0;
            if (!DA.GetData(0, ref mesh)) return;
            if (!DA.GetData(1, ref voxelSize)) return;
            double insideVal = 1.0;
            DA.GetData(2, ref insideVal);
            float insideF = (float)insideVal;

            if (mesh == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No mesh provided.");
                return;
            }

            if (voxelSize <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelSize must be greater than 0.");
                return;
            }

            if (insideF <= 0f)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Inside must be greater than 0.");
                return;
            }

            if (!mesh.IsClosed)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Design mesh should be closed for reliable inside tests.");
            }

            try
            {
                WorkflowAGrid.BuildGridFromMesh(mesh, voxelSize, out Box box, out int nx, out int ny, out int nz,
                    out _, out _, out _);
                float[,,] inside = WorkflowAGrid.VoxelizeMeshInside(mesh, box, nx, ny, nz, insideF);

                int insideCount = 0;
                float halfIn = insideF * 0.5f;
                for (int i = 0; i < nx; i++)
                    for (int j = 0; j < ny; j++)
                        for (int k = 0; k < nz; k++)
                            if (inside[i, j, k] > halfIn) insideCount++;

                if (insideCount == 0)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        "No voxels marked inside the mesh. Check that the mesh is closed and intersects the grid.");
                }

                DA.SetData(0, box);
                DA.SetData(1, new GH_ObjectWrapper(inside));
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
            }
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.VoxelDesignDomain;

        public override Guid ComponentGuid => new Guid("7cf85cc9-b294-4a10-9dec-8840c3f7b2b8");
    }
}
