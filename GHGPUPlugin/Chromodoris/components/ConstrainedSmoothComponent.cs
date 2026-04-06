using GHGPUPlugin.Chromodoris.MeshTools;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Laplacian smoothing with vertices under Support / Load voxels held fixed (same B, Cc, masks as Build IsoSurface).
    /// </summary>
    public class ConstrainedSmoothComponent : GH_Component
    {
        public ConstrainedSmoothComponent()
          : base("Smooth Masked", "SmoothMask",
              "Laplacian smooth a mesh while locking vertices that fall in Support and/or Load voxels. " +
              "Wire the same BoundingBox, masks, and CellCentered flag as Build IsoSurface.",
              "GPUTools", "Mesh")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Mesh to smooth (e.g. iso from Build IsoSurface).", GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Build IsoSurface / Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupportMask", "S", "Support voxel mask float[x,y,z] (same resolution as density).", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Load voxel mask float[x,y,z].", GH_ParamAccess.item);
            pManager.AddBooleanParameter("CellCentered", "Cc", "Must match Build IsoSurface (true for Voxel Design / SIMP workflow).", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("FixSupport", "Fs", "Lock vertices in support voxels.", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("FixLoad", "Fl", "Lock vertices in load voxels.", GH_ParamAccess.item, true);
            pManager.AddIntegerParameter("Dilate", "D",
                "Expand constraint region by this many voxel rings (0–3) so boundary vertices stay put.", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("StepSize", "St", "Laplacian step 0…1.", GH_ParamAccess.item, 0.5);
            pManager.AddIntegerParameter("Iterations", "I", "Smoothing iterations.", GH_ParamAccess.item, 1);
            pManager[7].Optional = true;
            pManager[8].Optional = true;
            pManager[9].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Smoothed mesh.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Mesh mesh = null;
            Box box = new Box();
            float[,,] support = null, load = null;
            bool cellCentered = true, fixSupport = true, fixLoad = true;
            int dilate = 1, iterations = 1;
            double step = 0.5;

            if (!DA.GetData(0, ref mesh)) return;
            if (!DA.GetData(1, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask")) return;
            DA.GetData(4, ref cellCentered);
            DA.GetData(5, ref fixSupport);
            DA.GetData(6, ref fixLoad);
            DA.GetData(7, ref dilate);
            DA.GetData(8, ref step);
            DA.GetData(9, ref iterations);

            int nx = support.GetLength(0), ny = support.GetLength(1), nz = support.GetLength(2);
            if (load.GetLength(0) != nx || load.GetLength(1) != ny || load.GetLength(2) != nz)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Support and Load masks must have the same dimensions.");
                return;
            }

            if (iterations < 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be >= 0.");
                return;
            }

            if (step < 0 || step > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "StepSize must be between 0 and 1.");
                return;
            }

            if (dilate < 0 || dilate > 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Dilate must be between 0 and 3.");
                return;
            }

            if (iterations == 0 || step == 0 || (!fixSupport && !fixLoad))
            {
                DA.SetData(0, mesh.DuplicateMesh());
                return;
            }

            if (!mesh.IsValid)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is invalid.");
                return;
            }

            bool[] flags;
            try
            {
                flags = ConstrainedVertexSmooth.BuildConstraints(mesh, box, nx, ny, nz, support, load,
                    cellCentered, fixSupport, fixLoad, dilate);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
                return;
            }

            int nLock = 0;
            for (int i = 0; i < flags.Length; i++)
                if (flags[i]) nLock++;

            var smooth = new ConstrainedVertexSmooth(mesh, step, iterations, flags);
            Mesh outMesh = smooth.Compute();

            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Locked {nLock} / {flags.Length} topology vertices (support/load voxels × dilate).");

            DA.SetData(0, outMesh);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.SmoothMasked;

        public override Guid ComponentGuid => new Guid("11cecc60-b658-46e5-9f7b-f5c8c44d20e0");
    }
}
