/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;

namespace GHGPUPlugin.Chromodoris
{
    public class IsosurfaceComponent : GH_Component
    {
        public IsosurfaceComponent()
          : base("Build IsoSurface GPU", "IsoSurfaceGPU",
              "Constructs a 3D isosurface mesh from voxel data and the same bounding box the grid was built on.",
              "GPUTools", "Mesh")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as the component that built the grid (e.g. Voxel Design Domain B, or Sample Voxels B).", GH_ParamAccess.item);
            pManager.AddGenericParameter("VoxelData", "D", "Voxel data (float[x,y,z]) from Sample Voxels.", GH_ParamAccess.item);
            pManager.AddNumberParameter("IsoValue", "V", "The threshold value at which to extract the surface.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("CellCentered", "Cc", "True: values live at cell centers (Voxel Design Domain / Laplace). False: corner grid (Sample Voxels).", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Reserved — marching cubes is CPU-only in this build.", GH_ParamAccess.item, true);
            pManager[4].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("IsoSurface", "M", "The generated isosurface mesh.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Box box = new Box();
            double isovalue = 0;
            float[,,] voxelData = null;
            bool cellCentered = true;

            if (!DA.GetData(0, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out voxelData, "Voxel data")) return;
            if (!DA.GetData(2, ref isovalue)) return;
            DA.GetData(3, ref cellCentered);
            bool useGpu = true;
            DA.GetData(4, ref useGpu);
            NativeLoader.EnsureLoaded();
            if (useGpu)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    "Marching cubes is CPU-only. GPU reserved for future release.");

            if (voxelData == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelData is null.");
                return;
            }

            if (voxelData.GetLength(0) < 2 || voxelData.GetLength(1) < 2 || voxelData.GetLength(2) < 2)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelData resolution is too low (minimum 2 in each dimension).");
                return;
            }

            VolumetricSpace vs = new VolumetricSpace(voxelData);
            HashIsoSurface isosurface = new HashIsoSurface(vs);
            Mesh mesh = new Mesh();

            isosurface.ComputeSurfaceMesh(isovalue, ref mesh);
            TransformMeshToBox(mesh, box, voxelData, cellCentered);

            if (mesh.Faces.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "No surface generated. Check that IsoValue is within the range of your voxel data.");
                return;
            }

            DA.SetData(0, mesh);
        }

        /// <summary>
        /// Maps marching-cubes coords (grid indices; v ≈ 0 … n−1) to world XYZ. Uses the box’s world
        /// axis-aligned BoundingBox so Grasshopper round-trips on Box.Plane don’t shift the mesh.
        /// </summary>
        private static void TransformMeshToBox(Mesh mesh, Box targetBox, float[,,] data, bool cellCentered)
        {
            BoundingBox bb = targetBox.BoundingBox;
            if (!bb.IsValid)
                return;

            int nx = data.GetLength(0);
            int ny = data.GetLength(1);
            int nz = data.GetLength(2);
            double denomX = Math.Max(1, nx - 1);
            double denomY = Math.Max(1, ny - 1);
            double denomZ = Math.Max(1, nz - 1);
            double sx = bb.Max.X - bb.Min.X;
            double sy = bb.Max.Y - bb.Min.Y;
            double sz = bb.Max.Z - bb.Min.Z;

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                Point3d v = mesh.Vertices[i];
                double tx, ty, tz;
                if (cellCentered)
                {
                    tx = (v.X + 0.5) / nx;
                    ty = (v.Y + 0.5) / ny;
                    tz = (v.Z + 0.5) / nz;
                }
                else
                {
                    tx = v.X / denomX;
                    ty = v.Y / denomY;
                    tz = v.Z / denomZ;
                }

                tx = Rhino.RhinoMath.Clamp(tx, 0.0, 1.0);
                ty = Rhino.RhinoMath.Clamp(ty, 0.0, 1.0);
                tz = Rhino.RhinoMath.Clamp(tz, 0.0, 1.0);

                Point3d w = new Point3d(
                    bb.Min.X + tx * sx,
                    bb.Min.Y + ty * sy,
                    bb.Min.Z + tz * sz);
                mesh.Vertices.SetVertex(i, w);
            }

            mesh.Normals.ComputeNormals();
            mesh.Faces.CullDegenerateFaces();
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.IsoSurface;

        public override Guid ComponentGuid => new Guid("214f9509-8626-431a-a047-91920973e6f9");
    }
}
