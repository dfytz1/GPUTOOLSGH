/*
 * Based on ChromodorisGH / IsosurfaceComponent (GPL-3.0)
 */

using GHGPUPlugin.Chromodoris.Topology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Build IsoSurface with optional Support/Load masks: boosts voxel values on S∪L so the surface
    /// reliably cuts through painted regions (same B and Cc as the rest of Workflow A).
    /// </summary>
    public class IsosurfaceSlComponent : GH_Component
    {
        public IsosurfaceSlComponent()
          : base("Build IsoSurface SL GPU", "IsoSLGPU",
              "Same as Build IsoSurface. Optional S/L masks pin the field; optional SupportGeometry/LoadGeometry " +
              "(same as Voxel Paint) snap vertices onto those surfaces. CloseBoundary (default on) reapplies the " +
              "same outer-layer zero as Close Voxel Data after pinning so the box stays sealed.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as the component that built the grid.", GH_ParamAccess.item);
            pManager.AddGenericParameter("VoxelData", "D", "Voxel data float[x,y,z].", GH_ParamAccess.item);
            pManager.AddNumberParameter("IsoValue", "V", "Iso threshold.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("CellCentered", "Cc", "True: cell-centered (Voxel Design / Laplace). False: corner grid (Sample Voxels).", GH_ParamAccess.item, true);
            pManager.AddGenericParameter("SupportMask", "S", "Optional. Same resolution as D; pins surface to support voxels.", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Optional. Same resolution as D; pins surface to load voxels.", GH_ParamAccess.item);
            pManager.AddGeometryParameter("SupportGeometry", "Sg", "Optional. Same support objects as Voxel Paint — snap verts onto them.", GH_ParamAccess.list);
            pManager.AddGeometryParameter("LoadGeometry", "Lg", "Optional. Same load objects as Voxel Paint.", GH_ParamAccess.list);
            pManager.AddNumberParameter("SnapBand", "Sb",
                "World units: also snap verts within this distance of Sg/Lg (use ~Voxel Paint D if masks miss edge verts). 0 = mask-only.",
                GH_ParamAccess.item, 0.0);
            pManager.AddIntegerParameter("PinBand", "Pb",
                "Voxel steps: BFS halo beyond S∪L into the bulk field (thicker collar = larger bulge). 0 = pin mask voxels only.",
                GH_ParamAccess.item, 3);
            pManager.AddNumberParameter("PinDelta", "Pd",
                "Added to max(D) for the pin level; larger values push the iso surface farther from the bulk (stronger bulge).",
                GH_ParamAccess.item, 0.02);
            pManager.AddBooleanParameter("CloseBoundary", "Cls",
                "True: after S/L pin, set the outer voxel layer to 0 (same as Close Voxel Data). Prevents pinning from reopening the mesh at the box; use False to skip.",
                GH_ParamAccess.item, true);
            pManager[4].Optional = true;
            pManager[5].Optional = true;
            pManager[6].Optional = true;
            pManager[7].Optional = true;
            pManager[8].Optional = true;
            pManager[9].Optional = true;
            pManager[10].Optional = true;
            pManager[11].Optional = true;
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Reserved — marching cubes is CPU-only in this build.", GH_ParamAccess.item, true);
            pManager[12].Optional = true;
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

            if (voxelData == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelData is null.");
                return;
            }

            int nx = voxelData.GetLength(0);
            int ny = voxelData.GetLength(1);
            int nz = voxelData.GetLength(2);

            if (nx < 2 || ny < 2 || nz < 2)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VoxelData resolution is too low (minimum 2 in each dimension).");
                return;
            }

            IGH_Goo gooS = null, gooL = null;
            bool dataS = DA.GetData(4, ref gooS);
            bool dataL = DA.GetData(5, ref gooL);

            var supportGoos = new List<IGH_GeometricGoo>();
            var loadGoos = new List<IGH_GeometricGoo>();
            DA.GetDataList(6, supportGoos);
            DA.GetDataList(7, loadGoos);
            double snapBand = 0;
            DA.GetData(8, ref snapBand);
            int pinBand = 3;
            DA.GetData(9, ref pinBand);
            double pinDelta = 0.02;
            DA.GetData(10, ref pinDelta);
            if (pinBand < 0)
                pinBand = 0;
            if (pinBand > 256)
                pinBand = 256;
            if (pinDelta < 0)
                pinDelta = 0;
            bool closeBoundary = true;
            DA.GetData(11, ref closeBoundary);
            bool useGpu = true;
            DA.GetData(12, ref useGpu);
            NativeLoader.EnsureLoaded();
            if (useGpu)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    "Marching cubes is CPU-only. GPU reserved for future release.");

            float[,,] support = new float[nx, ny, nz];
            float[,,] load = new float[nx, ny, nz];

            float[,,] work = voxelData;
            if (dataS || dataL)
            {
                if (dataS)
                {
                    if (!VoxelMaskGoo.TryUnwrap(gooS, this, out support, "SupportMask"))
                        return;
                    if (support.GetLength(0) != nx || support.GetLength(1) != ny || support.GetLength(2) != nz)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SupportMask dimensions must match VoxelData.");
                        return;
                    }
                }

                if (dataL)
                {
                    if (!VoxelMaskGoo.TryUnwrap(gooL, this, out load, "LoadMask"))
                        return;
                    if (load.GetLength(0) != nx || load.GetLength(1) != ny || load.GetLength(2) != nz)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "LoadMask dimensions must match VoxelData.");
                        return;
                    }
                }

                work = (float[,,])voxelData.Clone();
                ApplySupportLoadPin(work, voxelData, support, load, nx, ny, nz, isovalue, pinBand, pinDelta);
            }

            if (closeBoundary)
            {
                if (ReferenceEquals(work, voxelData))
                    work = (float[,,])voxelData.Clone();
                WorkflowAGrid.ZeroVoxelBoundaryInPlace(work);
            }

            var vs = new VolumetricSpace(work);
            var isosurface = new HashIsoSurface(vs);
            var mesh = new Mesh();

            isosurface.ComputeSurfaceMesh(isovalue, ref mesh);
            TransformMeshToBox(mesh, box, voxelData, cellCentered);

            if (mesh.Faces.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "No surface generated. Check that IsoValue is within the range of your voxel data.");
                return;
            }

            var geoS = ToGeometryBases(supportGoos);
            var geoL = ToGeometryBases(loadGoos);
            if (geoS.Count > 0 || geoL.Count > 0)
            {
                if (snapBand <= 1e-12 && !dataS && !dataL)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                        "Snap: wire S/L masks and/or set SnapBand > 0 so geometry snapping can select vertices.");
                }
                else
                {
                    IsoSlGeometrySnap.Apply(mesh, box, nx, ny, nz, support, load, dataS, dataL, geoS, geoL, cellCentered, snapBand);
                }
            }

            DA.SetData(0, mesh);
        }

        private static List<GeometryBase> ToGeometryBases(List<IGH_GeometricGoo> goos)
        {
            var list = new List<GeometryBase>();
            if (goos == null) return list;
            foreach (IGH_GeometricGoo g in goos)
            {
                if (g == null) continue;
                object sv = g.ScriptVariable();
                if (sv is GeometryBase gb)
                    list.Add(gb);
                else if (sv is Point3d ptd)
                    list.Add(new Point(ptd));
            }
            return list;
        }

        /// <summary>
        /// Raise values on S∪L and a short BFS band inside the original field so the level set stays connected
        /// to the bulk (avoids disjoint shells / marching-cubes degeneracy from isolated flat spikes).
        /// </summary>
        private static void ApplySupportLoadPin(float[,,] data, float[,,] original, float[,,] support, float[,,] load,
            int nx, int ny, int nz, double isovalue, int pinBand, double pinDelta)
        {
            float mx = 0f;
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        mx = Math.Max(mx, original[i, j, k]);

            float delta = (float)pinDelta;
            float pinBase = mx + delta;
            if (pinBase <= mx)
                pinBase = mx + 1e-4f;

            float isoF = (float)isovalue;
            float insideEps = 1e-8f;
            if (isoF > 1e-12f)
                insideEps = Math.Max(insideEps, isoF * 1e-4f);

            var pinCell = new bool[nx, ny, nz];
            var q = new Queue<(int i, int j, int k, int dist)>();

            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (support[i, j, k] < 0.5f && load[i, j, k] < 0.5f)
                            continue;
                        pinCell[i, j, k] = true;
                        q.Enqueue((i, j, k, 0));
                    }
                }
            }

            int[] di = { 1, -1, 0, 0, 0, 0 };
            int[] dj = { 0, 0, 1, -1, 0, 0 };
            int[] dk = { 0, 0, 0, 0, 1, -1 };
            int maxBand = pinBand;

            while (q.Count > 0)
            {
                var (i, j, k, d) = q.Dequeue();
                if (d >= maxBand)
                    continue;
                for (int a = 0; a < 6; a++)
                {
                    int ii = i + di[a], jj = j + dj[a], kk = k + dk[a];
                    if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz)
                        continue;
                    if (pinCell[ii, jj, kk])
                        continue;
                    if (original[ii, jj, kk] <= insideEps)
                        continue;
                    pinCell[ii, jj, kk] = true;
                    q.Enqueue((ii, jj, kk, d + 1));
                }
            }

            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        if (!pinCell[i, j, k])
                            continue;
                        float wobble = 1e-5f * (float)Math.Sin(i * 2.17 + j * 3.91 + k * 5.23);
                        float pin = pinBase + wobble;
                        data[i, j, k] = Math.Max(data[i, j, k], pin);
                    }
                }
            }
        }

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

        protected override System.Drawing.Bitmap Icon => Icons.IsoSurfaceSl;

        public override Guid ComponentGuid => new Guid("67cecc21-91fa-40fd-ae29-04debbaaf744");
    }
}
