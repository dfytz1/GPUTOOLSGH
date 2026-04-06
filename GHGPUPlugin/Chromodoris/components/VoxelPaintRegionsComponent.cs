using GHGPUPlugin.Chromodoris.Topology;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Workflow A — step 2: mark support and load voxels by proximity to geometry.
    /// </summary>
    public class VoxelPaintRegionsComponent : GH_Component
    {
        public VoxelPaintRegionsComponent()
          : base("Voxel Paint Regions", "VoxPaint",
              "Marks voxels near support geometry (φ=0) and load geometry (φ=1). Grid must match InsideMask dimensions.",
              "GPUTools", "Voxel")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("InsideMask", "I", "Voxel grid from Voxel Design Domain output I only (float[x,y,z]). Do not connect your design mesh or support/load meshes here.", GH_ParamAccess.item);
            pManager.AddGeometryParameter("SupportGeometry", "S", "Points, curves, meshes, or breps defining the support region.", GH_ParamAccess.list);
            pManager.AddGeometryParameter("LoadGeometry", "L", "Geometry defining where the scalar potential is fixed to 1 (load region).", GH_ParamAccess.list);
            pManager.AddNumberParameter("ProximityDistance", "D", "World units: voxels whose center is within this distance are painted.", GH_ParamAccess.item);
            pManager[2].Optional = true;
            pManager[3].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("SupportMask", "S", "float[x,y,z] — 1 where support BC applies.", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "float[x,y,z] — 1 where load BC applies.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Box box = new Box();
            float[,,] inside = null;
            var supportGoos = new List<IGH_GeometricGoo>();
            var loadGoos = new List<IGH_GeometricGoo>();
            double proximity = 0;

            if (!DA.GetData(0, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask (wire Voxel Design Domain output I)")) return;
            DA.GetDataList(2, supportGoos);
            DA.GetDataList(3, loadGoos);
            if (!DA.GetData(4, ref proximity)) return;

            if (inside == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "InsideMask is null.");
                return;
            }

            if (proximity <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "ProximityDistance must be greater than 0.");
                return;
            }

            int nx = inside.GetLength(0);
            int ny = inside.GetLength(1);
            int nz = inside.GetLength(2);

            var supportGeos = ToGeometryBases(supportGoos);
            var loadGeos = ToGeometryBases(loadGoos);

            if (supportGeos.Count == 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No support geometry: SupportMask will be empty.");
            if (loadGeos.Count == 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No load geometry: LoadMask will be empty.");

            float[,,] supportMask = WorkflowAGrid.PaintProximityMask(inside, box, nx, ny, nz, supportGeos, proximity);
            float[,,] loadMask = WorkflowAGrid.PaintProximityMask(inside, box, nx, ny, nz, loadGeos, proximity);

            bool overlap = false;
            for (int i = 0; i < nx && !overlap; i++)
                for (int j = 0; j < ny && !overlap; j++)
                    for (int k = 0; k < nz && !overlap; k++)
                        if (supportMask[i, j, k] > 0.5f && loadMask[i, j, k] > 0.5f)
                            overlap = true;
            if (overlap)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Support and load regions overlap in at least one voxel. Laplace solve may behave poorly.");

            DA.SetData(0, new GH_ObjectWrapper(supportMask));
            DA.SetData(1, new GH_ObjectWrapper(loadMask));
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

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.PaintRegions;

        public override Guid ComponentGuid => new Guid("0e3d2a21-ca0b-45b9-8c02-372e4ee49510");
    }
}
