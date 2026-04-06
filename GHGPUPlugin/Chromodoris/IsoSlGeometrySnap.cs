using GHGPUPlugin.Chromodoris.MeshTools;
using GHGPUPlugin.Chromodoris.Topology;
using Rhino.Geometry;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Projects iso-mesh vertices onto support/load reference geometry (same objects as Voxel Paint).
    /// </summary>
    internal static class IsoSlGeometrySnap
    {
        public static void Apply(Mesh mesh, Box box, int nx, int ny, int nz,
            float[,,] supportMask, float[,,] loadMask,
            bool haveMaskS, bool haveMaskL,
            List<GeometryBase> geoS, List<GeometryBase> geoL,
            bool cellCentered, double snapBandWorld)
        {
            if (mesh == null || mesh.Vertices.Count == 0) return;
            if (geoS.Count == 0 && geoL.Count == 0) return;

            BoundingBox bb = box.BoundingBox;
            if (!bb.IsValid) return;

            bool useBand = snapBandWorld > 1e-12;

            for (int vi = 0; vi < mesh.Vertices.Count; vi++)
            {
                Point3d p = mesh.Vertices[vi];

                VoxelWorldMapping.WorldToUnit(p, bb, out double tx, out double ty, out double tz);
                int ix, iy, iz;
                if (cellCentered)
                    VoxelWorldMapping.UnitToVoxelCellCentered(tx, ty, tz, nx, ny, nz, out ix, out iy, out iz);
                else
                    VoxelWorldMapping.UnitToVoxelCorner(tx, ty, tz, nx, ny, nz, out ix, out iy, out iz);

                bool inS = haveMaskS && supportMask[ix, iy, iz] >= 0.5f;
                bool inL = haveMaskL && loadMask[ix, iy, iz] >= 0.5f;

                double dS = double.MaxValue, dL = double.MaxValue;
                Point3d qS = p, qL = p;
                bool okS = geoS.Count > 0 && WorkflowAGrid.TryGetClosestPoint(p, geoS, out qS, out dS);
                bool okL = geoL.Count > 0 && WorkflowAGrid.TryGetClosestPoint(p, geoL, out qL, out dL);

                bool candS = okS && ((haveMaskS && inS) || (useBand && dS <= snapBandWorld));
                bool candL = okL && ((haveMaskL && inL) || (useBand && dL <= snapBandWorld));

                if (!candS && !candL)
                    continue;

                if (candS && !candL)
                    mesh.Vertices.SetVertex(vi, qS);
                else if (candL && !candS)
                    mesh.Vertices.SetVertex(vi, qL);
                else
                    mesh.Vertices.SetVertex(vi, dS <= dL ? qS : qL);
            }

            mesh.Normals.ComputeNormals();
            mesh.Faces.CullDegenerateFaces();
        }
    }
}
