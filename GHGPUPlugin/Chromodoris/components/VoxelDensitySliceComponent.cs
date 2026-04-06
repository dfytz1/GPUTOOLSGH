using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Drawing;

namespace GHGPUPlugin.Chromodoris;

/// <summary>
/// Samples a voxel density grid on an arbitrary plane and builds a quad mesh with greyscale vertex colours.
/// </summary>
public class VoxelDensitySliceComponent : GH_Component
{
    public VoxelDensitySliceComponent()
        : base("Voxel Density Slice GPU", "DensitySlice",
            "Arbitrary-plane slice through a float[x,y,z] density field with trilinear sampling and threshold or smooth greyscale vertex colours.",
            "GPUTools", "Voxel")
    {
    }

    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("DensityField", "D", "float[x,y,z] from Voxel SIMP Topology GPU Density (R).", GH_ParamAccess.item);
        pManager.AddBoxParameter("BoundingBox", "B", "Same Box as Voxel SIMP Topology GPU (B).", GH_ParamAccess.item);
        pManager.AddPlaneParameter("SlicePlane", "Pl", "Plane for the slice (origin and axes define position and orientation).", GH_ParamAccess.item);
        pManager.AddNumberParameter("PlaneWidth", "W", "Slice extent along plane X.", GH_ParamAccess.item, 10.0);
        pManager.AddNumberParameter("PlaneHeight", "H", "Slice extent along plane Y.", GH_ParamAccess.item, 10.0);
        pManager.AddIntegerParameter("ResolutionU", "RU", "Vertex count along plane U.", GH_ParamAccess.item, 100);
        pManager.AddIntegerParameter("ResolutionV", "RV", "Vertex count along plane V.", GH_ParamAccess.item, 100);
        pManager.AddNumberParameter("IsoThreshold", "T", "Iso threshold for hard mode.", GH_ParamAccess.item, 0.3);
        pManager.AddBooleanParameter("BlackOnWhite", "BW", "If true, rho ≥ T is black; if false, colours invert.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("HardThreshold", "Hard", "Hard threshold vs smooth gamma greyscale.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("SliceMesh", "M", "Quad mesh with VertexColors on the slice plane.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        float[,,] density = null;
        Box box = new Box();
        Plane pl = Plane.Unset;
        double w = 10, h = 10, isoT = 0.3;
        int ru = 100, rv = 100;
        bool blackOnWhite = true, hard = true;

        if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 0, this, out density, "DensityField"))
            return;
        if (!DA.GetData(1, ref box))
            return;
        if (!DA.GetData(2, ref pl))
            return;
        DA.GetData(3, ref w);
        DA.GetData(4, ref h);
        DA.GetData(5, ref ru);
        DA.GetData(6, ref rv);
        DA.GetData(7, ref isoT);
        DA.GetData(8, ref blackOnWhite);
        DA.GetData(9, ref hard);

        if (ru < 2 || rv < 2)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "ResolutionU and ResolutionV must be at least 2.");
            return;
        }

        if (w <= 0 || h <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "PlaneWidth and PlaneHeight must be positive.");
            return;
        }

        if (!pl.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SlicePlane is not valid.");
            return;
        }

        int nx = density.GetLength(0);
        int ny = density.GetLength(1);
        int nz = density.GetLength(2);
        BoundingBox bb = box.BoundingBox;
        if (!bb.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "BoundingBox is not valid.");
            return;
        }

        Plane slicePl = pl;
        if (!slicePl.XAxis.Unitize())
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SlicePlane XAxis is degenerate.");
            return;
        }

        slicePl.ZAxis = Vector3d.CrossProduct(slicePl.XAxis, slicePl.YAxis);
        if (!slicePl.ZAxis.Unitize())
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SlicePlane X and Y are parallel.");
            return;
        }

        slicePl.YAxis = Vector3d.CrossProduct(slicePl.ZAxis, slicePl.XAxis);
        slicePl.YAxis.Unitize();
        Vector3d xu = slicePl.XAxis;
        Vector3d yv = slicePl.YAxis;

        var mesh = new Mesh();

        for (int j = 0; j < rv; j++)
        {
            double sv = rv > 1 ? j / (double)(rv - 1) : 0.5;
            double yOff = (sv - 0.5) * h;
            for (int i = 0; i < ru; i++)
            {
                double su = ru > 1 ? i / (double)(ru - 1) : 0.5;
                double xOff = (su - 0.5) * w;
                Point3d pt = slicePl.Origin + xOff * xu + yOff * yv;
                float rho = SampleDensityTrilinear(bb, density, pt, nx, ny, nz);
                mesh.Vertices.Add(pt);
                Color col = RhoToVertexColor(rho, isoT, hard, blackOnWhite);
                mesh.VertexColors.Add(col.R, col.G, col.B);
            }
        }

        for (int j = 0; j < rv - 1; j++)
        {
            for (int i = 0; i < ru - 1; i++)
            {
                int v00 = j * ru + i;
                int v10 = j * ru + i + 1;
                int v11 = (j + 1) * ru + i + 1;
                int v01 = (j + 1) * ru + i;
                mesh.Faces.AddFace(v00, v10, v11, v01);
            }
        }

        mesh.Normals.ComputeNormals();
        mesh.Faces.CullDegenerateFaces();
        DA.SetData(0, mesh);
    }

    private static float SampleDensityTrilinear(BoundingBox bb, float[,,] vol, Point3d p, int nx, int ny, int nz)
    {
        double sx = bb.Max.X - bb.Min.X;
        double sy = bb.Max.Y - bb.Min.Y;
        double sz = bb.Max.Z - bb.Min.Z;
        if (sx < 1e-30 || sy < 1e-30 || sz < 1e-30)
            return 0f;

        double ux = (p.X - bb.Min.X) / sx;
        double uy = (p.Y - bb.Min.Y) / sy;
        double uz = (p.Z - bb.Min.Z) / sz;
        if (ux < 0 || ux > 1 || uy < 0 || uy > 1 || uz < 0 || uz > 1)
            return 0f;

        double tx = ux * nx - 0.5;
        double ty = uy * ny - 0.5;
        double tz = uz * nz - 0.5;
        return (float)TriSampleFloat(vol, tx, ty, tz, nx, ny, nz);
    }

    private static double TriSampleFloat(float[,,] c, double x, double y, double z, int nxc, int nyc, int nzc)
    {
        x = Clamp(x, 0, Math.Max(0, nxc - 1));
        y = Clamp(y, 0, Math.Max(0, nyc - 1));
        z = Clamp(z, 0, Math.Max(0, nzc - 1));

        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        int z0 = (int)Math.Floor(z);
        int x1 = Math.Min(x0 + 1, nxc - 1);
        int y1 = Math.Min(y0 + 1, nyc - 1);
        int z1 = Math.Min(z0 + 1, nzc - 1);

        double txx = x - x0;
        double tyy = y - y0;
        double tzz = z - z0;

        double c000 = c[x0, y0, z0];
        double c100 = c[x1, y0, z0];
        double c010 = c[x0, y1, z0];
        double c110 = c[x1, y1, z0];
        double c001 = c[x0, y0, z1];
        double c101 = c[x1, y0, z1];
        double c011 = c[x0, y1, z1];
        double c111 = c[x1, y1, z1];

        double c00 = c000 * (1 - txx) + c100 * txx;
        double c10 = c010 * (1 - txx) + c110 * txx;
        double c01 = c001 * (1 - txx) + c101 * txx;
        double c11 = c011 * (1 - txx) + c111 * txx;

        double c0 = c00 * (1 - tyy) + c10 * tyy;
        double c1 = c01 * (1 - tyy) + c11 * tyy;

        return c0 * (1 - tzz) + c1 * tzz;
    }

    private static double Clamp(double v, double a, double b)
    {
        if (v < a) return a;
        if (v > b) return b;
        return v;
    }

    private static Color RhoToVertexColor(double rho, double isoT, bool hard, bool blackOnWhite)
    {
        byte g;
        if (hard)
        {
            g = rho >= isoT ? (byte)0 : (byte)255;
            if (!blackOnWhite)
                g = (byte)(255 - g);
        }
        else
        {
            double t = Clamp(rho, 0, 1);
            t = Math.Pow(t, 2.2);
            g = (byte)(255 * (1.0 - t));
            if (!blackOnWhite)
                g = (byte)(255 - g);
        }

        return Color.FromArgb(g, g, g);
    }

    public override GH_Exposure Exposure => GH_Exposure.quinary;

    protected override System.Drawing.Bitmap Icon => null!;

    public override Guid ComponentGuid => new Guid("6c4a9f12-8d0e-4b7a-9c31-2e5f8041b6a2");
}
