using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

/// <summary>Displaces mesh vertices along normals (or plane normal) by sampling a float[nx,ny] field on a plane.</summary>
public class GH_MeshDisplaceField2DGPU : GH_Component
{
    public GH_MeshDisplaceField2DGPU()
        : base(
            "Mesh Displace Field 2D GPU",
            "MeshDispF2D",
            "Moves each vertex by Amplitude × sampled field (0–1 if Normalize), along vertex normal or plane Z. Wire B and DPl, Sx, Sy from Reaction Diffusion Mesh Color GPU outputs.",
            "GPUTools",
            "Field")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Source mesh (duplicated).", GH_ParamAccess.item);
        pManager.AddGenericParameter("Field", "F", "float[nx,ny] scalar (e.g. B from reaction diffusion).", GH_ParamAccess.item);
        pManager.AddPlaneParameter("Plane", "Pl", "Same plane as the field.", GH_ParamAccess.item, Plane.WorldXY);
        pManager.AddNumberParameter("SizeX", "Sx", "Domain extent along plane X.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("SizeY", "Sy", "Domain extent along plane Y.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("Amplitude", "Amp", "Displacement scale (world units).", GH_ParamAccess.item, 0.05);
        pManager.AddBooleanParameter("Normalize", "Norm", "Map field min–max to 0–1 before × Amplitude.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseMeshNormals", "MeshN", "True: along vertex normal; false: along plane Z.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Displaced mesh.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Mesh? meshIn = null;
        if (!DA.GetData(0, ref meshIn) || meshIn == null || !meshIn.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Valid mesh required.");
            return;
        }

        IGH_Goo? goo = null;
        if (!DA.GetData(1, ref goo) || goo == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Field is required.");
            return;
        }

        if (!Field2DDataHelper.TryUnwrapFloat2D(goo, out float[,]? field, out string msg) || field == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, msg);
            return;
        }

        int nx = field.GetLength(0);
        int ny = field.GetLength(1);
        if (nx < 2 || ny < 2)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Field must be at least 2×2.");
            return;
        }

        var pl = Plane.WorldXY;
        double sx = 1, sy = 1, amp = 0.05;
        bool norm = true, meshN = true;
        if (!DA.GetData(2, ref pl)) return;
        DA.GetData(3, ref sx);
        DA.GetData(4, ref sy);
        DA.GetData(5, ref amp);
        DA.GetData(6, ref norm);
        DA.GetData(7, ref meshN);

        if (sx <= 0 || sy <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SizeX and SizeY must be positive.");
            return;
        }

        Mesh outM = Field2DMeshDisplace.Build(meshIn, field, nx, ny, pl, sx, sy, amp, norm, meshN);
        DA.SetData(0, outM);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("b4d9f012-3c5e-5e9f-0d2b-6f708e9c0b1d");
}
