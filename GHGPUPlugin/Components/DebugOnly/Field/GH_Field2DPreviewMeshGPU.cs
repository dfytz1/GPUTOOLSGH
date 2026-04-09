using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

/// <summary>Turns a 2D scalar field <c>float[nx,ny]</c> into a quad mesh on a plane with vertex colors.</summary>
public class GH_Field2DPreviewMeshGPU : GH_Component
{
    public GH_Field2DPreviewMeshGPU()
        : base(
            "2D Field Preview Mesh GPU",
            "Field2DMesh",
            "Builds a quad mesh on a plane from float[nx,ny], colored by value (for reaction-diffusion B or any scalar field).",
            "GPUTools",
            "Field")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Field", "F", "float[nx,ny] values.", GH_ParamAccess.item);
        pManager.AddPlaneParameter("Plane", "Pl", "Domain corner and axes; mesh spans SizeX×SizeY from Origin.", GH_ParamAccess.item, Plane.WorldXY);
        pManager.AddNumberParameter("SizeX", "Sx", "Extent along plane X.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("SizeY", "Sy", "Extent along plane Y.", GH_ParamAccess.item, 1.0);
        pManager.AddBooleanParameter("Normalize", "Norm", "Map field min–max to colors before drawing.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Colored quad mesh.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        IGH_Goo? goo = null;
        if (!DA.GetData(0, ref goo) || goo == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Field is required.");
            return;
        }

        float[,]? field = null;
        if (goo is GH_ObjectWrapper ow && ow.Value is float[,] a)
            field = a;
        else if (goo.ScriptVariable() is float[,] b)
            field = b;

        if (field == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Field must be float[nx,ny].");
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
        double sx = 1, sy = 1;
        bool norm = true;
        if (!DA.GetData(1, ref pl)) return;
        DA.GetData(2, ref sx);
        DA.GetData(3, ref sy);
        DA.GetData(4, ref norm);

        if (sx <= 0 || sy <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SizeX and SizeY must be positive.");
            return;
        }

        DA.SetData(0, Field2DPreviewMesh.BuildMesh(field, pl, sx, sy, norm));
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("f8b2d5c3-405e-5f7b-0a1d-2e3f4a5b6c7d");
}
