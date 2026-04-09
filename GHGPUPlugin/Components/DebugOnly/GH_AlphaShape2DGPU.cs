using System.Drawing;
using System.Threading.Tasks;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DebugOnly;

/// <summary>2D alpha shape: Delaunay via GPU JFA edges + Triangle.NET when enabled, else Bowyer–Watson; circumradius filter on Metal (CPU fallback). Debug build only.</summary>
public class GH_AlphaShape2DGPU : GH_Component
{
    public GH_AlphaShape2DGPU()
        : base(
            "Alpha Shape 2D GPU",
            "AlphaShGPU",
            "Projects points to a plane. Optional JFA+Triangle.NET: GPU Jump-Flooding Delaunay edges seed constrained triangulation (Unofficial.Triangle.NET); on failure or if disabled, Bowyer–Watson (same as Anisotropic CVT remesh). Then keeps triangles whose circumcircle radius is at most Alpha radius. Non-positive Alpha keeps all Delaunay triangles. Circumradius filter uses Metal when available.",
            "GPUTools",
            "Graph")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pm)
    {
        pm.AddPointParameter("Points", "P", "Point cloud (projected to the plane).", GH_ParamAccess.list);
        pm.AddPlaneParameter("Plane", "Pl", "Plane for projection and mesh orientation.", GH_ParamAccess.item, Plane.WorldXY);
        pm.AddNumberParameter("AlphaRadius", "A", "Maximum circumcircle radius to keep a triangle (model units). Zero or negative keeps all Delaunay triangles.", GH_ParamAccess.item, 10.0);
        pm.AddBooleanParameter("JFASeed", "Hybrid", "True: JFA plus Triangle.NET Delaunay when Metal is on. False: Bowyer-Watson only.", GH_ParamAccess.item, true);
        pm.AddIntegerParameter("JFAGrid", "Res", "JFA grid resolution (next power of two). Only when Hybrid is true and Metal is on.", GH_ParamAccess.item, 512);
        pm.AddBooleanParameter("UseGPU", "Metal", "Use Apple Metal for JFA (if Hybrid) and for alpha circumradius filtering. CPU fallbacks if off or unavailable.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pm)
    {
        pm.AddMeshParameter("Mesh", "M", "Alpha shape mesh (triangles only).", GH_ParamAccess.item);
        pm.AddTextParameter("Info", "I", "Counts and path used.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        var points = new List<Point3d>();
        if (!DA.GetDataList(0, points) || points.Count < 3)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Need at least three points.");
            return;
        }

        var plane = Plane.WorldXY;
        DA.GetData(1, ref plane);

        double alphaR = 10.0;
        DA.GetData(2, ref alphaR);

        bool jfaSeed = true;
        DA.GetData(3, ref jfaSeed);

        int jfaGrid = 512;
        DA.GetData(4, ref jfaGrid);

        bool useGpu = true;
        DA.GetData(5, ref useGpu);

        var uv = new Vector2d[points.Count];
        var px = new float[points.Count];
        var py = new float[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            plane.ClosestParameter(points[i], out double u, out double v);
            uv[i] = new Vector2d(u, v);
            px[i] = (float)u;
            py[i] = (float)v;
        }

        string delaunayPath;
        List<int> tri;
        IntPtr ctxJfa = IntPtr.Zero;
        bool metalForJfa = useGpu && MetalGuard.EnsureReady(this) && MetalSharedContext.TryGetContext(out ctxJfa);

        if (jfaSeed && metalForJfa)
        {
            if (JfaSeededTriangleNetDelaunay2D.TryTriangulate(uv, ctxJfa, jfaGrid, out List<int>? triJfa, out string jfaDetail))
            {
                tri = triJfa!;
                delaunayPath = $"JFA+Triangle.NET ({jfaDetail})";
            }
            else
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    $"JFA+Triangle.NET failed ({jfaDetail}); using Bowyer–Watson.");
                tri = AnisoCvtDelaunay2D.BowyerWatson(uv);
                delaunayPath = "Bowyer–Watson";
            }
        }
        else
        {
            if (jfaSeed && useGpu && !metalForJfa)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "Metal unavailable for JFA seed; using Bowyer–Watson.");
            }

            tri = AnisoCvtDelaunay2D.BowyerWatson(uv);
            delaunayPath = "Bowyer–Watson";
        }

        int nTri = tri.Count / 3;
        if (nTri < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Delaunay triangulation produced no triangles (degenerate projection?).");
            return;
        }

        var keep = new byte[nTri];
        bool unfiltered = alphaR <= 0.0 || double.IsInfinity(alphaR) || double.IsNaN(alphaR);
        string filterPath = string.Empty;

        if (unfiltered)
        {
            Array.Fill(keep, (byte)1);
            filterPath = "full Delaunay (alpha not positive)";
        }
        else
        {
            float alphaF = (float)alphaR;
            bool ranGpu = false;
            filterPath = "CPU triangle filter";
            if (useGpu && MetalGuard.EnsureReady(this) && MetalSharedContext.TryGetContext(out IntPtr ctx))
            {
                int[] triIdx = tri.ToArray();
                int code = MetalBridge.AlphaShape2DTriFilter(ctx, px, py, points.Count, triIdx, nTri, alphaF, keep);
                if (code == 0)
                {
                    ranGpu = true;
                    filterPath = "Metal triangle filter";
                }
                else
                {
                    AddRuntimeMessage(
                        GH_RuntimeMessageLevel.Warning,
                        $"Metal alpha filter failed (code {code}); using CPU.");
                }
            }
            else if (useGpu)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "GPU unavailable; using CPU triangle filter.");
            }

            if (!ranGpu)
            {
                Parallel.For(0, nTri, t =>
                {
                    int i0 = tri[t * 3];
                    int i1 = tri[t * 3 + 1];
                    int i2 = tri[t * 3 + 2];
                    double R = AnisoCvtDelaunay2D.Circumradius(
                        new Point3d(uv[i0].X, uv[i0].Y, 0),
                        new Point3d(uv[i1].X, uv[i1].Y, 0),
                        new Point3d(uv[i2].X, uv[i2].Y, 0));
                    keep[t] = (byte)(R <= alphaR ? 1 : 0);
                });
            }
        }

        var mesh = new Mesh();
        for (int i = 0; i < points.Count; i++)
            mesh.Vertices.Add(points[i]);

        int kept = 0;
        for (int t = 0; t < nTri; t++)
        {
            if (keep[t] == 0)
                continue;
            mesh.Faces.AddFace(tri[t * 3], tri[t * 3 + 1], tri[t * 3 + 2]);
            kept++;
        }

        mesh.Normals.ComputeNormals();
        mesh.Compact();

        if (kept < 1)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No triangles passed the alpha test; try a larger Alpha radius.");

        DA.SetData(0, mesh);
        DA.SetData(1, $"{points.Count} pts, {nTri} Delaunay tris | {delaunayPath} → {kept} kept | {filterPath}");
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("a7e3f1c2-9b84-4d6e-8f0a-1c2d3e4f5a6c");
}
