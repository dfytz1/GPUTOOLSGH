using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DebugOnly;

/// <summary>Anisotropic centroidal Voronoi–style remeshing via GPU particle repulsion on a mesh, then 2D Delaunay reconstruction.</summary>
public class GH_AnisotropicCvtRemeshGPU : GH_Component
{
    public GH_AnisotropicCvtRemeshGPU()
        : base(
            "Anisotropic CVT Remesh GPU",
            "AnisoCVTGPU",
            "Distributes particles on a mesh with GPU anisotropic repulsion and Laplacian smoothing, projects to the surface, then builds a new mesh with planar Delaunay + optional circumradius filtering (default factor is loose enough to avoid punching holes).",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("InputMesh", "M", "Source triangle mesh (quads are triangulated once).", GH_ParamAccess.item);
        pManager.AddIntegerParameter("ParticleCount", "N", "Number of particles / remesh vertices.", GH_ParamAccess.item, 1000);
        pManager.AddIntegerParameter("Iterations", "I", "Relaxation iterations.", GH_ParamAccess.item, 80);
        pManager.AddNumberParameter("AnisotropyStrength", "A", "α in f(κ)=1+α|κ| for the metric tensor (0–2).", GH_ParamAccess.item, 0.5);
        pManager.AddNumberParameter("RepulsionStrength", "R", "Repulsion scale (0–1).", GH_ParamAccess.item, 0.5);
        pManager.AddBooleanParameter("BoundaryFixed", "B", "Fix boundary particles on naked edges.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal for metric, particle step, projection, and boundary snap when available.", GH_ParamAccess.item, true);
        pManager.AddNumberParameter(
            "CircumradiusFactor",
            "Cr",
            "Max circumradius / target spacing for keeping a Delaunay triangle. Small values (≈1.5) remove many boundary triangles and cause holes. Default 3.5; try 4–6 for curved surfaces. ≤0 disables filtering (full Delaunay).",
            GH_ParamAccess.item,
            3.5);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("RemeshedMesh", "Mesh", "Reconstructed triangle mesh.", GH_ParamAccess.item);
        pManager.AddParameter(new Grasshopper.Kernel.Parameters.Param_Point(), "ParticlePositions", "P", "Final particle positions.", GH_ParamAccess.list);
        pManager.AddNumberParameter("SolveTimeMs", "Ms", "Wall time for the solve phase (ms).", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Mesh? meshIn = null;
        if (!DA.GetData(0, ref meshIn) || meshIn == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Input mesh is required.");
            return;
        }

        int particleCount = 1000;
        DA.GetData(1, ref particleCount);

        int iterations = 80;
        DA.GetData(2, ref iterations);

        double aniso = 0.5;
        DA.GetData(3, ref aniso);

        double repulsion = 0.5;
        DA.GetData(4, ref repulsion);

        bool boundaryFixed = true;
        DA.GetData(5, ref boundaryFixed);

        bool useGpu = true;
        DA.GetData(6, ref useGpu);

        double circumradiusFactor = 3.5;
        DA.GetData(7, ref circumradiusFactor);

        if (!MeshAnisoCvtRemesh.TrySolve(
                this,
                meshIn,
                particleCount,
                iterations,
                aniso,
                repulsion,
                boundaryFixed,
                useGpu,
                circumradiusFactor,
                out Mesh? outMesh,
                out List<Point3d>? particles,
                out double ms,
                out string? err) || outMesh == null || particles == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, err ?? "Anisotropic CVT remesh failed.");
            return;
        }

        DA.SetData(0, outMesh);
        DA.SetDataList(1, particles.ConvertAll(p => new GH_Point(p)));
        DA.SetData(2, new GH_Number(ms));
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("a7c4e2b1-9d0f-4a8e-b6c5-3d2e1f0a9b8c");
}
