using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.Utilities;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Smoothing;

public class GH_MeshCurvature : GH_Component
{
    public GH_MeshCurvature()
        : base(
            "Mesh Curvature GPU",
            "MeshCurvGPU",
            "Discrete mean and Gaussian curvature per mesh vertex (cotangent Laplacian, parallel CPU).",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Triangle or quad mesh.", GH_ParamAccess.item);
        pManager.AddBooleanParameter("UseParallel", "P", "Parallel accumulation over vertices.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddNumberParameter("MeanCurvature", "H", "Mean curvature magnitude per mesh vertex.", GH_ParamAccess.list);
        pManager.AddNumberParameter("GaussianCurvature", "K", "Gaussian curvature per mesh vertex.", GH_ParamAccess.list);
        pManager.AddNumberParameter("PrincipalMin", "Kmin", "Minimum principal curvature estimate.", GH_ParamAccess.list);
        pManager.AddNumberParameter("PrincipalMax", "Kmax", "Maximum principal curvature estimate.", GH_ParamAccess.list);
        pManager.AddMeshParameter("CurvatureMesh", "CM", "Mesh coloured by mean curvature magnitude.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Mesh? mesh = null;
        if (!DA.GetData("Mesh", ref mesh) || mesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is required.");
            return;
        }

        if (!mesh.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        bool usePar = true;
        DA.GetData("UseParallel", ref usePar);

        MeshCurvatureDiscrete.Compute(mesh, usePar, out double[] meanTopo, out double[] gaussTopo, out double[] kMinTopo, out double[] kMaxTopo);

        var tv = mesh.TopologyVertices;
        int vc = mesh.Vertices.Count;
        var meanM = new double[vc];
        var gaussM = new double[vc];
        var kminM = new double[vc];
        var kmaxM = new double[vc];
        for (int mv = 0; mv < vc; mv++)
        {
            int ti = tv.TopologyVertexIndex(mv);
            meanM[mv] = meanTopo[ti];
            gaussM[mv] = gaussTopo[ti];
            kminM[mv] = kMinTopo[ti];
            kmaxM[mv] = kMaxTopo[ti];
        }

        var ghMean = new List<GH_Number>(vc);
        var ghGauss = new List<GH_Number>(vc);
        var ghKmin = new List<GH_Number>(vc);
        var ghKmax = new List<GH_Number>(vc);
        for (int i = 0; i < vc; i++)
        {
            ghMean.Add(new GH_Number(meanM[i]));
            ghGauss.Add(new GH_Number(gaussM[i]));
            ghKmin.Add(new GH_Number(kminM[i]));
            ghKmax.Add(new GH_Number(kmaxM[i]));
        }

        DA.SetDataList(0, ghMean);
        DA.SetDataList(1, ghGauss);
        DA.SetDataList(2, ghKmin);
        DA.SetDataList(3, ghKmax);

        Mesh cm = MeshColourHelper.ColourByScalar(mesh, meanM, normaliseMinMax: true);
        DA.SetData(4, cm);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("d4e5f6a7-b8c9-0123-defa-234567890123");
}
