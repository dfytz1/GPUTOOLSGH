using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.MeshTopology;

namespace GHGPUPlugin.Components.DataRelationships;

public class GH_CSRFromMesh : GH_Component
{
    public GH_CSRFromMesh()
        : base(
            "CSR From Mesh GPU",
            "MeshCSRGPU",
            "Topology-vertex adjacency in CSR form (column indices flat + row offsets). Extraction runs on the CPU.",
            "GPUTools",
            "DataRelationships")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("InputMesh", "InputMesh", "Triangle or quad mesh (topology edges define adjacency).", GH_ParamAccess.item);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddIntegerParameter("AdjFlat", "AdjFlat", "Neighbor topology-vertex indices, all rows concatenated.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("RowOffsets", "RowOffsets", "CSR row pointers; length is topology vertex count + 1.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("VertexCount", "VertexCount", "Number of topology vertices.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Mesh? mesh = null;
        if (!DA.GetData("InputMesh", ref mesh) || mesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No mesh provided.");
            return;
        }

        if (!mesh.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(mesh);
        MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);
        int tvCount = neighbors.Length;

        var ghAdj = new List<GH_Integer>(adjFlat.Length);
        for (int i = 0; i < adjFlat.Length; i++)
            ghAdj.Add(new GH_Integer(adjFlat[i]));

        var ghOff = new List<GH_Integer>(rowOffsets.Length);
        for (int i = 0; i < rowOffsets.Length; i++)
            ghOff.Add(new GH_Integer(rowOffsets[i]));

        DA.SetDataList(0, ghAdj);
        DA.SetDataList(1, ghOff);
        DA.SetData("VertexCount", tvCount);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("fbb28251-5eda-4d4e-91f2-4107daeab278");
}
