using Rhino.Geometry;

namespace GHGPUPlugin.Utilities;

/// <summary>Shared mesh preparation for GPU / brute-force triangle kernels.</summary>
public static class MeshTriangleUtils
{
    /// <summary>Use input mesh directly when already all triangles; otherwise duplicate once and quad-split.</summary>
    public static bool TryGetTriangleMeshForClosest(Mesh input, out Mesh triangleMesh)
    {
        triangleMesh = input;
        int fc = input.Faces.Count;
        for (int i = 0; i < fc; i++)
        {
            if (!input.Faces[i].IsTriangle)
            {
                Mesh dup = input.DuplicateMesh();
                dup.Faces.ConvertQuadsToTriangles();
                triangleMesh = dup;
                for (int j = 0; j < dup.Faces.Count; j++)
                {
                    if (!dup.Faces[j].IsTriangle)
                        return false;
                }

                return true;
            }
        }

        return true;
    }
}
