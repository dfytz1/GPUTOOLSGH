/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GHGPUPlugin.Chromodoris.MeshTools
{
    public class VertexSmooth
    {
        private int iterations;
        private double step;
        private Mesh mesh;

        private List<int[]> neighbourVerts;
        private Point3f[] topoVertLocations;
        private List<int[]> topoVertexIndices;

        public VertexSmooth(Mesh mesh, double step, int iterations)
        {
            this.iterations = iterations;
            this.step = step;
            this.mesh = mesh;
            neighbourVerts = new List<int[]>();
            topoVertLocations = new Point3f[mesh.TopologyVertices.Count];
            topoVertexIndices = new List<int[]>();
        }

        public Mesh Compute()
        {
            for (int i = 0; i < mesh.TopologyVertices.Count; i++)
            {
                int[] mvInds = mesh.TopologyVertices.MeshVertexIndices(i);
                topoVertLocations[i] = mesh.Vertices[mvInds[0]];
                topoVertexIndices.Add(mvInds);
                neighbourVerts.Add(mesh.TopologyVertices.ConnectedTopologyVertices(i));
            }

            for (int i = 0; i < iterations; i++)
                SmoothMultiThread();

            Point3f[] mVerts = new Point3f[mesh.Vertices.Count];
            for (int i = 0; i < topoVertLocations.Length; i++)
            {
                Point3f loc = topoVertLocations[i];
                foreach (int vInd in topoVertexIndices[i])
                    mVerts[vInd] = loc;
            }

            Mesh newMesh = new Mesh();
            newMesh.Vertices.AddVertices(mVerts);
            newMesh.Faces.AddFaces(mesh.Faces);
            return newMesh;
        }

        private void SmoothMultiThread()
        {
            var options = new System.Threading.Tasks.ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
            System.Threading.Tasks.Parallel.ForEach(
                Enumerable.Range(0, mesh.TopologyVertices.Count),
                options,
                v => SmoothTopoIndex(v));
        }

        private void SmoothTopoIndex(int v)
        {
            Point3d loc = topoVertLocations[v];
            int[] nvs = neighbourVerts[v];
            if (nvs.Length == 0) return;

            Point3d avg = new Point3d();
            foreach (int nv in nvs)
                avg += topoVertLocations[nv];
            avg /= nvs.Length;

            Vector3d pos = new Vector3d(loc) + (avg - loc) * step;
            topoVertLocations[v] = new Point3f((float)pos.X, (float)pos.Y, (float)pos.Z);
        }
    }
}
