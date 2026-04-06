using GHGPUPlugin.Chromodoris.MeshTools;
using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Diagnostics;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Laplacian smoothing with vertices under Support / Load voxels held fixed (same B, Cc, masks as Build IsoSurface).
    /// </summary>
    public class ConstrainedSmoothComponent : GH_Component
    {
        public ConstrainedSmoothComponent()
          : base("Smooth Masked GPU", "SmoothMaskGPU",
              "Laplacian smooth a mesh while locking vertices that fall in Support and/or Load voxels. " +
              "Wire the same BoundingBox, masks, and CellCentered flag as Build IsoSurface.",
              "GPUTools", "Mesh")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Mesh to smooth (e.g. iso from Build IsoSurface).", GH_ParamAccess.item);
            pManager.AddBoxParameter("BoundingBox", "B", "Same box as Build IsoSurface / Voxel Design Domain.", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupportMask", "S", "Support voxel mask float[x,y,z] (same resolution as density).", GH_ParamAccess.item);
            pManager.AddGenericParameter("LoadMask", "L", "Load voxel mask float[x,y,z].", GH_ParamAccess.item);
            pManager.AddBooleanParameter("CellCentered", "Cc", "Must match Build IsoSurface (true for Voxel Design / SIMP workflow).", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("FixSupport", "Fs", "Lock vertices in support voxels.", GH_ParamAccess.item, true);
            pManager.AddBooleanParameter("FixLoad", "Fl", "Lock vertices in load voxels.", GH_ParamAccess.item, true);
            pManager.AddIntegerParameter("Dilate", "D",
                "Expand constraint region by this many voxel rings (0–3) so boundary vertices stay put.", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("StepSize", "St", "Laplacian step 0…1.", GH_ParamAccess.item, 0.5);
            pManager.AddIntegerParameter("Iterations", "I", "Smoothing iterations.", GH_ParamAccess.item, 1);
            pManager.AddBooleanParameter("UseGPU", "GPU",
                "Use Metal Laplacian per iteration; constraint mask re-applied in C# after each GPU pass.",
                GH_ParamAccess.item, true);
            pManager[7].Optional = true;
            pManager[8].Optional = true;
            pManager[9].Optional = true;
            pManager[10].Optional = true;
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Smoothed mesh.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Mesh mesh = null;
            Box box = new Box();
            float[,,] support = null, load = null;
            bool cellCentered = true, fixSupport = true, fixLoad = true;
            int dilate = 1, iterations = 1;
            double step = 0.5;

            if (!DA.GetData(0, ref mesh)) return;
            if (!DA.GetData(1, ref box)) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask")) return;
            if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask")) return;
            DA.GetData(4, ref cellCentered);
            DA.GetData(5, ref fixSupport);
            DA.GetData(6, ref fixLoad);
            DA.GetData(7, ref dilate);
            DA.GetData(8, ref step);
            DA.GetData(9, ref iterations);
            bool useGpu = true;
            DA.GetData(10, ref useGpu);
            NativeLoader.EnsureLoaded();

            int nx = support.GetLength(0), ny = support.GetLength(1), nz = support.GetLength(2);
            if (load.GetLength(0) != nx || load.GetLength(1) != ny || load.GetLength(2) != nz)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Support and Load masks must have the same dimensions.");
                return;
            }

            if (iterations < 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be >= 0.");
                return;
            }

            if (step < 0 || step > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "StepSize must be between 0 and 1.");
                return;
            }

            if (dilate < 0 || dilate > 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Dilate must be between 0 and 3.");
                return;
            }

            if (iterations == 0 || step == 0 || (!fixSupport && !fixLoad))
            {
                DA.SetData(0, mesh.DuplicateMesh());
                return;
            }

            if (!mesh.IsValid)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is invalid.");
                return;
            }

            bool[] flags;
            try
            {
                flags = ConstrainedVertexSmooth.BuildConstraints(mesh, box, nx, ny, nz, support, load,
                    cellCentered, fixSupport, fixLoad, dilate);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
                return;
            }

            int nLock = 0;
            for (int i = 0; i < flags.Length; i++)
                if (flags[i]) nLock++;

            Mesh outMesh;
            bool gpuOk = false;
            var sw = Stopwatch.StartNew();

            if (useGpu && MetalSharedContext.TryGetContext(out IntPtr ctx))
            {
                try
                {
                    int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(mesh);
                    int nTopo = neighbors.Length;
                    MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);

                    var tv = mesh.TopologyVertices;
                    var vx = new float[nTopo];
                    var vy = new float[nTopo];
                    var vz = new float[nTopo];
                    for (int t = 0; t < nTopo; t++)
                    {
                        var p = tv[t];
                        vx[t] = p.X;
                        vy[t] = p.Y;
                        vz[t] = p.Z;
                    }

                    var ox = new float[nTopo];
                    var oy = new float[nTopo];
                    var oz = new float[nTopo];
                    var meshVertToTopo = new int[mesh.Vertices.Count];
                    for (int ti = 0; ti < nTopo; ti++)
                    {
                        int[] mvInds = tv.MeshVertexIndices(ti);
                        for (int k = 0; k < mvInds.Length; k++)
                            meshVertToTopo[mvInds[k]] = ti;
                    }

                    for (int mv = 0; mv < mesh.Vertices.Count; mv++)
                    {
                        int ti = meshVertToTopo[mv];
                        if (flags[ti])
                        {
                            var p = mesh.Vertices[mv];
                            ox[ti] = (float)p.X;
                            oy[ti] = (float)p.Y;
                            oz[ti] = (float)p.Z;
                        }
                    }

                    for (int it = 0; it < iterations; it++)
                    {
                        int code = MetalBridge.RunLaplacianIterations(
                            ctx, vx, vy, vz, adjFlat, rowOffsets, nTopo, (float)step, 1);
                        if (code != 0)
                            throw new Exception($"RunLaplacianIterations returned {code}");
                        for (int t = 0; t < nTopo; t++)
                        {
                            if (flags[t])
                            {
                                vx[t] = ox[t];
                                vy[t] = oy[t];
                                vz[t] = oz[t];
                            }
                        }
                    }

                    outMesh = mesh.DuplicateMesh();
                    for (int mv = 0; mv < outMesh.Vertices.Count; mv++)
                    {
                        int ti = meshVertToTopo[mv];
                        outMesh.Vertices.SetVertex(mv, vx[ti], vy[ti], vz[ti]);
                    }

                    outMesh.Normals.ComputeNormals();
                    gpuOk = true;
                }
                catch (Exception ex)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU smooth failed: {ex.Message} — CPU fallback.");
                    outMesh = new ConstrainedVertexSmooth(mesh, step, iterations, flags).Compute();
                }
            }
            else
            {
                outMesh = new ConstrainedVertexSmooth(mesh, step, iterations, flags).Compute();
            }

            sw.Stop();
            if (gpuOk)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    $"GPU constrained smooth ({sw.ElapsedMilliseconds} ms)");

            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Locked {nLock} / {flags.Length} topology vertices (support/load voxels × dilate).");

            DA.SetData(0, outMesh);
        }

        public override GH_Exposure Exposure => GH_Exposure.quinary;

        protected override System.Drawing.Bitmap Icon => Icons.SmoothMasked;

        public override Guid ComponentGuid => new Guid("11cecc60-b658-46e5-9f7b-f5c8c44d20e0");
    }
}
