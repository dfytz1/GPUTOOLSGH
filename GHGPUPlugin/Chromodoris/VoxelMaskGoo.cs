using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Chromodoris
{
    /// <summary>
    /// Unwrap float[,,] voxel grids from Generic / ObjectWrapper wires without triggering GH's Mesh → Single[,,] cast.
    /// </summary>
    internal static class VoxelMaskGoo
    {
        public static bool TryGetFloatTensor3(IGH_DataAccess DA, int paramIndex, GH_Component owner, out float[,,] tensor,
            string inputDescription = null)
        {
            tensor = null;
            IGH_Goo goo = null;
            if (!DA.GetData(paramIndex, ref goo) || goo == null)
                return false;
            return TryUnwrap(goo, owner, out tensor, inputDescription);
        }

        public static bool TryUnwrap(IGH_Goo goo, GH_Component owner, out float[,,] tensor, string inputDescription = null)
        {
            tensor = null;
            if (goo == null) return false;
            string label = string.IsNullOrEmpty(inputDescription) ? "This input" : inputDescription;

            if (goo is GH_ObjectWrapper ow)
            {
                if (ow.Value is float[,,] a)
                {
                    tensor = a;
                    return true;
                }

                if (ow.Value == null) return false;
                ReportWrongType(owner, ow.Value, label);
                return false;
            }

            object sv = goo.ScriptVariable();
            if (sv is float[,,] b)
            {
                tensor = b;
                return true;
            }

            if (sv is Mesh)
            {
                owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    label + " expects a voxel grid float[x,y,z], not a Mesh. " +
                    "Use Voxel Design Domain output I (InsideMask) for the domain grid; use Voxel Paint Regions outputs S/L for support/load masks; " +
                    "put your solid design mesh only on Voxel Design Domain's DesignMesh, and support/load volumes on SupportGeometry / LoadGeometry.");
                return false;
            }

            ReportWrongType(owner, sv, label);
            return false;
        }

        private static void ReportWrongType(GH_Component owner, object sv, string label)
        {
            string name = sv == null ? "null" : sv.GetType().Name;
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                label + " expects voxel data float[x,y,z]. Got: " + name + ".");
        }
    }
}
