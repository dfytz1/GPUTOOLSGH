using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;

namespace GHGPUPlugin.Components.Field;

internal static class Field2DDataHelper
{
    internal static bool TryUnwrapFloat2D(IGH_Goo goo, out float[,]? tensor, out string message)
    {
        tensor = null;
        message = string.Empty;
        if (goo == null)
        {
            message = "Expected float[nx,ny].";
            return false;
        }

        if (goo is GH_ObjectWrapper ow)
        {
            if (ow.Value is float[,] a)
            {
                tensor = a;
                return true;
            }

            message = "Expected float[nx,ny] in Object Wrapper.";
            return false;
        }

        if (goo.ScriptVariable() is float[,] b)
        {
            tensor = b;
            return true;
        }

        message = "Expected float[nx,ny].";
        return false;
    }
}
