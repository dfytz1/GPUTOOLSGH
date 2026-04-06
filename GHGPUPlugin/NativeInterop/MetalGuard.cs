using Grasshopper.Kernel;

namespace GHGPUPlugin.NativeInterop;

/// <summary>Centralized Metal readiness checks with runtime messages for Grasshopper components.</summary>
public static class MetalGuard
{
    /// <summary>
    /// Call before using Metal from a component. Returns false and posts a GH error if the native library or context is unavailable.
    /// </summary>
    public static bool EnsureReady(GH_Component component)
    {
        if (!NativeLoader.IsMetalAvailable)
        {
            component.AddRuntimeMessage(
                GH_RuntimeMessageLevel.Error,
                $"MetalBridge not loaded: {NativeLoader.LoadError ?? "unknown error"}");
            return false;
        }

        if (!MetalSharedContext.TryGetContext(out _))
        {
            component.AddRuntimeMessage(
                GH_RuntimeMessageLevel.Error,
                $"Metal context failed: {MetalSharedContext.InitError ?? "unknown error"}");
            return false;
        }

        return true;
    }
}
