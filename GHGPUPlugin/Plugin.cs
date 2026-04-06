using System.Drawing;
using Grasshopper.Kernel;
using GHGPUPlugin.Chromodoris;
using GHGPUPlugin.NativeInterop;

namespace GHGPUPlugin;

/// <summary>Grasshopper assembly metadata and native bridge bootstrap.</summary>
public class Plugin : GH_AssemblyInfo
{
    private sealed class ContextFinalizer
    {
        ~ContextFinalizer()
        {
            try
            {
                MetalSharedContext.DestroyCachedContext();
            }
            catch
            {
                // Best-effort teardown during AppDomain unload.
            }
        }
    }

    private static readonly ContextFinalizer _finalizer = new();

    static Plugin()
    {
        NativeLoader.EnsureLoaded();
        AccelerateInterop.EnsureLoaded();
        _ = GrasshopperVoxelTabRegistration.ComponentTypes.Length;
        _ = _finalizer;
    }

    public Plugin()
    {
    }

    public override string Name => "GHGPUPlugin";

    public override Bitmap? Icon => null;

    public override string Description =>
        "Metal GPU and Accelerate SIMD helpers for topology optimisation, linear algebra, and data relationships on Apple Silicon.";

    public override Guid Id => new("a8f3c2d1-5e4b-4a7c-9d0e-1f2a3b4c5d6e");

    public override string AuthorName => "GHGPUPlugin";

    public override string AuthorContact => string.Empty;

    public override string Version => "1.0.0";

    public override GH_LibraryLicense License => GH_LibraryLicense.opensource;
}
