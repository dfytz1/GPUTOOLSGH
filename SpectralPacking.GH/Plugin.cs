using System.Drawing;
using Grasshopper.Kernel;
using SpectralPacking.GH.Interop;

namespace SpectralPacking.GH;

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
                // ignored
            }
        }
    }

    private static readonly ContextFinalizer FinalizerHook = new();

    static Plugin()
    {
        NativeLoader.EnsureLoaded();
        _ = FinalizerHook;
        _ = SpectralGrasshopperRegistration.ComponentTypes.Length;
    }

    public Plugin()
    {
    }

    public override string Name => "SpectralPacking.GH";

    public override Bitmap? Icon => null;

    public override string Description =>
        "Spectral 3D object packing (Cui et al., TOG 2023) for Grasshopper on Apple Silicon.";

    public override Guid Id => new("d7e9c3b2-8a4f-5d8e-9c1b-3e7f2a4b6c8d");

    public override string AuthorName => "SpectralPacking";

    public override string AuthorContact => string.Empty;

    public override string Version => "0.1.0";

    public override GH_LibraryLicense License => GH_LibraryLicense.opensource;
}
