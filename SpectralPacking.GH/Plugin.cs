using System.Drawing;
using Grasshopper.Kernel;
using SpectralPacking.GH.Interop;

namespace SpectralPacking.GH;

public class Plugin : GH_AssemblyInfo
{
#if DEBUG
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
#endif

    static Plugin()
    {
#if DEBUG
        NativeLoader.EnsureLoaded();
        _ = FinalizerHook;
#endif
        _ = SpectralGrasshopperRegistration.ComponentTypes.Length;
    }

    public Plugin()
    {
    }

    public override string Name => "SpectralPacking.GH";

    public override Bitmap? Icon => null;

    public override string Description =>
#if DEBUG
        "Spectral 3D object packing (Cui et al., TOG 2023) for Grasshopper on Apple Silicon. Debug build includes Spectral Pack components.";
#else
        "SpectralPacking.GH (release shell). Build Debug for Spectral Pack Grasshopper components.";
#endif

    public override Guid Id => new("d7e9c3b2-8a4f-5d8e-9c1b-3e7f2a4b6c8d");

    public override string AuthorName => "SpectralPacking";

    public override string AuthorContact => string.Empty;

#if DEBUG
    public override string Version => "0.2.0-debug";
#else
    public override string Version => "0.2.0";
#endif

    public override GH_LibraryLicense License => GH_LibraryLicense.opensource;
}
