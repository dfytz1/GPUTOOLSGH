using System.Reflection;
using System.Runtime.InteropServices;

namespace SpectralPacking.GH.Interop;

public static class NativeLoader
{
    static NativeLoader()
    {
        try
        {
            string? pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (string.IsNullOrEmpty(pluginDir))
            {
                LoadError = "Could not resolve plugin directory for native library load.";
                return;
            }

            string dylib = Path.Combine(pluginDir, "MetalBridge.dylib");
            if (!File.Exists(dylib))
            {
                LoadError = $"MetalBridge.dylib not found at \"{dylib}\".";
                return;
            }

            NativeLibrary.Load(dylib);
            IsMetalAvailable = true;
        }
        catch (Exception ex)
        {
            LoadError = ex.ToString();
            IsMetalAvailable = false;
        }
    }

    public static bool IsMetalAvailable { get; private set; }
    public static string? LoadError { get; private set; }

    public static void EnsureLoaded()
    {
    }
}
