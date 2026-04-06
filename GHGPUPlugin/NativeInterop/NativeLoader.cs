using System.Reflection;
using System.Runtime.InteropServices;

namespace GHGPUPlugin.NativeInterop;

public static class NativeLoader
{
    static NativeLoader()
    {
        try
        {
            string? pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (string.IsNullOrEmpty(pluginDir))
                return;

            string dylib = Path.Combine(pluginDir, "MetalBridge.dylib");
            if (File.Exists(dylib))
            {
                NativeLibrary.Load(dylib);
                IsMetalAvailable = true;
            }
        }
        catch
        {
            IsMetalAvailable = false;
        }
    }

    /// <summary>True only when <c>MetalBridge.dylib</c> was found and loaded successfully.</summary>
    public static bool IsMetalAvailable { get; private set; }

    public static void EnsureLoaded()
    {
    }
}
