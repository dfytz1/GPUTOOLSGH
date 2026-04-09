using System.Runtime.InteropServices;

namespace SpectralPacking.GH.Interop;

public static class MetalSharedContext
{
    private const string LibName = "MetalBridge";

    [DllImport(LibName, EntryPoint = "mb_create_context", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CreateContext(out IntPtr ctx);

    [DllImport(LibName, EntryPoint = "mb_destroy_context", CallingConvention = CallingConvention.Cdecl)]
    private static extern void DestroyContext(IntPtr ctx);

    private static readonly object Gate = new();
    private static IntPtr _ctx;
    private static bool _ready;

    public static string? InitError { get; private set; }

    public static bool TryGetContext(out IntPtr ctx)
    {
        if (!NativeLoader.IsMetalAvailable)
        {
            InitError = NativeLoader.LoadError ?? "Metal native library not loaded.";
            ctx = IntPtr.Zero;
            return false;
        }

        lock (Gate)
        {
            if (!_ready)
            {
                InitError = null;
                int code = CreateContext(out _ctx);
                if (code != 0 || _ctx == IntPtr.Zero)
                {
                    InitError = $"mb_create_context returned error code {code}";
                    ctx = IntPtr.Zero;
                    return false;
                }

                _ready = true;
            }

            ctx = _ctx;
            return true;
        }
    }

    public static void DestroyCachedContext()
    {
        if (!NativeLoader.IsMetalAvailable)
            return;

        lock (Gate)
        {
            if (_ready && _ctx != IntPtr.Zero)
            {
                DestroyContext(_ctx);
                _ctx = IntPtr.Zero;
                _ready = false;
            }
        }
    }
}
