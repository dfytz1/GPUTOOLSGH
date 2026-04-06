namespace GHGPUPlugin.NativeInterop;

/// <summary>One Metal device/queue/PSO set for the process — avoids reloading <c>default.metallib</c> on every Grasshopper solve (~seconds).</summary>
public static class MetalSharedContext
{
    private static readonly object Gate = new();
    private static IntPtr _ctx;
    private static bool _ready;

    /// <summary>Set when <see cref="MetalBridge.CreateContext"/> returns non-zero or a null handle.</summary>
    public static string? InitError { get; private set; }

    public static bool TryGetContext(out IntPtr ctx)
    {
        lock (Gate)
        {
            if (!_ready)
            {
                InitError = null;
                int code = MetalBridge.CreateContext(out _ctx);
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

    /// <summary>Releases the native Metal context and clears cached state (e.g. on plugin unload).</summary>
    public static void DestroyCachedContext()
    {
        if (!NativeLoader.IsMetalAvailable)
            return;

        lock (Gate)
        {
            if (_ready && _ctx != IntPtr.Zero)
            {
                MetalBridge.DestroyContext(_ctx);
                _ctx = IntPtr.Zero;
                _ready = false;
            }
        }
    }
}
