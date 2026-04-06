namespace GHGPUPlugin.NativeInterop;

/// <summary>One Metal device/queue/PSO set for the process — avoids reloading <c>default.metallib</c> on every Grasshopper solve (~seconds).</summary>
public static class MetalSharedContext
{
    private static readonly object Gate = new();
    private static IntPtr _ctx;
    private static bool _ready;

    public static bool TryGetContext(out IntPtr ctx)
    {
        lock (Gate)
        {
            if (!_ready)
            {
                int code = MetalBridge.CreateContext(out _ctx);
                if (code != 0 || _ctx == IntPtr.Zero)
                {
                    ctx = IntPtr.Zero;
                    return false;
                }

                _ready = true;
            }

            ctx = _ctx;
            return true;
        }
    }
}
