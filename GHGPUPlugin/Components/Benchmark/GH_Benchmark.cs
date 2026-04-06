using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using Grasshopper.Kernel;
using GHGPUPlugin.NativeInterop;

namespace GHGPUPlugin.Components.Benchmark;

public class GH_Benchmark : GH_Component
{
    public GH_Benchmark()
        : base(
            "Benchmark GPU",
            "BenchGPU",
            "Times a dense N×N-style workload on CPU, Apple Accelerate (BLAS), or Metal.",
            "GPUTools",
            "Benchmark")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddIntegerParameter("ProblemSize", "ProblemSize", "Matrix dimension N (N×N floats per buffer).", GH_ParamAccess.item, 256);
        pManager.AddTextParameter("Backend", "Backend", "CPU, Accelerate, or Metal.", GH_ParamAccess.item, "Accelerate");
        pManager[1].Optional = true;
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "When Backend is Metal, run on GPU; if false, uses CPU instead.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddNumberParameter("ElapsedMs", "ElapsedMs", "Elapsed wall time in milliseconds.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        int problemSize = 256;
        if (!DA.GetData("ProblemSize", ref problemSize) || problemSize < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "ProblemSize must be at least 1.");
            return;
        }

        if (problemSize > 4096)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "ProblemSize is capped at 4096 for safety.");
            problemSize = 4096;
        }

        string backendRaw = "Accelerate";
        DA.GetData("Backend", ref backendRaw);
        string backend = string.IsNullOrWhiteSpace(backendRaw) ? "Accelerate" : backendRaw.Trim();

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        int n = problemSize;
        int count = n * n;
        float[] buffer = new float[count];
        for (int i = 0; i < count; i++)
            buffer[i] = (i % 997) * 0.001f;

        float[]? a = null;
        float[]? b = null;
        float[]? c = null;

        if (string.Equals(backend, "Accelerate", StringComparison.OrdinalIgnoreCase))
        {
            a = new float[count];
            b = new float[count];
            c = new float[count];
            Array.Copy(buffer, a, count);
            Array.Copy(buffer, b, count);
        }

        var sw = Stopwatch.StartNew();

        if (string.Equals(backend, "CPU", StringComparison.OrdinalIgnoreCase))
        {
            RunCpuMatmul(buffer, n);
        }
        else if (string.Equals(backend, "Accelerate", StringComparison.OrdinalIgnoreCase))
        {
            AccelerateInterop.cblas_sgemm(
                AccelerateInterop.CblasRowMajor,
                AccelerateInterop.CblasNoTrans,
                AccelerateInterop.CblasNoTrans,
                n,
                n,
                n,
                1.0f,
                a!,
                n,
                b!,
                n,
                0.0f,
                c!,
                n);
            buffer[0] = c![0];
        }
        else if (string.Equals(backend, "Metal", StringComparison.OrdinalIgnoreCase))
        {
            if (!useGpu)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "UseGPU is false — running CPU parallel instead of Metal.");
                RunCpuMatmul(buffer, n);
            }
            else if (!NativeLoader.IsMetalAvailable)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "Metal not available — falling back to CPU parallel.");
                RunCpuMatmul(buffer, n);
            }
            else
            {
                int code = RunMetalBenchmark(buffer, n);
                if (code != 0)
                {
                    sw.Stop();
                    AddRuntimeMessage(
                        GH_RuntimeMessageLevel.Error,
                        $"Metal benchmark failed with code {code}.");
                    return;
                }
            }
        }
        else
        {
            sw.Stop();
            AddRuntimeMessage(
                GH_RuntimeMessageLevel.Warning,
                $"Unknown Backend \"{backend}\". Use CPU, Accelerate, or Metal.");
            return;
        }

        sw.Stop();
        DA.SetData("ElapsedMs", sw.Elapsed.TotalMilliseconds);
    }

    private static void RunCpuMatmul(float[] buffer, int n)
    {
        int count = n * n;
        var a = new float[count];
        var b = new float[count];
        var c = new float[count];
        Array.Copy(buffer, a, count);
        Array.Copy(buffer, b, count);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                int rowOff = i * n;
                for (int k = 0; k < n; k++)
                    sum += a[rowOff + k] * b[k * n + j];
                c[rowOff + j] = sum;
            }
        }

        buffer[0] = c[0];
    }

    private static int RunMetalBenchmark(float[] buffer, int n)
    {
        int count = n * n;
        int inner = Math.Max(4, 64_000_000 / Math.Max(count, 1));
        int outer = 2;

        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return -1;

        var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        try
        {
            IntPtr ptr = handle.AddrOfPinnedObject();
            return MetalBridge.RunBenchmark(ctx, ptr, count, inner, outer);
        }
        finally
        {
            handle.Free();
        }
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("d7c3754a-1e4c-46c8-8f77-1044cf89227c");
}
