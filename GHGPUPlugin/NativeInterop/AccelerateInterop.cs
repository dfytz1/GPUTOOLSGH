using System.Reflection;
using System.Runtime.InteropServices;

namespace GHGPUPlugin.NativeInterop;

/// <summary>P/Invoke bindings for Apple Accelerate (BLAS / LAPACK / vDSP).</summary>
public static class AccelerateInterop
{
    private static readonly string[] VecLibRoots =
    {
        "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A",
        "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework",
    };

    static AccelerateInterop()
    {
        NativeLibrary.SetDllImportResolver(typeof(AccelerateInterop).Assembly, ResolveAccelerateDll);
    }

    /// <summary>Registers the vecLib DLL resolver before any Accelerate P/Invoke runs.</summary>
    public static void EnsureLoaded()
    {
    }

    private static IntPtr ResolveAccelerateDll(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName is not ("libBLAS.dylib" or "libvDSP.dylib" or "libLAPACK.dylib"))
            return IntPtr.Zero;

        string? path = ResolveVecLibPath(libraryName);
        if (path != null && NativeLibrary.TryLoad(path, out IntPtr handle))
            return handle;

        return IntPtr.Zero;
    }

    private static string? ResolveVecLibPath(string fileName)
    {
        foreach (string root in VecLibRoots)
        {
            string candidate = Path.Combine(root, fileName);
            if (File.Exists(candidate))
                return candidate;
        }

        return null;
    }

    public const int CblasRowMajor = 101;
    public const int CblasColMajor = 102;
    public const int CblasNoTrans = 111;
    public const int CblasTrans = 112;
    public const int CblasConjTrans = 113;

    [DllImport("libBLAS.dylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm(
        int order,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        float alpha,
        float[] a,
        int lda,
        float[] b,
        int ldb,
        float beta,
        float[] c,
        int ldc);

    [DllImport("libBLAS.dylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_ssymv(
        int order,
        int uplo,
        int n,
        float alpha,
        float[] a,
        int lda,
        float[] x,
        int incX,
        float beta,
        float[] y,
        int incY);

    [DllImport("libvDSP.dylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void vDSP_dotpr(
        float[] __a,
        int __strideA,
        float[] __b,
        int __strideB,
        ref float __result,
        nuint __length);

    /// <summary>LU factorization of a general M-by-N matrix (column-major Fortran layout).</summary>
    [DllImport("libLAPACK.dylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void sgetrf_(
        ref int m,
        ref int n,
        float[] a,
        ref int lda,
        int[] ipiv,
        ref int info);
}
