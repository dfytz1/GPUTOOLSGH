using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace SpectralPacking.Core.Metrics;

/// <summary>3D FFT-based linear cross-correlation on zero-padded real grids (row-major x fastest).</summary>
public interface IFFTBackend
{
    /// <summary>
    /// Linear cross-correlation of <paramref name="paddedA"/> with <paramref name="paddedB"/> at all shifts (output length Px×Py×Pz).
    /// Uses R = IFFT( FFT(A) · conj(FFT(B)) ) with separable 3D complex FFT.
    /// </summary>
    void CorrelateReal3D(
        ReadOnlySpan<float> paddedA,
        ReadOnlySpan<float> paddedB,
        int px, int py, int pz,
        Span<float> correlationRealOut);
}

public static class FftGridDims
{
    public static int NextPow2(int n)
    {
        n = Math.Max(1, n);
        int p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    /// <summary>FFT size per axis for linear correlation of two grids of size (nx,ny,nz).</summary>
    public static (int px, int py, int pz) PaddedDims(int nx, int ny, int nz)
    {
        int px = NextPow2(2 * nx - 1);
        int py = NextPow2(2 * ny - 1);
        int pz = NextPow2(2 * nz - 1);
        return (px, py, pz);
    }
}

public sealed class MathNetFFTBackend : IFFTBackend
{
    public void CorrelateReal3D(
        ReadOnlySpan<float> paddedA,
        ReadOnlySpan<float> paddedB,
        int px, int py, int pz,
        Span<float> correlationRealOut)
    {
        int n = px * py * pz;
        if (paddedA.Length < n || paddedB.Length < n || correlationRealOut.Length < n)
            throw new ArgumentException("Buffer sizes must be at least Px*Py*Pz.");

        var fa = new Complex[n];
        var fb = new Complex[n];
        for (int i = 0; i < n; i++)
        {
            fa[i] = new Complex(paddedA[i], 0);
            fb[i] = new Complex(paddedB[i], 0);
        }

        Fft3D.Forward(fa, px, py, pz);
        Fft3D.Forward(fb, px, py, pz);

        for (int i = 0; i < n; i++)
            fa[i] *= Complex.Conjugate(fb[i]);

        Fft3D.Inverse(fa, px, py, pz);

        for (int i = 0; i < n; i++)
            correlationRealOut[i] = (float)fa[i].Real;
    }
}

/// <summary>Uses MetalBridge spectral entry points when available; otherwise delegates to <see cref="MathNetFFTBackend"/>.</summary>
public sealed class MetalFFTBackend : IFFTBackend
{
    private readonly MathNetFFTBackend _cpu = new();

    public void CorrelateReal3D(
        ReadOnlySpan<float> paddedA,
        ReadOnlySpan<float> paddedB,
        int px, int py, int pz,
        Span<float> correlationRealOut)
    {
        // Native batched Metal FFT can replace this path; current bridge focuses on voxel + DF kernels.
        _cpu.CorrelateReal3D(paddedA, paddedB, px, py, pz, correlationRealOut);
    }
}

internal static class Fft3D
{
    public static void Forward(Complex[] buf, int nx, int ny, int nz)
    {
        TransformAxis(buf, nx, ny, nz, axis: 0, forward: true);
        TransformAxis(buf, nx, ny, nz, axis: 1, forward: true);
        TransformAxis(buf, nx, ny, nz, axis: 2, forward: true);
    }

    public static void Inverse(Complex[] buf, int nx, int ny, int nz)
    {
        TransformAxis(buf, nx, ny, nz, axis: 0, forward: false);
        TransformAxis(buf, nx, ny, nz, axis: 1, forward: false);
        TransformAxis(buf, nx, ny, nz, axis: 2, forward: false);
    }

    private static void TransformAxis(Complex[] buf, int nx, int ny, int nz, int axis, bool forward)
    {
        int lineLen = axis == 0 ? nx : axis == 1 ? ny : nz;
        var line = new Complex[lineLen];

        if (axis == 0)
        {
            for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
            {
                int baseIdx = y * nx + z * nx * ny;
                for (int x = 0; x < nx; x++)
                    line[x] = buf[baseIdx + x];
                Run1D(line, forward);
                for (int x = 0; x < nx; x++)
                    buf[baseIdx + x] = line[x];
            }
        }
        else if (axis == 1)
        {
            for (int z = 0; z < nz; z++)
            for (int x = 0; x < nx; x++)
            {
                int baseIdx = x + z * nx * ny;
                for (int y = 0; y < ny; y++)
                    line[y] = buf[baseIdx + y * nx];
                Run1D(line, forward);
                for (int y = 0; y < ny; y++)
                    buf[baseIdx + y * nx] = line[y];
            }
        }
        else
        {
            for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                int baseIdx = x + y * nx;
                for (int z = 0; z < nz; z++)
                    line[z] = buf[baseIdx + z * nx * ny];
                Run1D(line, forward);
                for (int z = 0; z < nz; z++)
                    buf[baseIdx + z * nx * ny] = line[z];
            }
        }
    }

    private static void Run1D(Complex[] line, bool forward)
    {
        if (forward)
            Fourier.Forward(line, FourierOptions.AsymmetricScaling);
        else
            Fourier.Inverse(line, FourierOptions.AsymmetricScaling);
    }
}
