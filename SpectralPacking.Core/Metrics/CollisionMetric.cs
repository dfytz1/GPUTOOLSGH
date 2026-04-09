namespace SpectralPacking.Core.Metrics;

public static class CollisionMetric
{
    public static void Compute(
        IFFTBackend fft,
        ReadOnlySpan<float> paddedObject,
        ReadOnlySpan<float> paddedContainer,
        int px, int py, int pz,
        Span<float> zetaOut) =>
        fft.CorrelateReal3D(paddedObject, paddedContainer, px, py, pz, zetaOut);
}
