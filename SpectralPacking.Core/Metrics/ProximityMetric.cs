namespace SpectralPacking.Core.Metrics;

public static class ProximityMetric
{
    public static void Compute(
        IFFTBackend fft,
        ReadOnlySpan<float> paddedObject,
        ReadOnlySpan<float> paddedPhi,
        int px, int py, int pz,
        Span<float> rhoOut) =>
        fft.CorrelateReal3D(paddedObject, paddedPhi, px, py, pz, rhoOut);
}
