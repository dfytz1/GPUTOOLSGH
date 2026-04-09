namespace SpectralPacking.GH;

public static class SpectralGrasshopperRegistration
{
#if DEBUG
    public static Type[] ComponentTypes { get; } =
    {
        typeof(Components.DebugOnly.GH_VoxelizeGeometry),
        typeof(Components.DebugOnly.GH_PackObjects),
        typeof(Components.DebugOnly.GH_DisassemblyCheck),
        typeof(Components.DebugOnly.GH_PackingVisualizer),
    };
#else
    public static Type[] ComponentTypes { get; } = Array.Empty<Type>();
#endif
}
