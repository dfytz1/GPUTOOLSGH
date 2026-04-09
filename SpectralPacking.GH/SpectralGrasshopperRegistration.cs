namespace SpectralPacking.GH;

public static class SpectralGrasshopperRegistration
{
    public static Type[] ComponentTypes { get; } =
    {
        typeof(Components.GH_VoxelizeGeometry),
        typeof(Components.GH_PackObjects),
        typeof(Components.GH_DisassemblyCheck),
        typeof(Components.GH_PackingVisualizer),
    };
}
