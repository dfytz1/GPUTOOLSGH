using System;

namespace GHGPUPlugin;

/// <summary>References GPUTools-tab components so aggressive trimming does not drop them.</summary>
internal static class GrasshopperGpuToolsRegistration
{
#if DEBUG
    internal static readonly Type[] ComponentTypes =
    {
        typeof(Components.Field.GH_ReactionDiffusion2DGPU),
        typeof(Components.Field.GH_ReactionDiffusionMeshColorGPU),
        typeof(Components.Field.GH_MeshDisplaceField2DGPU),
        typeof(Components.Field.GH_Field2DPreviewMeshGPU),
        typeof(Components.DebugOnly.GH_AnisotropicCvtRemeshGPU),
        typeof(Components.DebugOnly.GH_MeshCollisionGPU),
    };
#else
    internal static readonly Type[] ComponentTypes = Array.Empty<Type>();
#endif
}
