using System;
using Grasshopper.Kernel;

namespace GHGPUPlugin.Chromodoris;

/// <summary>
/// Grasshopper discovers public <see cref="GH_Component"/> subclasses in this assembly automatically.
/// This list groups the voxel-tab workflow components for maintainers and keeps references from being trimmed.
/// </summary>
internal static class GrasshopperVoxelTabRegistration
{
    internal static readonly Type[] ComponentTypes =
    {
#if DEBUG
        typeof(VoxelSimpAutoComponent),
#endif
        typeof(VoxelSimpTopologyComponent),
        typeof(VoxelDensitySliceComponent),
        typeof(VoxelDensitySmoothComponent),
    };
}
