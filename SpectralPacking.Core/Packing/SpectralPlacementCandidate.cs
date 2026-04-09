using System.Numerics;

namespace SpectralPacking.Core.Packing;

public sealed class SpectralPlacementCandidate
{
    public Matrix4x4 Rotation { get; init; }
    public int Tx { get; init; }
    public int Ty { get; init; }
    public int Tz { get; init; }
    public int Sx { get; init; }
    public int Sy { get; init; }
    public int Sz { get; init; }
    public required float[] LocalOccupancy { get; init; }
    public Vector3 TranslationWorld { get; init; }
    public float Score { get; init; }
}
