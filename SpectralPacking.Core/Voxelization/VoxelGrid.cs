namespace SpectralPacking.Core.Voxelization;

/// <summary>
/// Row-major occupancy/scalar field: index = x + y * Width + z * Width * Height (x fastest).
/// </summary>
public sealed class VoxelGrid
{
    public VoxelGrid(int width, int height, int depth, float[] data)
    {
        if (data.Length != width * height * depth)
            throw new ArgumentException("Data length must equal W*H*D.");
        Width = width;
        Height = height;
        Depth = depth;
        Data = data;
    }

    public static VoxelGrid CreateZero(int width, int height, int depth) =>
        new(width, height, depth, new float[width * height * depth]);

    public int Width { get; }
    public int Height { get; }
    public int Depth { get; }
    public float[] Data { get; }

    public int LinearSize => Width * Height * Depth;

    public int Index(int x, int y, int z) => x + y * Width + z * Width * Height;

    public float this[int x, int y, int z]
    {
        get => Data[Index(x, y, z)];
        set => Data[Index(x, y, z)] = value;
    }

    public VoxelGrid Clone() => new(Width, Height, Depth, (float[])Data.Clone());

    public void CopyFrom(VoxelGrid other)
    {
        if (other.Width != Width || other.Height != Height || other.Depth != Depth)
            throw new ArgumentException("Grid dimensions must match.");
        Array.Copy(other.Data, Data, Data.Length);
    }

    public double OccupiedVolume(float voxelSize) =>
        Data.Sum(v => v > 0.5f ? 1 : 0) * voxelSize * voxelSize * voxelSize;
}
