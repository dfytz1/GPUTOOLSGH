using System.Drawing;
using System.IO;
using System.Reflection;

namespace GHGPUPlugin.Chromodoris
{
    internal static class Icons
    {
        private static Bitmap _sample;
        private static Bitmap _sampleCustom;
        private static Bitmap _isoSurface;
        private static Bitmap _isoSurfaceSl;
        private static Bitmap _closeVolume;
        private static Bitmap _smooth;
        private static Bitmap _smoothMasked;
        private static Bitmap _laplaceField;
        private static Bitmap _laplaceProximity;
        private static Bitmap _voxelDesignDomain;
        private static Bitmap _paintRegions;

        public static Bitmap Sample       => _sample       ??= Load("sample.png");
        public static Bitmap SampleCustom => _sampleCustom ??= Load("samplecustom.png");
        public static Bitmap IsoSurface   => _isoSurface   ??= Load("isosurface.png");
        public static Bitmap IsoSurfaceSl => _isoSurfaceSl ??= Load("isosurface_sl.png");
        public static Bitmap CloseVolume  => _closeVolume  ??= Load("closevolume.png");
        public static Bitmap Smooth       => _smooth       ??= Load("smooth.png");
        public static Bitmap SmoothMasked => _smoothMasked ??= Load("smooth_masked.png");
        public static Bitmap LaplaceField => _laplaceField ??= Load("laplace_field.png");
        public static Bitmap LaplaceProximity => _laplaceProximity ??= Load("laplace_proximity.png");
        public static Bitmap VoxelDesignDomain => _voxelDesignDomain ??= Load("voxel_design_domain.png");
        public static Bitmap PaintRegions => _paintRegions ??= Load("paint_regions.png");

        private static Bitmap Load(string fileName)
        {
            try
            {
                var asm = Assembly.GetExecutingAssembly();
                using Stream stream = asm.GetManifestResourceStream($"GHGPUPlugin.Chromodoris.Resources.{fileName}");
                if (stream == null) return null;
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                ms.Position = 0;
                return new Bitmap(ms);
            }
            catch
            {
                return null;
            }
        }
    }
}
