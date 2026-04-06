/*
 * Based on toxiclibs by Karsten Schmidt (LGPL 2.1)
 * https://bitbucket.org/postspectacular/toxiclibs
 */

using System;

namespace GHGPUPlugin.Chromodoris
{
    public class VolumetricSpace
    {
        public int resX, resY, resZ;
        public int resX1, resY1, resZ1;
        public int sliceRes;
        public int numCells;

        private float[,,] data;

        public VolumetricSpace(float[,,] isoData)
        {
            resX = isoData.GetLength(0);
            resY = isoData.GetLength(1);
            resZ = isoData.GetLength(2);
            resX1 = resX - 1;
            resY1 = resY - 1;
            resZ1 = resZ - 1;
            sliceRes = resX * resY;
            numCells = sliceRes * resZ;
            data = isoData;
        }

        public double getVoxelAt(int index)
        {
            int xVal = 0, yVal = 0, zVal = 0;

            if (index >= sliceRes)
            {
                zVal = (int)Math.Floor((double)index / sliceRes);
                index = index - zVal * sliceRes;
            }

            if (index >= resX)
            {
                yVal = (int)Math.Floor((double)index / resX);
                index = index - yVal * resX;
            }

            xVal = index;
            return data[xVal, yVal, zVal];
        }

        public double getVoxelAt(int x, int y, int z) => data[x, y, z];
    }
}
