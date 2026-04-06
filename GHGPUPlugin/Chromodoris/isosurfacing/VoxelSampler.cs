/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GHGPUPlugin.Chromodoris
{
    internal class VoxelSampler
    {
        public int xRes;
        public int yRes;
        public int zRes;

        private Box _box;
        private double xSpace;
        private double ySpace;
        private double zSpace;

        private RTree rTree;
        private List<Point3d> points;
        private List<double> values;
        private double range;
        private double rangeSq;
        private bool bulge = false;
        private bool linear;

        private Transform xfm;
        private Transform xfmToGrid;
        private Transform xfmFromGrid;
        private bool useXfm = false;

        private float[,,] gdata;

        public VoxelSampler(List<Point3d> points, List<double> values, double cellSize, double range, bool bulge, bool linear)
        {
            this.points = points;
            this.values = values;
            this.range = range;
            rangeSq = this.range * this.range;
            this.bulge = bulge;
            this.linear = linear;
            CreateEnvironment(cellSize, out _box, out xRes, out yRes, out zRes);
        }

        public VoxelSampler(List<Point3d> points, List<double> values, double cellSize, Box box, double range, bool bulge, bool linear)
        {
            this.points = points;
            this.values = values;
            this.range = range;
            rangeSq = this.range * this.range;
            this.bulge = bulge;
            this.linear = linear;
            CreateEnvironment(cellSize, box, out _box, out xRes, out yRes, out zRes);

            if (_box.Plane.ZAxis != Vector3d.ZAxis || _box.Plane.YAxis != Vector3d.YAxis || _box.Plane.XAxis != Vector3d.XAxis)
            {
                xfm = GetBoxTransform(_box, xRes, yRes, zRes);
                useXfm = true;
            }
        }

        public VoxelSampler(List<Point3d> points, List<double> values, Box box, int resX, int resY, int resZ, double range, bool bulge, bool linear)
        {
            this.points = points;
            this.values = values;
            this.range = range;
            rangeSq = this.range * this.range;
            this.bulge = bulge;
            this.linear = linear;
            _box = box;
            _box.RepositionBasePlane(box.Center);
            xRes = resX;
            yRes = resY;
            zRes = resZ;

            if (_box.Plane.ZAxis != Vector3d.ZAxis || _box.Plane.YAxis != Vector3d.YAxis || _box.Plane.XAxis != Vector3d.XAxis)
            {
                xfm = GetBoxTransform(_box, xRes, yRes, zRes);
                useXfm = true;
            }
        }

        public Transform GetBoxTransform(Box box, int x, int y, int z)
        {
            Box gridBox = new Box(Plane.WorldXY, new Interval(0, x), new Interval(0, y), new Interval(0, z));
            gridBox.RepositionBasePlane(gridBox.Center);
            Transform trans = Transform.PlaneToPlane(gridBox.Plane, box.Plane);
            trans *= Transform.Scale(gridBox.Plane, box.X.Length / gridBox.X.Length, box.Y.Length / gridBox.Y.Length, box.Z.Length / gridBox.Z.Length);
            return trans;
        }

        public void Initialize()
        {
            xSpace = (_box.X.Max - _box.X.Min) / (xRes - 1);
            ySpace = (_box.Y.Max - _box.Y.Min) / (yRes - 1);
            zSpace = (_box.Z.Max - _box.Z.Min) / (zRes - 1);

            if (values.Count == 1)
            {
                double val = values[0];
                for (int i = 0; i < points.Count - 1; i++)
                    values.Add(val);
            }
            else if (values.Count < points.Count)
            {
                for (int i = 0; i < points.Count; i++)
                    values.Add(1);
            }

            Box gridbox = new Box(Plane.WorldXY, new Interval(0, xRes - 1), new Interval(0, yRes - 1), new Interval(0, zRes - 1));
            gridbox.RepositionBasePlane(gridbox.Center);
            xfmToGrid = BoxToBoxTransform(_box, gridbox);
            xfmFromGrid = BoxToBoxTransform(gridbox, _box);

            rTree = new RTree();
            int ind = 0;
            foreach (Point3d p in points)
            {
                rTree.Insert(p, ind);
                ind++;
            }

            gdata = new float[xRes, yRes, zRes];
        }

        public Transform BoxToBoxTransform(Box source, Box target)
        {
            Transform trans = Transform.PlaneToPlane(source.Plane, target.Plane);
            trans *= Transform.Scale(source.Plane, target.X.Length / source.X.Length, target.Y.Length / source.Y.Length, target.Z.Length / source.Z.Length);
            return trans;
        }

        public Box Box => _box;
        public float[,,] Data => gdata;

        public void ExecuteMultiThread()
        {
            System.Threading.Tasks.ParallelOptions pLel = new System.Threading.Tasks.ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
            System.Threading.Tasks.Parallel.ForEach(Enumerable.Range(0, zRes), pLel, z => AssignSection(z));
        }

        public void ExecuteInverseMultiThread()
        {
            System.Threading.Tasks.ParallelOptions pLel = new System.Threading.Tasks.ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
            System.Threading.Tasks.Parallel.For(0, points.Count, pLel, i => AssignPointValue(i));
        }

        public void AssignPointValue(int i)
        {
            Point3d p = points[i];
            Point3d ptX = new Point3d(p);
            ptX.Transform(xfmToGrid);

            int[] closeCell = new int[] { (int)ptX.X, (int)ptX.Y, (int)ptX.Z };

            if (ptX.X < 0 || ptX.X >= xRes || ptX.Y < 0 || ptX.Y >= yRes || ptX.Z < 0 || ptX.Z >= zRes)
                return;

            if (AssignValueFromScaledPoint(ptX, closeCell, values[i]))
            {
                int indstep = 0;
                bool inprogress = true;
                while (inprogress)
                {
                    inprogress = false;
                    indstep++;
                    List<int[]> neighbours = GetNeighbouringCells(closeCell[0], closeCell[1], closeCell[2], indstep);
                    foreach (int[] cell in neighbours)
                    {
                        if (AssignValueFromScaledPoint(ptX, cell, values[i]))
                            inprogress = true;
                    }
                }
            }
        }

        public bool AssignValueFromScaledPoint(Point3d ptX, int[] closeCell, double scalar)
        {
            Vector3d closestVec = new Point3d(closeCell[0], closeCell[1], closeCell[2]) - ptX;
            closestVec.Transform(xfmFromGrid);

            if (linear)
            {
                double len = closestVec.Length;
                if (len > range) return false;
                AssignValuesToCell(closeCell[0], closeCell[1], closeCell[2], scalar / len);
                return true;
            }
            else
            {
                double len = closestVec.SquareLength;
                if (len > rangeSq) return false;
                AssignValuesToCell(closeCell[0], closeCell[1], closeCell[2], scalar / (len * len));
                return true;
            }
        }

        public List<int[]> GetNeighbouringCells(int cx, int cy, int cz, int indstep)
        {
            List<int[]> neighbours = new List<int[]>();
            for (int x = cx - indstep; x <= cx + indstep; x++)
            {
                for (int y = cy - indstep; y <= cy + indstep; y++)
                {
                    for (int z = cz - indstep; z <= cz + indstep; z++)
                    {
                        if (x < 0 || x >= xRes || y < 0 || y >= yRes || z < 0 || z >= zRes)
                            continue;
                        if (x == cx - indstep || x == cx + indstep ||
                            y == cy - indstep || y == cy + indstep ||
                            z == cz - indstep || z == cz + indstep)
                        {
                            neighbours.Add(new int[] { x, y, z });
                        }
                    }
                }
            }
            return neighbours;
        }

        public void AssignValuesToCell(int cx, int cy, int cz, double value)
        {
            if (bulge)
            {
                lock (gdata)
                    gdata[cx, cy, cz] += (float)value;
            }
            else
            {
                lock (gdata)
                {
                    if (value > gdata[cx, cy, cz])
                        gdata[cx, cy, cz] = (float)value;
                }
            }
        }

        public void AssignSection(int z)
        {
            if (!useXfm)
            {
                double zVal = _box.Z.Min + z * zSpace + _box.Center.Z;
                for (int y = 0; y < yRes; y++)
                {
                    double yVal = _box.Y.Min + y * ySpace + _box.Center.Y;
                    for (int x = 0; x < xRes; x++)
                    {
                        double xVal = _box.X.Min + x * xSpace + _box.Center.X;
                        gdata[x, y, z] = (float)AssignValues(xVal, yVal, zVal);
                    }
                }
            }
            else
            {
                for (int y = 0; y < yRes; y++)
                {
                    for (int x = 0; x < xRes; x++)
                    {
                        Point3d p = new Point3d(x, y, z);
                        p.Transform(xfm);
                        gdata[x, y, z] = (float)AssignValues(p.X, p.Y, p.Z);
                    }
                }
            }
        }

        public double DistanceSq(double x1, double y1, double z1, double x2, double y2, double z2)
            => (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);

        public double Distance(double x1, double y1, double z1, double x2, double y2, double z2)
            => Math.Sqrt(DistanceSq(x1, y1, z1, x2, y2, z2));

        public double AssignValues(double x, double y, double z)
        {
            Sphere searchSphere = new Sphere(new Point3d(x, y, z), range);
            double biggestCharge = 0;
            rTree.Search(searchSphere, (obj, arg) =>
            {
                Point3d p = points[arg.Id];
                double charge = linear
                    ? (double)values[arg.Id] / Distance(p.X, p.Y, p.Z, x, y, z)
                    : (double)values[arg.Id] / DistanceSq(p.X, p.Y, p.Z, x, y, z);

                if (!bulge)
                {
                    if (charge > biggestCharge) biggestCharge = charge;
                }
                else
                {
                    biggestCharge += charge;
                }
            });
            return biggestCharge;
        }

        public void CreateEnvironment(double cellSize, out Box box, out int xDim, out int yDim, out int zDim)
        {
            box = new Box(Plane.WorldXY, points);
            box.Inflate(range);
            box.RepositionBasePlane(box.Center);

            xDim = (int)Math.Floor(box.X.Length / cellSize);
            yDim = (int)Math.Floor(box.Y.Length / cellSize);
            zDim = (int)Math.Floor(box.Z.Length / cellSize);

            box.X = new Interval(-(xDim * cellSize) / 2, (xDim * cellSize) / 2);
            box.Y = new Interval(-(yDim * cellSize) / 2, (yDim * cellSize) / 2);
            box.Z = new Interval(-(zDim * cellSize) / 2, (zDim * cellSize) / 2);
        }

        public void CreateEnvironment(double cellSize, Box boxIn, out Box box, out int xDim, out int yDim, out int zDim)
        {
            box = boxIn;
            box.RepositionBasePlane(box.Center);

            xDim = (int)Math.Floor(box.X.Length / cellSize);
            yDim = (int)Math.Floor(box.Y.Length / cellSize);
            zDim = (int)Math.Floor(box.Z.Length / cellSize);

            box.X = new Interval(-(xDim * cellSize) / 2, (xDim * cellSize) / 2);
            box.Y = new Interval(-(yDim * cellSize) / 2, (yDim * cellSize) / 2);
            box.Z = new Interval(-(zDim * cellSize) / 2, (zDim * cellSize) / 2);
        }
    }
}
