using System;

namespace GHGPUPlugin.Chromodoris.Topology
{
    /// <summary>
    /// 8-node linear hexahedron stiffness (24×24) for isotropic elasticity, 2×2×2 Gauss integration.
    /// Brick aligned with axes: local node 0 at origin, 1 at +dx, 2 at (+dx,+dy), … (standard Z-order bottom + top).
    /// </summary>
    internal static class Hex8BrickKe
    {
        /// <summary>Build symmetric Ke for E = 1 (unit Young's modulus). Scale moduli in the operator.</summary>
        public static void BuildUnitKe(double nu, double dx, double dy, double dz, double[,] Ke)
        {
            double lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            double mu = 1.0 / (2.0 * (1.0 + nu));

            var D = new double[6, 6];
            D[0, 0] = lam + 2 * mu; D[0, 1] = lam; D[0, 2] = lam;
            D[1, 0] = lam; D[1, 1] = lam + 2 * mu; D[1, 2] = lam;
            D[2, 0] = lam; D[2, 1] = lam; D[2, 2] = lam + 2 * mu;
            D[3, 3] = mu; D[4, 4] = mu; D[5, 5] = mu;

            for (int r = 0; r < 24; r++)
                for (int c = 0; c < 24; c++)
                    Ke[r, c] = 0;

            double g = 1.0 / Math.Sqrt(3.0);
            double[] gp = { -g, g };
            double w = 1.0;

            // Corner offsets in physical space from node 0 of element
            var xn = new double[8, 3];
            xn[0, 0] = 0; xn[0, 1] = 0; xn[0, 2] = 0;
            xn[1, 0] = dx; xn[1, 1] = 0; xn[1, 2] = 0;
            xn[2, 0] = dx; xn[2, 1] = dy; xn[2, 2] = 0;
            xn[3, 0] = 0; xn[3, 1] = dy; xn[3, 2] = 0;
            xn[4, 0] = 0; xn[4, 1] = 0; xn[4, 2] = dz;
            xn[5, 0] = dx; xn[5, 1] = 0; xn[5, 2] = dz;
            xn[6, 0] = dx; xn[6, 1] = dy; xn[6, 2] = dz;
            xn[7, 0] = 0; xn[7, 1] = dy; xn[7, 2] = dz;

            var dN = new double[3, 8];
            var J = new double[3, 3];
            var invJ = new double[3, 3];
            var dNx = new double[3, 8];
            var B = new double[6, 24];
            var DB = new double[6, 24];

            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++)
                    {
                        double xi = gp[a];
                        double eta = gp[b];
                        double zt = gp[c];
                        ShapeDerivs(xi, eta, zt, dN);

                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                            {
                                J[ii, jj] = 0;
                                for (int n = 0; n < 8; n++)
                                    J[ii, jj] += dN[ii, n] * xn[n, jj];
                            }

                        double detJ = J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
                                    - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
                                    + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]);
                        if (detJ <= 0) throw new InvalidOperationException("Hex Jacobian non-positive.");

                        Invert3(J, invJ);

                        for (int ii = 0; ii < 3; ii++)
                            for (int n = 0; n < 8; n++)
                            {
                                dNx[ii, n] = 0;
                                for (int jj = 0; jj < 3; jj++)
                                    dNx[ii, n] += invJ[ii, jj] * dN[jj, n];
                            }

                        for (int col = 0; col < 24; col++)
                            for (int row = 0; row < 6; row++)
                                B[row, col] = 0;

                        for (int n = 0; n < 8; n++)
                        {
                            int b0 = n * 3;
                            B[0, b0 + 0] = dNx[0, n];
                            B[1, b0 + 1] = dNx[1, n];
                            B[2, b0 + 2] = dNx[2, n];
                            B[3, b0 + 0] = dNx[1, n];
                            B[3, b0 + 1] = dNx[0, n];
                            B[4, b0 + 1] = dNx[2, n];
                            B[4, b0 + 2] = dNx[1, n];
                            B[5, b0 + 0] = dNx[2, n];
                            B[5, b0 + 2] = dNx[0, n];
                        }

                        for (int i = 0; i < 6; i++)
                            for (int j = 0; j < 24; j++)
                            {
                                DB[i, j] = 0;
                                for (int k = 0; k < 6; k++)
                                    DB[i, j] += D[i, k] * B[k, j];
                            }

                        double coeff = w * w * w * Math.Abs(detJ);
                        for (int i = 0; i < 24; i++)
                            for (int j = 0; j < 24; j++)
                            {
                                double s = 0;
                                for (int k = 0; k < 6; k++)
                                    s += B[k, i] * DB[k, j];
                                Ke[i, j] += coeff * s;
                            }
                    }

            for (int i = 0; i < 24; i++)
                for (int j = i + 1; j < 24; j++)
                    Ke[j, i] = Ke[i, j];
        }

        private static void ShapeDerivs(double xi, double eta, double zt, double[,] dN)
        {
            double xm = 1 - xi, xp = 1 + xi;
            double em = 1 - eta, ep = 1 + eta;
            double zm = 1 - zt, zp = 1 + zt;
            double c = 0.125;

            dN[0, 0] = -c * em * zm; dN[1, 0] = -c * xm * zm; dN[2, 0] = -c * xm * em;
            dN[0, 1] = c * em * zm; dN[1, 1] = -c * xp * zm; dN[2, 1] = -c * xp * em;
            dN[0, 2] = c * ep * zm; dN[1, 2] = c * xp * zm; dN[2, 2] = -c * xp * ep;
            dN[0, 3] = -c * ep * zm; dN[1, 3] = c * xm * zm; dN[2, 3] = -c * xm * ep;
            dN[0, 4] = -c * em * zp; dN[1, 4] = -c * xm * zp; dN[2, 4] = c * xm * em;
            dN[0, 5] = c * em * zp; dN[1, 5] = -c * xp * zp; dN[2, 5] = c * xp * em;
            dN[0, 6] = c * ep * zp; dN[1, 6] = c * xp * zp; dN[2, 6] = c * xp * ep;
            dN[0, 7] = -c * ep * zp; dN[1, 7] = c * xm * zp; dN[2, 7] = c * xm * ep;
        }

        private static void Invert3(double[,] J, double[,] inv)
        {
            double a = J[0, 0], b = J[0, 1], c = J[0, 2];
            double d = J[1, 0], e = J[1, 1], f = J[1, 2];
            double g = J[2, 0], h = J[2, 1], i = J[2, 2];
            double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            double id = 1.0 / det;
            inv[0, 0] = id * (e * i - f * h);
            inv[0, 1] = id * (c * h - b * i);
            inv[0, 2] = id * (b * f - c * e);
            inv[1, 0] = id * (f * g - d * i);
            inv[1, 1] = id * (a * i - c * g);
            inv[1, 2] = id * (c * d - a * f);
            inv[2, 0] = id * (d * h - e * g);
            inv[2, 1] = id * (b * g - a * h);
            inv[2, 2] = id * (a * e - b * d);
        }

        /// <summary>ce = u^T K0 u (twice strain energy at unit E).</summary>
        public static double ElementEnergy(double[,] K0, double[] ue)
        {
            double s = 0;
            for (int i = 0; i < 24; i++)
            {
                double t = 0;
                for (int j = 0; j < 24; j++)
                    t += K0[i, j] * ue[j];
                s += ue[i] * t;
            }
            return s;
        }

        public static void MultiplyKe(double[,] K0, double[] ue, double[] fe)
        {
            for (int i = 0; i < 24; i++)
            {
                double t = 0;
                for (int j = 0; j < 24; j++)
                    t += K0[i, j] * ue[j];
                fe[i] = t;
            }
        }
    }
}
