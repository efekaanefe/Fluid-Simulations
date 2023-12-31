import numpy as np

class Fluid:
    def __init__(self, density, numX, numY, h):
        self.density = density
        self.numX = numX + 2
        self.numY = numY + 2
        self.numCells = self.numX * self.numY
        self.h = h
        self.u = np.zeros(self.numCells, dtype=np.float32)
        self.v = np.zeros(self.numCells, dtype=np.float32)
        self.newU = np.zeros(self.numCells, dtype=np.float32)
        self.newV = np.zeros(self.numCells, dtype=np.float32)
        self.p = np.zeros(self.numCells, dtype=np.float32)
        self.s = np.zeros(self.numCells, dtype=np.float32)
        self.m = np.ones(self.numCells, dtype=np.float32)
        self.newM = np.zeros(self.numCells, dtype=np.float32)

    def integrate(self, dt, gravity):
        n = self.numY
        for i in range(1, self.numX):
            for j in range(1, self.numY - 1):
                if self.s[i * n + j] != 0.0 and self.s[i * n + j - 1] != 0.0:
                    self.v[i * n + j] += gravity * dt

    def solveIncompressibility(self, numIters, dt):
        n = self.numY
        cp = self.density * self.h / dt

        for iter in range(numIters):
            for i in range(1, self.numX - 1):
                for j in range(1, self.numY - 1):
                    if self.s[i * n + j] == 0.0:
                        continue

                    s = self.s[i * n + j]
                    sx0 = self.s[(i - 1) * n + j]
                    sx1 = self.s[(i + 1) * n + j]
                    sy0 = self.s[i * n + j - 1]
                    sy1 = self.s[i * n + j + 1]
                    s_sum = sx0 + sx1 + sy0 + sy1
                    if s_sum == 0.0:
                        continue

                    div = (
                        self.u[(i + 1) * n + j] - self.u[i * n + j]
                        + self.v[i * n + j + 1] - self.v[i * n + j]
                    )
                    p = -div / s_sum
                    p *= scene.overRelaxation
                    self.p[i * n + j] += cp * p

                    self.u[i * n + j] -= sx0 * p
                    self.u[(i + 1) * n + j] += sx1 * p
                    self.v[i * n + j] -= sy0 * p
                    self.v[i * n + j + 1] += sy1 * p

    def extrapolate(self):
        n = self.numY
        for i in range(self.numX):
            self.u[i * n] = self.u[i * n + 1]
            self.u[i * n + self.numY - 1] = self.u[i * n + self.numY - 2]
        for j in range(self.numY):
            self.v[j] = self.v[n + j]
            self.v[(self.numX - 1) * n + j] = self.v[(self.numX - 2) * n + j]

    def sampleField(self, x, y, field):
        n = self.numY
        h = self.h
        h1 = 1.0 / h
        h2 = 0.5 * h

        x = max(min(x, self.numX * h), h)
        y = max(min(y, self.numY * h), h)

        dx = 0.0
        dy = 0.0

        if field == "U_FIELD":
            f = self.u
            dy = h2
        elif field == "V_FIELD":
            f = self.v
            dx = h2
        elif field == "S_FIELD":
            f = self.m
            dx = h2
            dy = h2

        x0 = min(int((x - dx) * h1), self.numX - 1)
        tx = ((x - dx) - x0 * h) * h1
        x1 = min(x0 + 1, self.numX - 1)

        y0 = min(int((y - dy) * h1), self.numY - 1)
        ty = ((y - dy) - y0 * h) * h1
        y1 = min(y0 + 1, self.numY - 1)

        sx = 1.0 - tx
        sy = 1.0 - ty

        val = (
            sx * sy * f[x0 * n + y0]
            + tx * sy * f[x1 * n + y0]
            + tx * ty * f[x1 * n + y1]
            + sx * ty * f[x0 * n + y1]
        )

        return val

    def avgU(self, i, j):
        n = self.numY
        u = (
            self.u[i * n + j - 1]
            + self.u[i * n + j]
            + self.u[(i + 1) * n + j - 1]
            + self.u[(i + 1) * n + j]
        ) * 0.25
        return u

    def avgV(self, i, j):
        n = self.numY
        v = (
            self.v[(i - 1) * n + j]
            + self.v[i * n + j]
            + self.v[(i - 1) * n + j + 1]
            + self.v[i * n + j + 1]
        ) * 0.25
        return v

    def advectVel(self, dt):
        self.newU[:] = self.u
        self.newV[:] = self.v
        n = self.numY
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX):
            for j in range(1, self.numY):
                # u component
                if (
                    self.s[i * n + j] != 0.0
                    and self.s[(i - 1) * n + j] != 0.0
                    and j < self.numY - 1
                ):
                    x = i * h
                    y = j * h + h2
                    u = self.u[i * n + j]
                    v = self.avgV(i, j)
                    x = x - dt * u
                    y = y - dt * v
                    u = self.sampleField(x, y, "U_FIELD")
                    self.newU[i * n + j] = u

                # v component
                if (
                    self.s[i * n + j] != 0.0
                    and self.s[i * n + j - 1] != 0.0
                    and i < self.numX - 1
                ):
                    x = i * h + h2
                    y = j * h
                    u = self.avgU(i, j)
                    v = self.v[i * n + j]
                    x = x - dt * u
                    y = y - dt * v
                    v = self.sampleField(x, y, "V_FIELD")
                    self.newV[i * n + j] = v

        self.u[:] = self.newU
        self.v[:] = self.newV

    def advectSmoke(self, dt):
        self.newM[:] = self.m
        n = self.numY
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                if self.s[i * n + j] != 0.0:
                    u = (self.u[i * n + j] + self.u[(i + 1) * n + j]) * 0.5
                    v = (self.v[i * n + j] + self.v[i * n + j + 1]) * 0.5
                    x = i * h + h2 - dt * u
                    y = j * h + h2 - dt * v
                    self.newM[i * n + j] = self.sampleField(x, y, "S_FIELD")

        self.m[:] = self.newM

# Example usage
density = 1.0
numX = 10
numY = 10
h = 1.0
gravity = 9.81
dt = 0.1
numIters = 100

scene = Fluid(density, numX, numY, h)
scene.integrate(dt, gravity)
scene.solveIncompressibility(numIters, dt)
scene.extrapolate()
scene.advectVel(dt)
scene.advectSmoke(dt)

