N = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
R = 52435875175126190479447740508185965837690552500527637822603658699938581184513

class FP:
    def __init__(self, value):
        self.value = value % N

    def __add__(self, other):
        return FP(self.value + other.value)

    def __mul__(self, other):
        if isinstance(other, int):
            return FP(self.value * other)
        # global num_mul
        # num_mul += 1
        return FP(self.value * other.value)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return FP(self.value - other.value)

    def __str__(self):
        # return str(self.value)
        return hex(self.value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        assert isinstance(other, self.__class__)
        return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self.__class__(-self.value)

    def __truediv__(self, other):
        on = other.value if isinstance(other, FP) else other
        assert isinstance(on, (int))
        return FP(self.value * prime_field_inv(on, N))

    @classmethod
    def one(cls):
        return cls(1)

    @classmethod
    def zero(cls):
        return cls(0)

    def __pow__(self, other):
        o = self.__class__.one()
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o

    def inv(self):
        n = N
        a = self.value
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % n, n
        while low > 1:
            r = high // low
            nm, new = hm - lm * r, high - low * r
            lm, low, hm, high = nm, new, lm, low
        return FP(lm % n)


def deg(p):
    d = len(p) - 1
    while p[d] == 0 and d:
        d -= 1
    return d


# Extended euclidean algorithm to find modular inverses for
# integers
def prime_field_inv(a, n):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % n, n
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % n


def poly_rounded_div(a, b):
    dega = deg(a)
    degb = deg(b)
    temp = [x for x in a]
    o = [FP(0) for x in a]
    for i in range(dega - degb, -1, -1):
        o[i] = (o[i] + temp[degb + i] * FP(prime_field_inv(b[degb].value, N)))
        for c in range(degb + 1):
            temp[c + i] = (temp[c + i] - o[c])
    return [x for x in o[:deg(o) + 1]]


class FQ2:
    FIELD_COEFFS = {0: FP(1)}
    DEGREE = 2

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = [c if isinstance(c, FP) else FP(c) for c in coeffs]

    def mul(cls, a, b):
        r = [FP(0) for i in range(0, cls.DEGREE * 2 - 1)]
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = r[i + j] + a.coeffs[i] * b.coeffs[j]

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ2(r[:cls.DEGREE])

    # https://cr.yp.to/papers/m3-20010811-retypeset-20220327.pdf
    def kmul(cls, a, b):
        global red_mul_fq2
        r = [FP(0) for i in range(0, cls.DEGREE * 2 - 1)]

        t = a.coeffs[0] * b.coeffs[0]
        u = a.coeffs[1] * b.coeffs[1]

        r[0] = t
        r[1] = (a.coeffs[0] + a.coeffs[1]) * (b.coeffs[0] + b.coeffs[1]) - u - t
        r[2] = u

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                # red_mul_fq2 = red_mul_fq2 + 1
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ2(r[:cls.DEGREE])

    def inv(self):
        FQ2_modulus_coeffs = [1, 0]
        FQ2_modulus_coeffs = [FP(c) for c in FQ2_modulus_coeffs]

        p12 = self.coeffs
        degree = 2
        lm, hm = [FP(1)] + [FP(0)] * degree, [FP(0)] * (degree + 1)
        low, high = p12 + [FP(0)], FQ2_modulus_coeffs + [FP(1)]
        while deg(low):
            r = poly_rounded_div(high, low)
            r += [FP(0)] * (degree + 1 - len(r))
            nm = [x for x in hm]
            new = [x for x in high]
            # assert len(lm) == len(hm) == len(low) == len(high) == len(nm) == len(new) == self.degree + 1
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    nm[i + j] -= lm[i] * r[j]
                    new[i + j] -= low[i] * r[j]
            # nm = [x % N for x in nm]
            # new = [x % N for x in new]
            lm, low, hm, high = nm, new, lm, low

        return FQ2([c / low[0] for c in lm[:degree]])
        # return self.__class__(lm[:degree]) / low[0]

    def __mul__(self, other):
        if isinstance(other, (int)):
            return self.__class__([c * other for c in self.coeffs])
        if isinstance(other, (FP)):
            return self.__class__([c * other.value for c in self.coeffs])
        return self.kmul(self, other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return FQ2([self.coeffs[i] + other.coeffs[i] for i in range(self.DEGREE)])

    def __sub__(self, other):
        assert isinstance(other, FQ2)
        return FQ2([(x - y) for x, y in zip(self.coeffs, other.coeffs)])

    def __str__(self):
        return str([c for c in self.coeffs])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, self.__class__)

        return self.coeffs[0] == other.coeffs[0] and self.coeffs[1] == other.coeffs[1]

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self.__class__([-c for c in self.coeffs])

    @classmethod
    def one(cls):
        return cls([1, 0])

    @classmethod
    def zero(cls):
        return cls([0, 0])

    def deg(self):
        return 1 if self.coeffs[1] != 0 else 0

    def __pow__(self, other):
        o = self.__class__.one()
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o


class FP:
    def __init__(self, value):
        self.value = value % N

    def __add__(self, other):
        return FP(self.value + other.value)

    def __mul__(self, other):
        if isinstance(other, int):
            return FP(self.value * other)
        # global num_mul
        # num_mul += 1
        return FP(self.value * other.value)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return FP(self.value - other.value)

    def __str__(self):
        # return str(self.value)
        return hex(self.value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        assert isinstance(other, self.__class__)
        return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self.__class__(-self.value)

    def __truediv__(self, other):
        on = other.value if isinstance(other, FP) else other
        assert isinstance(on, (int))
        return FP(self.value * prime_field_inv(on, N))

    @classmethod
    def one(cls):
        return cls(1)

    @classmethod
    def zero(cls):
        return cls(0)

    def __pow__(self, other):
        o = self.__class__.one()
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o

    def inv(self):
        n = N
        a = self.value
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % n, n
        while low > 1:
            r = high // low
            nm, new = hm - lm * r, high - low * r
            lm, low, hm, high = nm, new, lm, low
        return FP(lm % n)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "x: " + str(self.x) + "\n" + "y: " + str(self.y) + "\n" + "z: " + str(self.z) + "\n"

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return self.__class__(self.x, -self.y, self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def to_affine(self):
        z_inv = self.z.inv()
        return Point(self.x * z_inv, self.y * z_inv, self.y.__class__.one())


def double(pt):
    x, y, z = pt.x, pt.y, pt.z
    W = x * x
    W = W * 3
    S = y * z
    B = x * y * S
    H = W * W - 8 * B
    S_squared = S * S
    newx = 2 * H * S
    newy = W * (4 * B - H) - 8 * y * y * S_squared
    newz = 8 * S * S_squared
    return Point(newx, newy, newz)


def add(p1, p2):
    one, zero = p1.x.__class__.one(), p1.x.__class__.zero()
    if p1.z == zero or p2.z == zero:
        return p1 if p2.z == zero else p2
    x1, y1, z1 = p1.x, p1.y, p1.z
    x2, y2, z2 = p2.x, p2.y, p2.z
    U1 = y2 * z1
    U2 = y1 * z2
    V1 = x2 * z1
    V2 = x1 * z2
    if V1 == V2 and U1 == U2:
        return double(p1)
    elif V1 == V2:
        return Point(one, one, zero)
    U = U1 - U2
    V = V1 - V2
    V_squared = V * V
    V_squared_times_V2 = V_squared * V2
    V_cubed = V * V_squared
    W = z1 * z2
    A = U * U * W - V_cubed - 2 * V_squared_times_V2
    newx = V * A
    newy = U * (V_squared_times_V2 - A) - V_cubed * U2
    newz = V_cubed * W
    return Point(newx, newy, newz)


def multiply(pt, n):
    if n == 0:
        return Point(pt[0].__class__.one(), pt[0].__class__.one(), pt[0].__class__.zero())
    elif n == 1:
        return pt
    elif not n % 2:
        return multiply(double(pt), n // 2)
    else:
        return add(multiply(double(pt), int(n // 2)), pt)


def is_on_curve(p: Point):
    return (p.y ** 2) == (p.x ** 3) + FP(4)


def is_on_twisted_curve(p: Point):
    return (p.y ** 2) == (p.x ** 3) + 4 * FQ2([1,1])


assert (N % 4 == 3)
assert ((N + 1) % 4 == 0)

def sqrt(v: FP):
    if v ** ((N - 1) // 2) == 1:
        return v ** ((N + 1) // 4)
    else:
        return False

assert(sqrt(FP(N - 1)) == False)
assert(sqrt(FP(N // 2)) * sqrt(FP(N // 2)) == FP(N // 2))


def sqrt_fq2(a: FQ2):
    a1 = (a ** ((N - 3) // 4))
    alpha = a1 * (a1 * a)
    a0 = (alpha ** N) * alpha
    if a0 == FQ2([-1, 0]):
        return False

    x0 = a1 * a
    if alpha == FQ2([-1, 0]):
        return FQ2([0, 1]) * x0
    else:
        b = (FQ2.one() + alpha) ** ((N - 1) // 2)
        return b * x0


def gen_twisted_curve_point(x: FQ2):
    y2 = x ** 3 + 4 * FQ2([1,1])
    y = sqrt_fq2(y2)
    return Point(x, y, FQ2.one()) if y else Point(FQ2.zero(), FQ2.zero(), FQ2.zero())


def gen_curve_point(x: FP):
    y2 = x ** 3 + FP(4)
    y = sqrt(y2)
    return Point(x, y, FP.one()) if y else Point(FP.zero(), FP.zero(), FP.zero())