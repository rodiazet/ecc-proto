# degree = 2
#
# def mul(a_coeffs, b_coeffs, m_coeffs):
#     r = [0 for i in range(0, degree * 2 - 1)]
#     for j in range(degree):
#         for i in range(degree):
#             r[i + j] = r[i + j] + a_coeffs[i] * b_coeffs[j]
#
#     print(r);
#
#     for i in range(0, len(r) - len(m_coeffs) + 1):
#         T = r[i]
#         for j in range(len(m_coeffs)):
#             r[i + j] = r[i + j] - T * m_coeffs[j]
#             print(r)
#
#
# def reminder(a_coeffs, m_coeffs):
#     r = [i for i in a_coeffs]
#
#     for i in range(0, len(r) - len(m_coeffs) + 1):
#         T = r[i]
#         for j in range(len(m_coeffs)):
#             r[i + j] = r[i + j] - T * m_coeffs[j]
#             print(r)
#     return r[len(a_coeffs) - len(m_coeffs) + 1:]
#
# print(reminder([3,5,6,7,2], [1, 2, 1]))
#
# mul([4, 3], [5, 2], [1, 2, 1])
#
#
# print ("---------------")
# ##################################
# def reminder(a_coeffs, m_coeffs):
#     r = [i for i in a_coeffs]
#
#     for i in range(0, len(r) - len(m_coeffs)):
#         T = r[i]
#         r[i] = 0
#         for j in range(len(m_coeffs)):
#             r[i + 1 + j] = r[i + 1 + j] - T * m_coeffs[j]
#             print(r)
#     return r[len(a_coeffs) - len(m_coeffs):]
#
# print(reminder([3,5,6,7,2], [2, 1]))
#
#
# print ("---------------")
# ##################################
# def reminder(a_coeffs, m_coeffs, degree):
#     r = [i for i in a_coeffs]
#
#     for i in range(0, len(r) - degree):
#         for j in m_coeffs.keys():
#             r[i + 1 + j] = r[i + 1 + j] - r[i] * m_coeffs[j]
#             print(r)
#     return r[len(a_coeffs) - degree:]
#
# print(reminder([3,5,6,7,2], {0: 2, 1: 1}, 2))
#
# print(reminder([3,5,6,7,2], {0: 2, 2: 1}, 3))
# print(reminder([3,5,6,7,2], {2: 1, 0: 2}, 3))
#
#
# print ("---------------")
# ##################################
# def reminder(a_coeffs, m_coeffs, degree):
#     r = [i for i in a_coeffs]
#
#     for i in range(len(r) - 1, degree - 1, -1):
#         d = i - degree
#         for j in m_coeffs.keys():
#             r[d + j] = r[d + j] - r[i] * m_coeffs[j]
#     return r
#
# print(reminder([2,7,6,5,3,0,0,0], {0: 2, 1: 1}, 2))
#
# print(reminder([2,7,6,5,3], {0: 2, 2: 1}, 3))
# print(reminder([2,7,6,5,3], {2: 1, 0: 2}, 3))
import random

NAF_STR = "001001001010001010001100001001100001001001010001001100001100001001001010001100001001001001100001001010001100001001010001001001001001100001001100001010001100001001001100001100001001001010001010010"
NAF = [NAF_STR[i:i+3] for i in range(0, len(NAF_STR), 3)]

NAF_REP = []

for i in range(len(NAF) - 1, -1, -1):
    if NAF[i] == "010":
        NAF_REP += [1]
    elif NAF[i] == "100":
        NAF_REP += [-1]
    else:
        NAF_REP += [0]

assert(len(NAF_STR) % 3 == 0)
assert(len(NAF_REP) == len(NAF_STR) // 3)

NAF_REP.reverse()

print(len(NAF_REP))
print(NAF_REP)

print(sum([e * 2**i for i, e in enumerate(NAF_REP)]))

s = ""
for i in NAF_REP:
    if i == 0:
        s = s + "00"
    elif i == 1:
        s = s + "01"
    else:
        s = s + "10"

print (s)

N = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
R = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
T = 4965661367192848881
X = 4965661367192848881

print(6 * (T ** 2) + 1)


# num_mul = 0
# red_mul_fq2 = 0
# red_mul_fq6 = 0


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


class FQ6:
    FIELD_COEFFS = {0: FQ2([FP(-9), FP(-1)])}
    DEGREE = 3

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = coeffs

    def mul(cls, a, b):
        r = [FQ2([0, 0]) for i in range(0, cls.DEGREE * 2 - 1)]
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = r[i + j] + a.coeffs[i] * b.coeffs[j]

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ6(r[:cls.DEGREE])

    def kmul(cls, a, b):
        r = [FQ2([0, 0]) for i in range(0, cls.DEGREE * 2 - 1)]

        s = a.coeffs[0] * b.coeffs[0]
        t = a.coeffs[1] * b.coeffs[1]
        v = a.coeffs[2] * b.coeffs[2]

        r[0] = s
        r[1] = (a.coeffs[0] + a.coeffs[1]) * (b.coeffs[0] + b.coeffs[1]) - s - t
        r[2] = t + (a.coeffs[0] + a.coeffs[2]) * (b.coeffs[0] + b.coeffs[2]) - s - v
        r[3] = (a.coeffs[1] + a.coeffs[2]) * (b.coeffs[1] + b.coeffs[2]) - t - v
        r[4] = v

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ6(r[:cls.DEGREE])

    def __mul__(self, other):
        if isinstance(other, (int)):
            return self.__class__([c * other for c in self.coeffs])
        return self.kmul(self, other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return FQ6([self.coeffs[i] + other.coeffs[i] for i in range(self.DEGREE)])

    def __sub__(self, other):
        assert isinstance(other, FQ6)
        return FQ6([(x - y) for x, y in zip(self.coeffs, other.coeffs)])

    def __str__(self):
        return str([c for c in self.coeffs])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, self.__class__)

        return self.coeffs[0] == other.coeffs[0] and self.coeffs[1] == other.coeffs[1] and self.coeffs[2] == \
            other.coeffs[2]

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self.__class__([-c for c in self.coeffs])

    @classmethod
    def one(cls):
        return cls([FQ2.one()] + [FQ2.zero()] * (cls.DEGREE - 1))

    @classmethod
    def zero(cls):
        return cls([FQ2.zero()] * (cls.DEGREE))


class FQ6N:
    FIELD_COEFFS = {3: FP(-18), 0: FP(82)}
    DEGREE = 6

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = coeffs

    def mul(cls, a, b):
        r = [FP(0) for i in range(0, cls.DEGREE * 2 - 1)]
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = (r[i + j] + a.coeffs[i] * b.coeffs[j])

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ6N(r[:cls.DEGREE])

    def __mul__(self, other):
        return self.mul(self, other)

    def __str__(self):
        return str([c for c in self.coeffs])

    def __repr__(self):
        return self.__str__()


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


# Utility methods for polynomial math
def deg(p):
    d = len(p) - 1
    while p[d] == 0 and d:
        d -= 1
    return d


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


class FQ12:
    FIELD_COEFFS = {0: FQ6([FQ2([0, 0]), FQ2([-1, 0]), FQ2([0, 0])])}
    DEGREE = 2

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = coeffs

    def mul(cls, a, b):
        r = [FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])]) for i in range(0, cls.DEGREE * 2 - 1)]
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = r[i + j] + a.coeffs[i] * b.coeffs[j]

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ12(r[:cls.DEGREE])

    def kmul(cls, a, b):
        r = [FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])]) for i in range(0, cls.DEGREE * 2 - 1)]

        t = a.coeffs[0] * b.coeffs[0]
        u = a.coeffs[1] * b.coeffs[1]

        r[0] = t
        r[1] = (a.coeffs[0] + a.coeffs[1]) * (b.coeffs[0] + b.coeffs[1]) - u - t
        r[2] = u

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ12(r[:cls.DEGREE])

    def __pow__(self, other):
        o = self.__class__.one()
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o

    def __mul__(self, other):
        if isinstance(other, (int)):
            return self.__class__([c * other for c in self.coeffs])
        if isinstance(other, FP):
            return self.__class__([c * other.value for c in self.coeffs])
        return self.kmul(self, other)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        assert isinstance(other, FQ12)
        return FQ12([(x - y) for x, y in zip(self.coeffs, other.coeffs)])

    def __add__(self, other):
        assert isinstance(other, FQ12)
        return FQ12([(x + y) for x, y in zip(self.coeffs, other.coeffs)])

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
    def zero(cls):
        return cls([FQ6.zero()] * cls.DEGREE)

    @classmethod
    def one(cls):
        return cls([FQ6.one(), FQ6.zero()])

    def __div__(self, other):
        return self * other.inv()

    # Extended euclidean algorithm used to find the modular inverse
    def inv(self):
        FQ12_modulus_coeffs = [82, 0, 0, 0, 0, 0, -18, 0, 0, 0, 0, 0]
        FQ12_modulus_coeffs = [FP(c) for c in FQ12_modulus_coeffs]

        p12 = self.get_12_degree_poly()
        degree = 12
        lm, hm = [FP(1)] + [FP(0)] * degree, [FP(0)] * (degree + 1)
        low, high = p12 + [FP(0)], FQ12_modulus_coeffs + [FP(1)]
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

        return FQ12.from_12_degree_poly([c / low[0] for c in lm[:degree]])
        # return self.__class__(lm[:degree]) / low[0]

    def get_12_degree_poly(self):
        r = [0] * 12

        for i, c in enumerate(self.coeffs):
            for j, c2 in enumerate(c.coeffs):
                r[i + j * 2] = c2.coeffs[0] - 9 * c2.coeffs[1]
                r[i + j * 2 + 6] = c2.coeffs[1]

        return r

    @classmethod
    def from_12_degree_poly(cls, p):
        assert (len(p) == 12)

        x = FQ6([FQ2([p[0] + 9 * p[6], p[6]]), FQ2([p[2] + 9 * p[8], p[8]]), FQ2([p[4] + 9 * p[10], p[10]])])
        y = FQ6([FQ2([p[1] + 9 * p[7], p[7]]), FQ2([p[3] + 9 * p[9], p[9]]), FQ2([p[5] + 9 * p[11], p[11]])])

        return FQ12([x, y])


class FQ12N:
    FIELD_COEFFS = {6: FP(-18), 0: FP(82)}
    DEGREE = 12

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = coeffs

    def mul(cls, a, b):
        r = [FP(0) for i in range(0, cls.DEGREE * 2 - 1)]
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = (r[i + j] + a.coeffs[i] * b.coeffs[j])

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ12N(r[:cls.DEGREE])

    def __mul__(self, other):
        return self.mul(self, other)

    def __str__(self):
        return str([c for c in self.coeffs])

    def __repr__(self):
        return self.__str__()


class FQ12_6:
    FIELD_COEFFS = {0: FQ2([FP(0), FP(-1)])}
    DEGREE = 6

    def __init__(self, coeffs):
        assert len(coeffs) == self.DEGREE
        self.coeffs = coeffs

    def mul(cls, a, b):
        r = [FQ2([0, 0])] * (cls.DEGREE * 2 - 1)
        for j in range(cls.DEGREE):
            for i in range(cls.DEGREE):
                r[i + j] = r[i + j] + a.coeffs[i] * b.coeffs[j]

        # print(r)

        for i in range(len(r) - 1, cls.DEGREE - 1, -1):
            d = i - cls.DEGREE
            for j in cls.FIELD_COEFFS.keys():
                r[d + j] = r[d + j] - r[i] * cls.FIELD_COEFFS[j]

        return FQ12_6(r[:cls.DEGREE])

    def __pow__(self, other):
        o = self.__class__.one()
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o

    def __mul__(self, other):
        return self.mul(self, other)

    def __str__(self):
        return str([c for c in self.coeffs])

    def __repr__(self):
        return self.__str__()

    @classmethod
    def one(cls):
        return FQ12_6([FQ2.one(), FQ2.zero(), FQ2.zero(), FQ2.zero(), FQ2.zero(), FQ2.zero()])


# x = FQ6([FQ2([FP(1), FP(2)]), FQ2([FP(3), FP(4)]), FQ2([FP(5), FP(6)])])
# print(x)
# y = FQ6([FQ2([FP(7), FP(8)]), FQ2([FP(9), FP(10)]), FQ2([FP(11), FP(12)])])
# print(y)
# fq = x * y
# print(str(fq))
# # print(num_mul)
# # print(red_mul_fq2)
# # print(red_mul_fq6)
#
# # [FQ2([c.coeffs[0] + FP(9) * c.coeffs[1], c.coeffs[1]]) for c in x.coeffs]
#
# x1 = FQ6N([FP(1) - FP(9) * FP(2), FP(3) - FP(9) * FP(4), FP(5) - FP(9) * FP(6), FP(2), FP(4), FP(6)])
# print(x1)
# y1 = FQ6N([FP(7) - FP(9) * FP(8), FP(9) - FP(9) * FP(10), FP(11) - FP(9) * FP(12), FP(8), FP(10), FP(12)])
# print(y1)
#
# # num_mul = 0
# # red_mul_fq2 = 0
# # red_mul_fq6 = 0
# fq2 = x1 * y1
# print(str(fq2))
# # print(num_mul)
# # print(red_mul_fq2)
# # print(red_mul_fq6)
#
# fq2 = [fq2.coeffs[i] + FP(9) * fq2.coeffs[i + 3] for i in range(3)] + fq2.coeffs[3:]
# print(str(fq2))
#
# x = FQ12([FQ6([FQ2([FP(1), FP(2)]), FQ2([FP(3), FP(4)]), FQ2([FP(5), FP(6)])]),
#           FQ6([FQ2([FP(7), FP(8)]), FQ2([FP(9), FP(10)]), FQ2([FP(11), FP(12)])])])
# print(x)


def conjugate(f):
    return f.__class__([f.coeffs[0], -f.coeffs[1]])


l = [FQ2([FP(9), FP(1)]) ** (i * (N - 1) // 6) for i in range(1, 6)]

l2 = [l[i] * conjugate(l[i]) for i in range(5)]

l3 = [l[i] * l2[i] for i in range(5)]

print(l)
print(l2)
print(l3)


def fp2_pow_N(f: FQ2):
    r = conjugate(f)

    return r


def fp2_pow_N2(f: FQ2):
    return f


def fp2_pow_N3(f: FQ2):
    r = conjugate(f)

    return r


def fp12_pow_N(f: FQ12):
    r = FQ12([FQ6([conjugate(f.coeffs[0].coeffs[0]),
                   conjugate(f.coeffs[0].coeffs[1]) * l[1],
                   conjugate(f.coeffs[0].coeffs[2]) * l[3]]),
              FQ6([conjugate(f.coeffs[1].coeffs[0]) * l[0],
                   conjugate(f.coeffs[1].coeffs[1]) * l[2],
                   conjugate(f.coeffs[1].coeffs[2]) * l[4]])])

    return r


def fp12_pow_N2(f: FQ12):
    r = FQ12([FQ6([f.coeffs[0].coeffs[0],
                   f.coeffs[0].coeffs[1] * l2[1],
                   f.coeffs[0].coeffs[2] * l2[3]]),
              FQ6([f.coeffs[1].coeffs[0] * l2[0],
                   f.coeffs[1].coeffs[1] * l2[2],
                   f.coeffs[1].coeffs[2] * l2[4]])])

    return r


def fp12_pow_N3(f: FQ12):
    r = FQ12([FQ6([conjugate(f.coeffs[0].coeffs[0]),
                   conjugate(f.coeffs[0].coeffs[1]) * l3[1],
                   conjugate(f.coeffs[0].coeffs[2]) * l3[3]]),
              FQ6([conjugate(f.coeffs[1].coeffs[0]) * l3[0],
                   conjugate(f.coeffs[1].coeffs[1]) * l3[2],
                   conjugate(f.coeffs[1].coeffs[2]) * l3[4]])])

    return r


g = [FQ2([FP(9), FP(1)]) ** (i * (N - 1) // 6) for i in range(1, 6)]

g2 = [l[i] * conjugate(l[i]) for i in range(5)]

y = FQ12([FQ6([FQ2([FP(13), FP(14)]), FQ2([FP(15), FP(16)]), FQ2([FP(17), FP(18)])]),
          FQ6([FQ2([FP(19), FP(20)]), FQ2([FP(21), FP(22)]), FQ2([FP(23), FP(24)])])])

y = FQ12([FQ6([FQ2([FP(13), FP(14)]), FQ2([FP(17), FP(18)]), FQ2([FP(21), FP(22)])]),
          FQ6([FQ2([FP(15), FP(16)]), FQ2([FP(19), FP(20)]), FQ2([FP(23), FP(24)])])])

p12 = fp12_pow_N(y)

assert y ** N == p12

p122 = fp12_pow_N2(y)

assert y ** (N ** 2) == p122

p123 = fp12_pow_N3(y)

assert y ** (N ** 3) == p123


# print((FQ2([FP(13), FP(14)]) ** (N * N * N)))

#
# # num_mul = 0
# # red_mul_fq2 = 0
# # red_mul_fq6 = 0
# fq = x * y
# print(str(fq))
# # print(num_mul)
# # print(red_mul_fq2)
# # print(red_mul_fq6)
#
# x1 = FQ12N([
#     FP(1) - FP(9) * FP(2),
#     FP(7) - FP(9) * FP(8),
#     FP(3) - FP(9) * FP(4),
#     FP(9) - FP(9) * FP(10),
#     FP(5) - FP(9) * FP(6),
#     FP(11) - FP(9) * FP(12),
#     FP(2), FP(8), FP(4),
#     FP(10), FP(6), FP(12)])
# print(x1)
# y1 = FQ12N([
#     FP(13) - FP(9) * FP(14),
#     FP(19) - FP(9) * FP(20),
#     FP(15) - FP(9) * FP(16),
#     FP(21) - FP(9) * FP(22),
#     FP(17) - FP(9) * FP(18),
#     FP(23) - FP(9) * FP(24),
#     FP(14), FP(20), FP(16),
#     FP(22), FP(18), FP(24)])
# print(y1)
#
# # num_mul = 0
# # red_mul_fq2 = 0
# # red_mul_fq6 = 0
#
# fq2 = x1 * y1
# print(str(fq2))
#
# # print(num_mul)
# # print(red_mul_fq2)
# # print(red_mul_fq6)
#
# fq2 = [fq2.coeffs[i] + FP(9) * fq2.coeffs[i + 6] for i in range(6)] + fq2.coeffs[6:]
# print(str(fq2))


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


w = FQ12([FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([1, 0]), FQ2([0, 0]), FQ2([0, 0])])])
#
# w2 = w * w
#
w2 = FQ12([FQ6([FQ2([0, 0]), FQ2([1, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])
#
w3 = FQ12([FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([1, 0]), FQ2([0, 0])])])

w3 = w * w * w
w3 = w2 * w
print(w3 * w3)
print(w.inv())


def untwist(pt: Point):
    # x = FQ2([pt.x.coeffs[0] - FP(9) * pt.x.coeffs[1], pt.x.coeffs[1]])
    # y = FQ2([pt.y.coeffs[0] - FP(9) * pt.y.coeffs[1], pt.y.coeffs[1]])
    # z = FQ2([pt.z.coeffs[0] - FP(9) * pt.z.coeffs[1], pt.z.coeffs[1]])
    # #
    # x = FQ12([FQ6([x, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])
    # y = FQ12([FQ6([y, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])
    # z = FQ12([FQ6([z, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])

    x = FQ12([FQ6([pt.x, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])
    y = FQ12([FQ6([pt.y, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])
    z = FQ12([FQ6([pt.z, FQ2([0, 0]), FQ2([0, 0])]), FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])])

    return Point(x * w2, y * w3, z)


def twist(pt: Point):
    w2_inv = w2.inv()
    w3_inv = w3.inv()

    x = pt.x * w2_inv
    y = pt.y * w3_inv

    return Point(x.coeffs[0].coeffs[0], y.coeffs[0].coeffs[0], pt.z.coeffs[0].coeffs[0])


def linear_func(P1: Point, P2: Point, T: Point):
    zero = P1.x.__class__.zero()

    n = P2.y * P1.z - P1.y * P2.z
    d = P2.x * P1.z - P1.x * P2.z

    if d != zero:
        return (n * (T.x * P1.z - T.z * P1.x) - d * (T.y * P1.z - P1.y * T.z)) * (d * T.z * P1.z).inv()
    elif n == zero:
        n = 3 * P1.x * P1.x
        d = 2 * P1.y * P1.z
        return (n * (T.x * P1.z - T.z * P1.x) - d * (T.y * P1.z - P1.y * T.z)) * (d * T.z * P1.z).inv()
    else:
        return T.x * P1.z - P1.x * T.z, T.z * P1.z


P = Point(FQ2([1, 2]), FQ2([3, 4]), FQ2([5, 6]))
print(P)
print(untwist(P))


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

def double_jac(pt):
    X = pt.x
    Y = pt.y
    Z = pt.z
    S = 4 * X * Y ** 2
    M = 3 * X ** 2
    Xp = M ** 2 - 2 * S
    Yp = M * (S - Xp) - 8 * Y ** 4
    Zp = 2 * Y * Z
    return Point(Xp, Yp, Zp)

P = Point(FQ2([0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678,
               0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7]),
          FQ2([0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550,
               0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d]),
          FQ2.one())

r = double_jac(P)

r.x = r.x * (r.z.inv() ** 2)
r.y = r.y * (r.z.inv() ** 3)

print(r)

r = double(P)

r.x = r.x * (r.z.inv())
r.y = r.y * (r.z.inv())

print(r)

Q = Point(r.x, r.y, FQ2.one())


# Elliptic curve addition
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

def add_jac(p1: Point, p2: Point):
    X1 = p1.x
    Y1 = p1.y
    Z1 = p1.z

    X2 = p2.x
    Y2 = p2.y
    Z2 = p2.z

    U1 = X1 * Z2 ** 2
    U2 = X2 * Z1 ** 2
    S1 = Y1 * Z2 ** 3
    S2 = Y2 * Z1 ** 3
    if U1 == U2:
        if S1 != S2:
            return Point(p1.x.__class__.zero(), p1.x.__class__.zero(), p1.x.__class__.zero())
        else:
            return double_jac(p1)
    H = U2 - U1
    R = S2 - S1
    X3 = R ** 2 - H ** 3 - 2 * U1 * H ** 2
    Y3 = R * (U1 * H ** 2 - X3) - S1 * H ** 3
    Z3 = H * Z1 * Z2
    return Point(X3, Y3, Z3)

add_jac(P, -P)

r1_jac = add_jac(P, Q)
print(r1_jac)
# r1.x = r1.x * (r1.z.inv() ** 2)
# r1.y = r1.y * (r1.z.inv() ** 3)
r2_jac = add_jac(r1_jac, r1_jac)

r1_proj = add(P, Q)
print(r1_proj)
# r2.x = r2.x * (r2.z.inv())
# r2.y = r2.y * (r2.z.inv())
r2_proj = add(r1_proj, r1_proj)

print(r1_jac)
print(r2_jac)

print(r1_proj)
print(r2_proj)

print(r1_jac.x * (r1_jac.z.inv() ** 2))
print(r1_jac.y * (r1_jac.z.inv() ** 3))

print(r1_proj.x * (r1_proj.z.inv()))
print(r1_proj.y * (r1_proj.z.inv()))

print(r2_jac.x * (r2_jac.z.inv() ** 2))
print(r2_jac.y * (r2_jac.z.inv() ** 3))

print(r2_proj.x * (r2_proj.z.inv()))
print(r2_proj.y * (r2_proj.z.inv()))

def lin_func_jac(P1, P2, T):
    x1 = P1.x
    y1 = P1.y
    z1 = P1.z

    x2 = P2.x
    y2 = P2.y
    z2 = P2.z

    xt = FQ2([T.x, 0])
    yt = FQ2([T.y, 0])
    zt = FQ2([1, 0])
    zt_3 = zt

    z1_3 = z1 ** 3
    z2_3 = z2 ** 3

    n = y2 * z1_3 - y1 * z2_3
    d = x2 * z2 * z1_3 - x1 * z1 * z2_3

    if d != FQ2.zero():
        n = n * (xt * z1_3 * zt - x1 * zt_3 * z1) - d * (yt * z1_3 - y1 * zt_3)
        d = d * (zt_3 * z1_3)

    else:
        assert False

    return n, d


def lin_func_jac_twisted(P1, P2, T):
    x1 = P1.x
    y1 = P1.y
    z1 = P1.z

    x2 = P2.x
    y2 = P2.y
    z2 = P2.z

    xt = FQ2([T.x, 0])
    yt = FQ2([T.y, 0])
    zt = FQ2([1, 0])
    zt_3 = zt ** 3
    zt_2 = zt ** 2

    z1_2 = z1 ** 2

    z1_3 = z1 ** 3
    z2_3 = z2 ** 3

    x1_2 = x1 ** 2

    d = x2 * z2 * z1_3 - x1 * z1 * z2_3

    if d == FQ2.zero():
        m = (3 * x1_2) * (2 * z1 * y1).inv()
    else:
        m = (y2 * z1_3 - y1 * z2_3) * d.inv()

    l0 = FQ6([-yt * zt_3.inv(), FQ2.zero(), FQ2.zero()])
    l1 = FQ6([m * xt * zt_2.inv(), y1 * z1_3.inv() - m * x1 * z1_2.inv(), FQ2.zero()])

    return FQ12([l0, l1])



n = linear_func(r1_proj, r2_proj, Point(2, 3, 1))
print(n)

n_jac, d_jac = lin_func_jac(r1_jac, r2_jac, Point(2, 3, 1))

print((n_jac, d_jac))

print((n_jac * d_jac.inv()))

print(P)
print(Q)

print(linear_func(P, Q, Point(2, 3, 1)))
print(lin_func_jac(P, Q, Point(2, 3, 1)))


def frobenius_endomophism(pt: Point):
    #assert (pt.z == pt.z.__class__.one())
    return Point(fp12_pow_N(pt.x), fp12_pow_N(pt.y), fp12_pow_N(pt.z))


def fq2_frobenius_endomophism(pt: Point):
    assert (pt.z == pt.z.__class__.one())
    return Point(fp2_pow_N(pt.x), fp2_pow_N(pt.y), fp2_pow_N(pt.z))


def dbl_lin_func_jac(P, T):
    x = P.x
    y = P.y
    z = P.z

    y_squared = y * y
    x_squared = x * x
    z_squared = z * z
    y_4 = y_squared * y_squared

    R = 2 * y_squared
    S = 2 * x * R
    M = 3 * x_squared

    Xp = M ** 2 - 2 * S
    Yp = M * (S - Xp) - 8 * y_4
    Zp = 2 * y * z

    I = 3 * x_squared

    t = -Zp * z_squared * T.y
    tw = I * z_squared * T.x
    twv = R - I * x

    inv = (2 * y * z_squared * z).inv()

    t = t * inv
    tw = tw * inv
    twv = twv * inv

    return FQ12([FQ6([t, FQ2.zero(), FQ2.zero()]), FQ6([tw, twv, FQ2.zero()])]), Point(Xp, Yp, Zp)


def add_lin_func_jac(P0, P1, T):
    x0 = P0.x
    y0 = P0.y
    z0 = P0.z

    x1 = P1.x
    y1 = P1.y
    z1 = P1.z
    assert(z1 == FQ2.one())

    z0_squared = z0 * z0
    z0_cubed = z0 * z0_squared

    z1_squared = z1 * z1
    z1_cubed = z1 * z1_squared

    U1 = x0 * z1_squared
    U2 = x1 * z0_squared
    S1 = y0 * z1_cubed
    S2 = y1 * z0_cubed
    H = U2 - U1 # x1 * z0^2 - x0 * z1^2
    R = S2 - S1 # y1 * z0^3 - y0 * z1 ^3

    H_squared = H * H
    H_cubed = H * H_squared
    R_squared = R * R

    G = z1 * U2
    I = z0_squared * z1_cubed

    X3 = R_squared - H_cubed - 2 * U1 * H_squared
    Y3 = R * (U1 * H_squared - X3) - S1 * H_cubed
    Z3 = H * z0 * z1

    t = (z0 * I * x0 - G * z0_cubed) * T.y
    tw = (S2 * z0_squared - y0 * I) * T.x
    twv = y0 * G - x0 * S2

    inv = ((x1 * (z0 ** 2) - x0 * (z1 ** 2)) * (z0 ** 3) * z1).inv()

    t = t * inv
    tw = tw * inv
    twv = twv * inv

    return FQ12([FQ6([t, FQ2.zero(), FQ2.zero()]), FQ6([tw, twv, FQ2.zero()])]), Point(X3, Y3, Z3)


def multiply_jac(pt: Point, n):
    if n == 0:
        return Point(pt.x.__class__.one(), pt.x.__class__.one(), pt.x.__class__.zero())
    elif n == 1:
        return pt
    elif not n % 2:
        return multiply_jac(double_jac(pt), n // 2)
    else:
        return add_jac(multiply_jac(double_jac(pt), int(n // 2)), pt)

xi = FQ2([9, 1])

assert ((N - 1) % 3 == 0)
assert ((N - 1) % 2 == 0)

xi_2 = (xi ** ((N - 1) // 2))
print(xi_2)
xi_3 = (xi ** ((N - 1) // 3))
print(xi_3)



def mi(pt: Point):
    return Point(fp2_pow_N(pt.x) * xi_3, fp2_pow_N(pt.y) * xi_2, fp2_pow_N(pt.z))

def from_jac(pt: Point):
    zinv = pt.z.inv()
    zinv2 = zinv * zinv
    zinv3 = zinv * zinv2

    return Point(pt.x * zinv2, pt.y * zinv3, pt.x.__class__.one())

def check_subgroup_G2(pt: Point):
    xp = multiply_jac(pt, X)
    _2xp = add_jac(xp, xp)
    x_plus_1p = add_jac(xp, pt)

    e_xp = mi(xp)
    e2_xp = mi(e_xp)

    e_2xp = mi(_2xp)
    e2_2xp = mi(e_2xp)
    e3_2xp = mi(e2_2xp)

    l = add_jac(add_jac(x_plus_1p, e_xp), e2_xp)
    r = e3_2xp

    r1 = from_jac(l) == from_jac(r)
    r2 = (l.x * (r.z ** 2) == r.x * (l.z ** 2)) and (l.y * (r.z ** 3) == r.y * (l.z ** 3))
    assert (r1 == r2)
    return r2

def untwist_frob_twist(pt: Point):
    pt_untwisted = untwist(pt)
    pt_n = frobenius_endomophism(pt_untwisted)
    pt_twisted = twist(pt_n)

    return Point(pt_twisted.x, pt_twisted.y, FQ2.one())

_3_div_xi = FQ2([19485874751759354771024239261021720505790618469301721065564631296452457478373,
                 266929791119991161246907387137283842545076965332900288569378510910307636690])

def is_on_twisted_curve(p: Point):
    return (p.y ** 2) == (p.x ** 3) + _3_div_xi

def is_on_curve(p: Point):
    return (p.y ** 2) == (p.x ** 3) + FP(3)

ate_loop_count = 29793968203157093288
log_ate_loop_count = 63


def miller_loop(Q, P):
    R = Q
    R_t = Q

    f = FQ12.one()
    f_t = FQ12.one()

    for i in range(log_ate_loop_count, -1, -1):
        print(i)
        f_tmp_t = linear_func(untwist(R_t), untwist(R_t), cast_to_fq12(P))
        f_t = f_t * f_t * f_tmp_t
        R_t = double(R_t)

        f_tmp, R = dbl_lin_func_jac(R, P)
        f = f * f * f_tmp
        assert (R_t.to_affine() == from_jac(R))
        assert (f_tmp_t == f_tmp)

        print(from_jac(R))
        print(f)
        if ate_loop_count & (2 ** i):
            f_tmp_t = linear_func(untwist(R_t), untwist(Q), cast_to_fq12(P))
            f_t = f_t * f_tmp_t
            R_t = add(R_t, Q)

            f_tmp, R = add_lin_func_jac(R, Q, P)
            f = f * f_tmp
            assert (R_t.to_affine() == from_jac(R))
            assert (f_tmp_t == f_tmp)
            print(from_jac(R))
            print(f)

    print(f)
    Q1_t = frobenius_endomophism(untwist(Q))
    nQ2_t = -frobenius_endomophism(Q1_t)

    _n1_t = linear_func(untwist(R_t), Q1_t, cast_to_fq12(P))
    R_t = twist(add(untwist(R_t), Q1_t))
    _n2_t = linear_func(untwist(R_t), nQ2_t, cast_to_fq12(P))


    Q1 = untwist_frob_twist(Q)

    Q1t = fq2_frobenius_endomophism(Q)
    Q1t.x = Q1t.x * l[1]
    Q1t.y = Q1t.y * l[2]
    assert(Q1 == Q1t)

    _n1, R = add_lin_func_jac(R, Q1, P)

    assert (_n1 == _n1_t)

    nQ2 = -untwist_frob_twist(Q1)

    nQ2t = Point(Q.x * l2[1], -Q.y * l2[2], Q.z)

    assert(nQ2 == nQ2t)

    _n2, R = add_lin_func_jac(R, nQ2, P)

    assert (_n2 == _n2_t)

    assert (f_t * _n1_t * _n2_t == f * _n1 * _n2)

    print(f * _n1 * _n2)
    return f * _n1 * _n2


def final_exp_naive(f: FQ12):
    return f ** ((N ** 12 - 1) // R)


def final_exp(f: FQ12):
    print(f)

    f1 = conjugate(f)

    f2 = f.inv()

    f = f1 * f2  # easy 1
    f = fp12_pow_N2(f) * f  # easy 2

    print(f)

    f1 = conjugate(f)

    ft1 = f ** T
    print(ft1)
    ft2 = ft1 ** T
    print(ft2)
    ft3 = ft2 ** T
    print(ft3)
    fp1 = fp12_pow_N(f)
    print(fp1)
    fp2 = fp12_pow_N2(f)
    print(fp2)
    fp3 = fp12_pow_N3(f)
    print(fp3)
    y0 = fp1 * fp2 * fp3
    y1 = f1
    y2 = fp12_pow_N2(ft2)
    print(y2)
    y3 = fp12_pow_N(ft1)

    y3 = conjugate(y3)
    y4 = fp12_pow_N(ft2) * ft1

    y4 = conjugate(y4)
    y5 = conjugate(ft2)

    y6 = fp12_pow_N(ft3) * ft3
    y6 = conjugate(y6)

    print(y1)
    print(y2)
    print(y3)
    print(y4)
    print(y5)
    print(y6)

    t0 = (y6 ** 2) * y4 * y5
    t1 = y3 * y5 * t0
    t0 = t0 * y2
    t1 = ((t1 ** 2) * t0) ** 2
    t0 = t1 * y1
    t1 = t1 * y0
    t0 = t0 ** 2
    return t1 * t0


def cast_to_fq12(pt: Point):
    return Point(
        FQ12([
            FQ6([FQ2([pt.x, 0]), FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]),
        FQ12([
            FQ6([FQ2([pt.y, 0]), FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]),
        FQ12([
            FQ6([FQ2([pt.z, 0]), FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]))


def cast_fq2_to_fq12(pt: Point):
    return Point(
        FQ12([
            FQ6([pt.x, FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]),
        FQ12([
            FQ6([pt.y, FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]),
        FQ12([
            FQ6([pt.z, FQ2([0, 0]), FQ2([0, 0])]),
            FQ6([FQ2([0, 0]), FQ2([0, 0]), FQ2([0, 0])])
        ]))


def cast_fq12_to_fq2(pt: Point):
    return Point(
        pt.x.coeffs[0].coeffs[0],
        pt.y.coeffs[0].coeffs[0],
        pt.z.coeffs[0].coeffs[0])


def pairing(Q: Point, P: Point):
    f = miller_loop(Q, P)

    fn = final_exp_naive(f)
    f = final_exp(f)
    assert (f == fn)

    return f


def lin_func_and_double_twisted(P, T):
    x = P.x
    y = P.y
    z = P.z

    z_2 = z ** 2
    z_3 = z ** 3

    xt = FQ2([T.x, 0])
    yt = FQ2([T.y, 0])

    S = 4 * x * (y ** 2)
    M = 3 * (x ** 2)
    Xd = (M ** 2) - 2 * S
    Yd = M * (S - Xd) - 8 * (y ** 4)
    Zd = 2 * y * z

    x_2 = x ** 2

    m = (3 * x_2) * (2 * z * y).inv()

    l0 = FQ6([-yt, FQ2.zero(), FQ2.zero()])
    l1 = FQ6([m * xt, y * z_3.inv() - m * x * z_2.inv(), FQ2.zero()])

    return FQ12([l0, l1]), Point(Xd, Yd, Zd)


P1 = Point(FP(0x1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59),
           FP(0x3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41), FP(1))
Q1 = Point(FQ2([0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7,
                0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678]),
           FQ2([0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d,
                0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550]), FQ2.one())


# print(Q1)
# tQ1 = untwist(Q1)
# print(tQ1)
# print(twist(tQ1))
#
# n_proj, d_proj = linear_func(untwist(r1_proj), untwist(r2_proj), Point(2, 3, 1))
# n_jac, d_jac = lin_func_jac(r1_jac, r2_jac, Point(2, 3, 1))
#
# print((n_proj, d_proj))
# print((n_jac, d_jac))
#
# print(n_proj * d_proj.inv())
# print((n_jac * d_jac.inv()))
#
# print(lin_func_jac_twisted(r1_jac, r2_jac, Point(2, 3, 1)))
#
# n_proj, d_proj = linear_func(untwist(r1_proj), untwist(r1_proj), Point(2, 3, 1))
# print(n_proj * d_proj.inv())
#
# print(lin_func_jac_twisted(r1_jac, r1_jac, Point(2, 3, 1)))
#
# k, p = lin_func_and_double_twisted(r1_jac, Point(2, 3, 1))
# print(k)
# print(p)
# print(add_jac(r1_jac, r1_jac))
#
# assert (k == lin_func_jac_twisted(r1_jac, r1_jac, Point(2, 3, 1)))
# assert (p == add_jac(r1_jac, r1_jac))
#
# print(P)
# print(Q)
#
# print(linear_func(P, Q, Point(2, 3, 1)))
# print(lin_func_jac(P, Q, Point(2, 3, 1)))
#
P1 = Point(FP(0x1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59),
           FP(0x3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41), FP(1))
Q1 = Point(FQ2([0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678,
                0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7]),
           FQ2([0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550,
                0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d]), FQ2.one())


assert (is_on_curve(P1))
assert (is_on_twisted_curve(Q1))
assert (check_subgroup_G2(Q1))

print(from_jac(multiply_jac(P1, 17)))
print(-from_jac(multiply_jac(Q1, 17)))

p1 = pairing(Q1, from_jac(multiply_jac(P1, 17)))

p2 = pairing(-from_jac(multiply_jac(Q1, 17)), P1)

assert(p1 * p2 == FQ12.one())


p1 = pairing(Q1, P1)
# #
# # print(p1)
#
P2 = -P1
Q2 = Q1

# print(P2)
# print(Q2)
#
#
p2 = pairing(Q2, P2)
assert(p1 * p2 == FQ12.one())


def multiply(pt, n):
    if n == 0:
        return Point(pt[0].__class__.one(), pt[0].__class__.one(), pt[0].__class__.zero())
    elif n == 1:
        return pt
    elif not n % 2:
        return multiply(double(pt), n // 2)
    else:
        return add(multiply(double(pt), int(n // 2)), pt)



# P1 = Point(FP(0x1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59),
#            FP(0x3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41), FP(1))
# P1 = multiply(P1, 17)
# P1.x = P1.x / P1.z
# P1.y = P1.y / P1.z
# P1.z = P1.z / P1.z
Q1 = Point(FQ2([0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678,
                0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7]),
           FQ2([0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550,
                0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d]), FQ2.one())


#
#
def sqrt(a: FQ2):
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


def sqrt_c(a):
    a1 = (a ** ((N - 3) // 4))
    a0 = a1 * (a1 * a)

    if a0 == -a.__class__.one():
        return False

    return a1 * a

# s = sqrt(FQ2([2, 3]))
# print(s)
# assert (s * s == FQ2([2, 3]))
#

assert(N % 4 == 3)


def gen_twisted_curve_point(x: FQ2):
    y2 = x ** 3 + _3_div_xi
    y = sqrt(y2)
    return Point(x, y, FQ2.one()) if y else Point(FQ2.zero(), FQ2.zero(), FQ2.zero())




# New G2 subgroup inclusion testing
assert (X % 13 != 4)
assert (X % 97 != 92)
assert (X % 3061 != 564)



# | ecpairing_two_point_match_3  | 7721656 |
# | ecpairing_three_point_match_1  | 7722011 |


s = 2 * T * (6 * (T ** 2) + 3 * T + 1)
print(s)


def final_exp_test():
    input = FQ12([
        FQ6(
            [
                FQ2([0x2fd6674db39c7b96cb997f9ed3a901a216ae87b599679807a63184d740ca9ad3,
                     0x16d8bf23c2eba89adc463d30c86a7efd7948746f1c72ea674a06e5d7c36b4b1a]),
                FQ2([0xf1bc5baa6bf63219df3f50f41c276a38a4185ddb006ac01c4473706b309ae62,
                     0x55765de8957745c6e1117b0ae6a5f8f01896d13a567600c53b7df89ed9e197a]),
                FQ2([0x2c026466ad28bb8b5f1a6cef88be229c4d045fb37342108831d749d0ff1f7331,
                     0x2c3d1f171bc35bbb09c80fae0c60971272bf4f0ae53dedfa4f67f61376e80c4c])
            ]
        ),
        FQ6(
            [
                FQ2([0x2201a6fd7c400059bdf999e1faa72146d7a30b79a960e2dfe15437692164da06,
                     0xfb4c1dd1a86de356cd28ab6f7da52c909f37ca18206ea9f9328584636552a63]),
                FQ2([0x2cb1711330c4d98e449f8bd3e3bdbb02d6554616a9aba2438c1e67a260b2e766,
                     0x27127cd11258856759814ebcfe7218fd8ff029d573f9460fc2a8a48d871beece]),
                FQ2([0x25def3b6fd12fa7bf46dbe2007de4528454eafbc9df955dafc59457d2f9af0e2,
                     0x18cce3479836fe137367bc697a1afbf28f7f1e6ce73342214c7f5dfa2d2c5ee6])
            ]
        )
    ])

    out = final_exp(input) ** s

    print(out)


def fp12Expt_test():
    input = FQ12([
        FQ6(
            [
                FQ2([0xc10b092335fbc34e8f5a33d748c5341b05330d0e6498d8ae42d044a72d9134,
                     0x1ac6245627253458e438792b7078bead027464a4acb9c5ecb6d4fbcc809e21ce]),
                FQ2([0x2f2369b4d6ed1e31adccae685482d4d60cf280ae8250195a11b78075d0d153bb,
                     0x17c0b7d47bcd91965b7d88d6cce03cfe023316025f56b53363198a482ee0dfc3]),
                FQ2([0x20957bc30c84a0b3ca572e04cce9d73e3d378c1c4bb935583bc44e17ff90a1c5,
                     0x13e116bea9297399c3512bbe0d05ec6d6e110a6abef0e3942d6bebcefa305d0c])
            ]
        ),
        FQ6(
            [
                FQ2([0x1b07dc7feccff524c4ef351f94a5acfc83aae953315c00e6d17361a55f03516e,
                     0x54c9d99763c07b92e2e9f414ace87ab859852da93d0e861f18fb3dba79a1f2e]),
                FQ2([0xf3dc5c3289112dcef5ee4e35f8c34247277a132c759d403e482df47bbaadb3d,
                     0x1e00057c7f2bc96c76c4f1a5ace67be105eda25315a165e2be1532dd99c7b6a9]),
                FQ2([0x7d827f9ec2f35d66182210482e44a81dc569147b11f069065590bc16673dcf6,
                     0x4fee37ea6e016e5acb63b37b98204e36d7ee89250d9551f516a469a265d9eb])
            ]
        )
    ])

    out = input ** T

    print(out)


fp12Expt_test()
final_exp_test()

p = gen_twisted_curve_point(FQ2([2, 3]))
print(p)
assert (is_on_twisted_curve(p))
assert (is_on_twisted_curve(Q1))

n= linear_func(untwist(Q1), untwist(Q1), cast_to_fq12(Point(1,2,1)))
f_proj = n
print(f_proj)

f_jac, p = dbl_lin_func_jac(Q1, Point(1,2,1))
print(f_jac)

assert (p == double_jac(Q1))
assert (from_jac(p) == double(Q1).to_affine())
assert(final_exp(f_jac) == final_exp(f_proj))

Q2_proj = double(Q1)
Q2 = add_jac(Q1, Q1)

assert(from_jac(Q2) == Q2_proj.to_affine())

n = linear_func(untwist(Q2_proj), untwist(Q1), cast_to_fq12(Point(1,2,1)))
f_proj = n
print(f_proj)

assert (f_proj == linear_func(untwist(Q2_proj.to_affine()), untwist(Q1), cast_to_fq12(Point(1,2,1))))

f_jac, p = add_lin_func_jac(Q2, Q1, Point(1,2,1))
print(f_jac)

assert (p == add_jac(Q2, Q1))
assert(f_jac == f_proj)
assert(final_exp(f_jac) == final_exp(f_proj))


n= linear_func(untwist(Q2_proj), untwist(Q2_proj), cast_to_fq12(Point(1,2,1)))
f_proj = n
print(f_proj)

f_jac, p = dbl_lin_func_jac(Q2, Point(1,2,1))
print(f_jac)
assert (p == double_jac(Q2))

f_jac, p = dbl_lin_func_jac(from_jac(Q2), Point(1,2,1))
print(f_jac)

assert (p == double_jac(from_jac(Q2)))
assert (from_jac(p) == double(Q2_proj).to_affine())
assert(final_exp(f_jac) == final_exp(f_proj))

Q2_proj = double(Q1)
Q2 = add_jac(Q1, Q1)

assert(from_jac(Q2) == Q2_proj.to_affine())

p_untwisted = untwist(p)

k = 12

def trace_map(pt: Point):
    f = pt
    ret = f
    for i in range(k):
        f = frobenius_endomophism(f)
        ret = add_jac(ret, f)
        # ret.x = ret.x + f.x
        # ret.y = ret.y + f.y
        # ret.z = ret.z + f.z


    return ret

print(trace_map(p_untwisted))
print(trace_map(untwist(Q1)))

#
# assert (from_jac(multiply_jac(p, 6 * (X ** 2))) != mi(p) == untwist_frob_twist(p))  # p is not in G2.
# assert (from_jac(multiply_jac(Q1, 6 * (X ** 2))) == mi(Q1) == untwist_frob_twist(Q1))  # Q1 is in G2.


# 0000000000000000000000000000000000000000000000000000000000000001
# 0000000000000000000000000000000000000000000000000000000000000002
# 203e205db4f19b37b60121b83a7333706db86431c6d835849957ed8c3928ad79
# 27dc7234fd11d3e8c36c59277c3e6f149d5cd3cfa9a62aee49f8130962b4b3b9
# 195e8aa5b7827463722b8c153931579d3505566b4edf48d498e185f0509de152
# 04bb53b8977e5f92a0bc372742c4830944a59b4fe6b1c0466e2a6dad122b5d2e
# 030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3
# 1a76dae6d3272396d0cbe61fced2bc532edac647851e3ac53ce1cc9c7e645a83
# 198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
# 1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
# 090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
# 12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa

# assert (not check_subgroup_G2(p))
#
C = 21888242871839275222246405745257275088844257914179612981679871602714643921549
#
E_order = 479095176016622842441988045216678740799252316531100822436447802254070093686356349204969212544220033486413271283566945264650845755880805213916963058350733
#
print(N * N - E_order)

Order = 21888242871839275222246405745257275088548364400416034343698204186575808495617
#
p_times_cofactor = multiply_jac(p, C)
#
print(Order - C)
#
assert (check_subgroup_G2(p_times_cofactor))
#
print(p_times_cofactor)
print(multiply_jac(p_times_cofactor, Order))
#
# C_factors = [
#     # 1
#     10069,
#     5864401,
#     59048653669,
#     1875725156269,
#     18886676598472561,
#     11000004482149079869,
#     110759045130759085200961,
#     197620364512881247228717050342013327560683201906968909,
#     1989839450280201278345951979893732195208519160001269944721,
#     1158925063269705299129335498742753300160198129946430376908509,
#     11669216462062662656933279136840782979313034970430607465091777121,
#     370681489107860919521524912437157688734843183477352535634149440521,
#     3732391913827051598662234343329740767871136014433462681300250716605949,
#     2173824895405628684302950218021379986974303100027769687325441613140792921,
#     21888242871839275222246405745257275088844257914179612981679871602714643921549]
#
E_order_factors = [
    10069,
    5864401,
    59048653669,
    1875725156269,
    # 7273918018348900
    # 14547836036697800
    18886676598472561,
    # 29095672073395599
    # 58191344146791198
    # 116382688293582396
    # 232765376587164791
    # 465530753174329582
    # 931061506348659165
    # 1862123012697318330
    # 3724246025394636661
    # 7448492050789273322
    11000004482149079869,
    # 14896984101578546644
    # 29793968203157093288
    110759045130759085200961,
    197620364512881247228717050342013327560683201906968909,  ####
    1989839450280201278345951979893732195208519160001269944721,
    1158925063269705299129335498742753300160198129946430376908509,
    11669216462062662656933279136840782979313034970430607465091777121,
    370681489107860919521524912437157688734843183477352535634149440521,
    3732391913827051598662234343329740767871136014433462681300250716605949,
    2173824895405628684302950218021379986974303100027769687325441613140792921,
    21888242871839275222246405745257275088548364400416034343698204186575808495617,
    21888242871839275222246405745257275088844257914179612981679871602714643921549,
    220392717476549662212799059448995502866593481147789049806697217954631815742367573,
    128361433385857117452617044098892509286558116738164192221218092329959357917504830417,
    1292471272762195315630401017031748676006353677436575251475444971670360774871356137468773,
    41056327781234549855699839121746281721076501627334722507946584903239578422456215126572973,
    413396164429250682497041680116863310649519294885633320932514163390719315135711630109463265137,
    240770769696599675408315992245408016271362757219843374010324460453143086940230650444489669434173,
    2424320880075062131686333725919013315836351602446602932909956992302697742401182419325566481532687937,
    4325562534879352222670518607448050578567023108339265450480744498132211313285910814452290134187992287242251900602427507079681771853,
    43554089163700197530069451858394421275591355677868063820890616351693235713475835990720109361138894340242234387165842568785315760787857,
    25366833255109008053981211992036955260999028883567896647064728515591038157845208666184824715222196157295749288144776474945592862536505053,
    255418644045692602095536823547820102522999221828645151339294751423486163211343406059815000057572293107810899582329754326227174532880069378657,
    8113566461687904710930150999454766938782045281979802804171667154934504044235744778314518975594698812410839681316950590957780195343759230696457,
    81695500702735512534355690413510048306596413944254634435204516583035521221409714172848891565263022342164744751180375500353888786916311693882625533,
    47581207271489010074683488451353534690560365133687637544587129035065060451520145913692443394996527310200940637954806362563397135354136976255533127257,
    479095176016622842441988045216678740799252316531100822436447802254070093686356349204969212544220033486413271283566945264650845755880805213916963058350733
]

miller_multipies = [
                    0x33b,
                    0x676,
                    0xcec,
                    0x19d8,
                    0x33b0,
                    0x33af,
                    0x675e,
                    0xcebc,
                    0xcebd,
                    0x19d7a,
                    0x33af4,
                    0x33af3,
                    0x675e6,
                    0xcebcc,
                    0x19d798,
                    0x19d797,
                    0x33af2e,
                    0x675e5c,
                    0xcebcb8,
                    0x19d7970,
                    0x33af2e0,
                    0x675e5c0,
                    0x675e5c1,
                    0xcebcb82,
                    0x19d79704,
                    0x33af2e08,
                    0x33af2e07,
                    0x675e5c0e,
                    0xcebcb81c,
                    0xcebcb81d,
                    0x19d79703a,
                    0x33af2e074,
                    0x675e5c0e8,
                    0x675e5c0e7,
                    0xcebcb81ce,
                    0x19d797039c,
                    0x33af2e0738,
                    0x675e5c0e70,
                    0xcebcb81ce0,
                    0xcebcb81cdf,
                    0x19d797039be,
                    0x33af2e0737c,
                    0x33af2e0737d,
                    0x675e5c0e6fa,
                    0xcebcb81cdf4,
                    0x19d797039be8,
                    0x33af2e0737d0,
                    0x33af2e0737cf,
                    0x675e5c0e6f9e,
                    0xcebcb81cdf3c,
                    0xcebcb81cdf3b,
                    0x19d797039be76,
                    0x33af2e0737cec,
                    0x675e5c0e6f9d8,
                    0x675e5c0e6f9d9,
                    0xcebcb81cdf3b2,
                    0x19d797039be764,
                    0x33af2e0737cec8,
                    0x675e5c0e6f9d90,
                    0x675e5c0e6f9d8f,
                    0xcebcb81cdf3b1e,
                    0x19d797039be763c,
                    0x33af2e0737cec78,
                    0x33af2e0737cec77,
                    0x675e5c0e6f9d8ee,
                    0xcebcb81cdf3b1dc,
                    0xcebcb81cdf3b1dd,
                    0x19d797039be763ba,
                    0x33af2e0737cec774,
                    0x33af2e0737cec775,
                    0x675e5c0e6f9d8eea,
                    0xcebcb81cdf3b1dd4,
                    0x19d797039be763ba8]

for so in E_order_factors:
    for m in miller_multipies:
        if m % so < 1000:
            print(str(m % so) + ":" + str(so) + ":" + str(m))



print(E_order_factors[len(E_order_factors) - 2] * E_order_factors[1])

print(E_order_factors[len(E_order_factors) - 1] // E_order_factors[len(E_order_factors) - 2])

div = E_order_factors[len(E_order_factors) - 1] // E_order_factors[0]
print(div)

small_group_generator = None

while not small_group_generator:
    x = FQ2([random.randint(0, N), random.randint(0, N)])
    pt = gen_twisted_curve_point(x)
    if pt.z == FQ2.zero():
        continue

    for i, f in enumerate(E_order_factors):
        pr = multiply_jac(pt, f)
        if pr.z == FQ2.zero():
            if i == len(E_order_factors) - 1 and f % div == 0:
                ps = multiply_jac(pt, div)
                small_group_generator = ps
                assert (multiply_jac(ps, E_order_factors[0]).z == FQ2.zero())
                break
                # pa = ps
                # for i in range(1, E_order_factors[0]):
                #     pa = add_jac(pa, ps)
                #     print(pa)
                # pass

print(from_jac(small_group_generator))

print(multiply_jac(small_group_generator, E_order_factors[0]))
print(from_jac(multiply_jac(small_group_generator, E_order_factors[0] + 1)))

#
# private_key = random.randint(0, N)
#
# shared_secret = from_jac(multiply_jac(small_group_generator, private_key))
#
# print(hex(private_key))
# print(hex(private_key % E_order_factors[0]))
#
# n1 = 0
#
# p = small_group_generator
# for i in range(1, E_order_factors[0] + 1):
#     if shared_secret == from_jac(p):
#         print("Found")
#         n1 = i
#         print(hex(i))
#         print(private_key % E_order_factors[0])
#     p = add_jac(p, small_group_generator)
#
# shared_secret_2 = from_jac(multiply_jac(p_times_cofactor, private_key))
#
# print(hex(private_key))

# for i in range(private_key - 10 * E_order_factors[0], N, E_order_factors[0]):
#     print (i)
#     if from_jac(multiply_jac(p_times_cofactor, i)) == shared_secret_2:
#         print ("Private key is " + str(hex(i)))


# print(p_times_cofactor)
# print(from_jac(p_times_cofactor))
# print(multiply_jac(p, 21888242871839275222246405745257275088844257914179612981679871602714643921549))

assert (check_subgroup_G2(Q1))
print (Q1)

t = 6 * (T**2) + 1

print(from_jac(multiply_jac(Q1, t)))

assert (from_jac(multiply_jac(Q1, t - 1)) == untwist_frob_twist(Q1) == mi(Q1))

sk = random.randint(0, N)
print(sk)
g1 = P1
pk = from_jac(multiply_jac(g1, sk))

# small_group_generator = from_jac(Point(FQ2([0xb546e05f013b24e37fd242435087984da917288e5a3c585fe10cbc503674144,
#                                             0x10121391b3887da86e8a139fd77772f7ee506bb4850f71b155929baa4694a3d4]),
#                                        FQ2([0x21b024afaedfe2a61b58a932061e23fc3f8f2538e5eab2b429199e34cfc66f04,
#                                             0x155a3b837f1ef474dc66003af5f7d55d1a951359c2cca78b2fc26b100ed4eb5d]),
#                                        FQ2([0xb4f30c7a8940b7971b86abf2cdd61f4dbf79891f4df454e4c722b09cbea94bc,
#                                             0x1ed2ee07378822a938d35138e264ced758960361ad100c683faa8eb01dad382d])))

small_group_generator = from_jac(Point(FQ2([0x2140576f64bcf3624997117a1dc244bc1fd2adb3f6884cdf8d35dcae92575814,
                                            0x1bf467ac9eac334aacf558da0a773a334dc076bb38d3a13b594290f9091c372a]),
                                       FQ2([0x2d6b2d5440fd6b2fbed06b4773bd26e4e3907abc9f6ea443b816b1551559aaeb,
                                            0x224240152de2d395ba9c22be7db2ff3d3dc696e335ef027f3515902f256faba7]),
                                       FQ2([0x26637f6e1274535644c78580c3a4bab0f6fc336ff4c20e857ce98bdf86918776,
                                            0x2094193cc75a067092ca0c0bde0c0471254fc0434c868a045345e6f656c84fa1])))
print("small_group_generator")
print(small_group_generator)

small_group_order = E_order_factors[7]

assert (not check_subgroup_G2(small_group_generator))
assert (multiply_jac(small_group_generator, small_group_order).z == FQ2.zero())

sk_fake = sk + small_group_order
assert (sk_fake % small_group_order == sk % small_group_order)

H = small_group_generator
shared_secret = from_jac(multiply_jac(H, sk_fake))

assert (shared_secret == from_jac(multiply_jac(H, sk)))

pair1 = pairing(H, pk)
print(H)
print(pk)
print(pair1)

print(shared_secret)
print(g1)
pair2 = pairing(shared_secret, g1)
print(shared_secret)
print(g1)

print(pair2)
# assert (pair1.inv() * pair2 == FQ12.one())


shared_secret = from_jac(multiply_jac(H, sk))

print(shared_secret)
print(g1)
assert (check_subgroup_G2(shared_secret))
pair2 = pairing(shared_secret, g1)
print(pair2)

assert (check_subgroup_G2(H))
pair1 = pairing(H, pk)

assert (pair1.inv() * pair2 == FQ12.one())

assert (check_subgroup_G2(Q1))

assert (check_subgroup_G2(double_jac(Q1)))

print(N % 4)

print(multiply(Q1, 6 * (X ** 2)).to_affine())
print(mi(Q1))
print(untwist_frob_twist(Q1))

print(Q1)

Q1_12 = cast_fq2_to_fq12(Q1)
print(Q1_12)
Q1_12_N1 = Point(fp12_pow_N(Q1_12.x), fp12_pow_N(Q1_12.y), fp12_pow_N(Q1_12.z))
print(Q1_12_N1)

Q1_12_N2 = Point(Q1_12.x ** N, Q1_12.y ** N, Q1_12.z ** N)
print(Q1_12_N2)

Q1_N1 = Point(Q1.x ** N, Q1.y ** N, Q1.z ** N)
print(Q1_N1)

Q1_N2 = Point(fp2_pow_N(Q1.x), fp2_pow_N(Q1.y), fp2_pow_N(Q1.z))
print(Q1_N2)

assert (Q1_N1 == Q1_N2 == cast_fq12_to_fq2(Q1_12_N1) == cast_fq12_to_fq2(Q1_12_N2))

QZ = Point(FQ2([0x27dc7234fd11d3e8c36c59277c3e6f149d5cd3cfa9a62aee49f8130962b4b3b9,
                0x203e205db4f19b37b60121b83a7333706db86431c6d835849957ed8c3928ad79]),
           FQ2([0x04bb53b8977e5f92a0bc372742c4830944a59b4fe6b1c0466e2a6dad122b5d2e,
                0x195e8aa5b7827463722b8c153931579d3505566b4edf48d498e185f0509de152]), FQ2.one())

assert (check_subgroup_G2(QZ))
assert (check_subgroup_G2(multiply_jac(QZ, X + 1)))
assert (from_jac(multiply_jac(QZ, 6 * (X ** 2))) == mi(QZ))

print(multiply_jac(QZ, 21888242871839275222246405745257275088548364400416034343698204186575808495617))

print(P1)
print(Q1)

p1 = pairing(Q1, P1)

P3 = Point(FP(0x1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59),
           FP(0x3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41), FP(1))
Q3 = Point(FQ2([0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678,
                0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7]), -FQ2(
    [0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550,
     0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d]), FQ2.one())
Q3 = multiply(Q3, 17)

print(Q3)

Q3.x = Q3.x * Q3.z.inv()
Q3.y = Q3.y * Q3.z.inv()
Q3.z = Q3.z * Q3.z.inv()

print(P3)
print(Q3)
p2 = pairing(Q3, P3)

print(p1 * p2)

P1 = Point(FP(0x1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59),
           FP(0x3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41), FP(1))
Q1 = Point(FQ2([0x04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678,
                0x209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7]),
           FQ2([0x120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550,
                0x2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d]), FQ2.one())

# print (Q1)
tQ1 = untwist(Q1)
print(tQ1)
print(twist(tQ1))
#
# r = tQ1.x * tQ1.x * tQ1.x
# print (r)

rrr = FQ2([3, 4]) * FQ2([1, 2])

p = pairing(Q1, P1)
print(p)

# Generator for curve over FQ
G1 = Point(FP(1), FP(2), FP(1))
# Generator for twisted curve over FQ2
G2 = Point(FQ2([10857046999023057135944570762232829481370756359578518086990519993285655852781,
                11559732032986387107991004021392285783925812861821192530917403151452391805634]),
           FQ2([8495653923123431417604973247489272438418190587263600148770280649306958101930,
                4082367875863433681332203403145435568316851327593401208105741076214120093531]), FQ2.one())

p = pairing(G2, G1)
p_inv = pairing(G2, -G1)

assert p * p_inv == FQ12.one()

print(untwist(G2))

# fq2 = FQ6N([FP(1), FP(2), FP(3), FP(4), FP(3), FP(1)]) * FQ6N([FP(2), FP(2), FP(1), FP(1), FP(2), FP(2)])
# print(str(fq2))

# fq2 = FQ6N([1,2,3,4,5,6]) * FQ6N([7,8,9,10,11,12])
# print (str(fq2))

# print (FQ2([1, 2]) * FQ2([1, 3]))
