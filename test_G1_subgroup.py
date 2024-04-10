from bls import gen_curve_point, is_on_curve, multiply, R, Point, FP, N
from bls_factors import bls_curve_order_factors

# R = G1 subgroup order

# Returns first smallest subgroup order which p belongs to.
def which_subgroup(p: Point):
    for i, f in enumerate(bls_curve_order_factors):
        if multiply(p, f).z == FP.zero():
            return i, f

    return -1, 0

# Returns all subgroups orders which p belongs to.
def which_subgroups(p: Point):
    sg = []
    for i, f in enumerate(bls_curve_order_factors):
        if multiply(p, f).z == FP.zero():
            sg = sg + [i, f]

    return sg

# G1 cofactor
h1 = 0x396c8c005555e1568c00aaab0000aaab
p1 = gen_curve_point(FP(54))
assert (is_on_curve(p1))

# p1 order is not R
assert(multiply(p1, R).to_affine().z != 0)

# G1 generator
g1 = multiply(p1, h1)

# g1 * R is infinity
assert(multiply(g1, R).z == 0)

# curve order
curve_order = h1 * R
# Verify that curve order has precalculated factors
assert([curve_order % f for f in bls_curve_order_factors] == [0] * len(bls_curve_order_factors))

# This point is in the 242nd subgroup
assert(which_subgroup(p1)[0] == 242)

# Verify that cofactor works
assert(multiply(p1, R).z != FP.zero())
p1_g1 = multiply(p1, h1)
assert(multiply(p1_g1, R).z == FP.zero())

# Multiply point by some big scalar but less than cofactor. Multiplying by cofactor moves point to G2 subgroup.
p1m = multiply(p1, bls_curve_order_factors[136] * R)

# We have found a point from very small subgroup.
[p1m_sg_index, p1m_sg_order] = which_subgroup(p1m)
assert(p1m_sg_index == 6)
assert(p1m_sg_order == 10177)

# Verify that its order is 23
assert(multiply(p1m, p1m_sg_order).z == FP.zero())

# Verify that its order is not R. So it in not in G2.
assert(multiply(p1m, R).z != FP.zero())
print(f"Point:\n {p1m.to_affine()} is not in G1 and has very small order {p1m_sg_order}")
