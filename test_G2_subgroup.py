from bls import gen_twisted_curve_point, FQ2, is_on_twisted_curve, multiply, R, Point
from bls_factors import bls_twisted_curve_order_factors

# R = G2 subgroup order

# G2 cofactor
h2 = 0x5d543a95414e7f1091d50792876a202cd91de4547085abaa68a205b2e5a7ddfa628f1cb4d9e82ef21537e293a6691ae1616ec6e786f0c70cf1c38e31c7238e5
p2 = gen_twisted_curve_point(FQ2([2, 0]))
assert (is_on_twisted_curve(p2))

# p2 order is not R
assert(multiply(p2, R).to_affine().z != FQ2.zero())

# G2 generator
g2 = multiply(p2, h2)

# g2 * R is infinity
assert(multiply(g2, R).z == FQ2.zero())

# twisted curve order
tw_curve_order = h2 * R
assert([tw_curve_order % f for f in bls_twisted_curve_order_factors] == [0] * len(bls_twisted_curve_order_factors))

def check_which_subgroup(p: Point):
    for i, f in enumerate(bls_twisted_curve_order_factors):
        if multiply(p, f).z == FQ2.zero():
            return i, f

    return -1, 0

# This point is in the largest subgroup
assert(check_which_subgroup(p2)[0] == len(bls_twisted_curve_order_factors) - 1)

# Multiply point by some big scalar but less than cofactor. Multiplying by cofactor moves point to G2 subgroup.
p2m = multiply(p2, bls_twisted_curve_order_factors[136] * R)

# We have found a point from smallest non-trivial subgroup.
[p2m_sg_index, p2m_sg_order] = check_which_subgroup(p2m)
assert(p2m_sg_index == 2)
assert(p2m_sg_order == 23)

# Verify that its order is 23
assert(multiply(p2m, 23).z == FQ2.zero())

# Verify that its order is not R. So it in not in G2.
assert(multiply(p2m, R).z != FQ2.zero())
print(f"Point:\n {p2m.to_affine()} is not in G2 and has very small order {p2m_sg_order}")