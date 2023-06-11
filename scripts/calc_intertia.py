# mass of cf = 0.027 kgs

### cylinder
HEIGHT = 0.005
RADIUS = 0.05
# in kgs
MASS = 0.005

ixx = 1 / 12 * MASS * (3 * RADIUS**2 + HEIGHT**2)
iyy = 1 / 12 * MASS * (3 * RADIUS**2 + HEIGHT**2)
izz = 1 / 2 * MASS * RADIUS**2

print("cylinder")

print(f"{ixx=}")
print(f"{iyy=}")
print(f"{izz=}")

# ### shere
# RADIUS = 0.02
# # in kgs
# MASS = 0.010

# ixx = 2 / 3 * MASS * RADIUS**2
# iyy = 2 / 3 * MASS * RADIUS**2
# izz = 2 / 3 * MASS * RADIUS**2

# print("sphere")

# print(f"{ixx=}")
# print(f"{iyy=}")
# print(f"{izz=}")
