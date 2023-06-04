# mass of cf = 0.027 kgs

### cylinder
HEIGHT = 0.3
RADIUS = 0.005
MASS = 0.027 * 0.1

ixx = 1 / 12 * MASS * (3 * RADIUS**2 + HEIGHT**2)
iyy = 1 / 12 * MASS * (3 * RADIUS**2 + HEIGHT**2)
izz = 1 / 2 * MASS * RADIUS**2

print(f"{ixx=}")
print(f"{iyy=}")
print(f"{izz=}")
