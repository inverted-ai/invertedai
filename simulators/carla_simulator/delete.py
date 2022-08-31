from dataclasses import dataclass


@dataclass
class data:
    x: int = 0
    y: int = 0


d = data()
print(d)
d.z = 0
print(d)
