from nbhelper import md, code, save

md(r"""
# 05 · Object-Oriented Python

Classes let you bundle data with the behaviour that belongs to it. Python's object model is flexible:
almost everything, numbers and functions included, is an object, and your own classes can behave like
built-in types. This module goes from the four classic pillars to the features professional code uses
every day.

**You will learn**
1. The four pillars: encapsulation, inheritance, polymorphism, abstraction
2. The data model: special ("dunder") methods that make objects printable, comparable, iterable and callable
3. Properties, class methods and static methods
4. `dataclasses`: classes for data with almost no boilerplate
5. Abstract base classes versus `Protocol` (structural typing)
6. Composition over inheritance, `super()` and the method resolution order
7. `__slots__`: when memory per object matters
""")

md(r"""
## 1. The four pillars
**Encapsulation** keeps data and the rules that protect it together. Python has no truly private
attributes; a leading underscore is the convention for "internal, do not touch".
""")
code(r"""
class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self._balance = balance                     # internal by convention

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self._balance:
            raise ValueError("insufficient funds")
        self._balance -= amount

    @property
    def balance(self) -> float:                      # read-only from outside
        return self._balance

acct = BankAccount("Alice", 100)
acct.deposit(50); acct.withdraw(30)
print(acct.balance)
try:
    acct.balance = 1_000_000
except AttributeError as e:
    print("cannot set it directly:", e)
""")

md(r"""
**Inheritance** reuses and extends a base class; **polymorphism** means code can call the same method on
different types and get the right behaviour for each.
""")
code(r"""
class Animal:
    def __init__(self, name: str):
        self.name = name
    def speak(self) -> str:
        raise NotImplementedError

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name}: woof"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name}: meow"

for pet in [Dog("Rex"), Cat("Tom"), Dog("Fido")]:
    print(pet.speak())                    # one call, the right behaviour per type
print(isinstance(Dog("x"), Animal), issubclass(Cat, Animal))
""")

md(r"""
**Abstraction** defines an interface and hides the implementation. An abstract base class cannot be
instantiated and forces subclasses to implement its abstract methods.
""")
code(r"""
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    def describe(self) -> str:            # concrete method built on the abstract one
        return f"{type(self).__name__} with area {self.area():.2f}"

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return math.pi * self.r ** 2

class Square(Shape):
    def __init__(self, side): self.side = side
    def area(self): return self.side ** 2

print([s.describe() for s in (Circle(1), Square(2))])
try:
    Shape()
except TypeError as e:
    print("abstract:", e)
""")

md(r"""
## 2. The data model: special methods
Python calls specially named methods for built-in operations: `len(x)` calls `x.__len__()`, `a + b`
calls `a.__add__(b)`, `for item in x` calls `x.__iter__()`. Implementing them makes your classes feel
built in.
""")
code(r"""
from functools import total_ordering

@total_ordering                          # derives <=, >, >= from __eq__ and __lt__
class Vector:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
    def __repr__(self):  return f"Vector({self.x}, {self.y})"         # for developers
    def __str__(self):   return f"({self.x}, {self.y})"                # for users
    def __add__(self, o): return Vector(self.x + o.x, self.y + o.y)
    def __mul__(self, k): return Vector(self.x * k, self.y * k)
    __rmul__ = __mul__                                                  # so 3 * v works too
    def __abs__(self):   return math.hypot(self.x, self.y)
    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)
    def __lt__(self, o): return abs(self) < abs(o)
    def __hash__(self):  return hash((self.x, self.y))
    def __iter__(self):  yield from (self.x, self.y)                   # enables unpacking
    def __bool__(self):  return bool(abs(self))

v, w = Vector(3, 4), Vector(1, 1)
x, y = v
print(repr(v + w), str(3 * v), abs(v), v > w, sorted([v, w]), x, y, bool(Vector(0, 0)))
print({v, Vector(3, 4)})                 # equal objects collapse in a set thanks to __hash__
""")
code(r"""
class Playlist:
    '''A container: supports len(), indexing, slicing, 'in' and iteration.'''
    def __init__(self, songs): self._songs = list(songs)
    def __len__(self): return len(self._songs)
    def __getitem__(self, i): return self._songs[i]
    def __contains__(self, song): return song in self._songs

class Multiplier:
    '''A callable object: instances can be used like functions.'''
    def __init__(self, k): self.k = k
    def __call__(self, x): return x * self.k

p = Playlist(["intro", "verse", "chorus", "outro"])
print(len(p), p[1], p[1:3], "chorus" in p, [s.upper() for s in p])
double = Multiplier(2)
print(double(21), list(map(double, [1, 2, 3])))
""")

md(r"""
## 3. Properties, class methods, static methods
""")
code(r"""
class Temperature:
    def __init__(self, celsius: float):
        self.celsius = celsius                 # goes through the setter below

    @property
    def celsius(self) -> float:
        return self._c

    @celsius.setter
    def celsius(self, value: float) -> None:   # validation on every assignment
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._c = value

    @property
    def fahrenheit(self) -> float:             # computed attribute
        return self._c * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(cls, f: float) -> "Temperature":   # alternative constructor
        return cls((f - 32) * 5 / 9)

    @staticmethod
    def is_freezing(celsius: float) -> bool:   # related helper, needs neither instance nor class
        return celsius <= 0

t = Temperature.from_fahrenheit(212)
print(t.celsius, t.fahrenheit, Temperature.is_freezing(-3))
try:
    t.celsius = -300
except ValueError as e:
    print(e)
""")

md(r"""
## 4. Dataclasses
Most classes mainly hold data. `@dataclass` writes `__init__`, `__repr__` and `__eq__` for you, and
optionally ordering, immutability (`frozen=True`) and memory-saving slots.
""")
code(r"""
from dataclasses import dataclass, field, asdict, replace

@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int = 0
    patch: int = 0
    def __str__(self): return f"{self.major}.{self.minor}.{self.patch}"

@dataclass
class Experiment:
    name: str
    learning_rate: float = 1e-3
    tags: list[str] = field(default_factory=list)      # never a mutable default directly
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

print(sorted([Version(2, 1), Version(1, 9, 3), Version(2, 0, 5)]))
e = Experiment("baseline", tags=["cpu"])
print(e, asdict(e), replace(e, learning_rate=0.01))
""")

md(r"""
## 5. Abstract base classes versus Protocols
An ABC requires classes to **inherit** from it (nominal typing). A `Protocol` only requires them to **have
the right methods** (structural typing, "duck typing" that type checkers understand). Protocols let you
describe what you need from objects you do not control.
""")
code(r"""
from typing import Protocol, runtime_checkable

@runtime_checkable
class SupportsArea(Protocol):
    def area(self) -> float: ...

class Rectangle:                      # does NOT inherit from anything
    def __init__(self, w, h): self.w, self.h = w, h
    def area(self): return self.w * self.h

def total_area(items: list[SupportsArea]) -> float:
    return sum(i.area() for i in items)

shapes = [Rectangle(2, 3), Circle(1), Square(2)]
print(total_area(shapes), all(isinstance(s, SupportsArea) for s in shapes))
""")

md(r"""
## 6. Composition over inheritance, and `super()`
Inheritance models "is a"; composition models "has a". Deep inheritance trees become hard to change, so
prefer giving an object the parts it needs.
""")
code(r"""
class Engine:
    def __init__(self, hp): self.hp = hp
    def start(self): return f"engine ({self.hp} hp) started"

class GPS:
    def route(self, to): return f"routing to {to}"

class Car:                                  # a car HAS an engine and a GPS
    def __init__(self, engine: Engine, gps: GPS | None = None):
        self.engine, self.gps = engine, gps
    def drive(self, to):
        steps = [self.engine.start()]
        if self.gps:
            steps.append(self.gps.route(to))
        return "; ".join(steps)

print(Car(Engine(150), GPS()).drive("Madrid"))
""")
code(r"""
class Base:
    def setup(self): return ["Base"]
class Logging(Base):
    def setup(self): return super().setup() + ["Logging"]
class Caching(Base):
    def setup(self): return super().setup() + ["Caching"]
class Service(Logging, Caching):           # multiple inheritance
    def setup(self): return super().setup() + ["Service"]

print([c.__name__ for c in Service.__mro__])   # method resolution order
print(Service().setup())                        # super() follows the MRO: every class runs once
""")

md(r"""
## 7. `__slots__`: memory per object
Each normal object carries a `__dict__` for its attributes. With `__slots__` the attributes are stored in
fixed slots instead, which saves memory when you create millions of small objects.
""")
code(r"""
import tracemalloc

class PointDict:
    def __init__(self, x, y): self.x, self.y = x, y

class PointSlots:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x, self.y = x, y

def measure(cls, n=200_000):
    tracemalloc.start()
    objs = [cls(i, i) for i in range(n)]
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current / n

print(f"per object: regular {measure(PointDict):.0f} bytes, with __slots__ {measure(PointSlots):.0f} bytes")
""")

md(r"""
## Summary
| Tool | Reach for it when |
|---|---|
| plain class | behaviour plus state with rules (an account, a connection) |
| `@dataclass` | mostly data (records, configuration, results) |
| `@property` | validation or computed attributes behind attribute syntax |
| `@classmethod` | alternative constructors (`from_json`, `from_fahrenheit`) |
| ABC | a family of classes you own that must implement an interface |
| `Protocol` | describing what you need from any object, including third-party ones |
| composition | combining capabilities without deep inheritance |
| `__slots__` | millions of small objects |

## Exercises
1. Make `Vector` support subtraction and the dot product (`v @ w` calls `__matmul__`).
2. Write a frozen dataclass `Money(amount, currency)` whose `+` raises an error for different currencies.
3. Replace the `Animal` hierarchy with a `Protocol` and check that a class not inheriting from anything still works.
""")

save("05_object_oriented_python.ipynb")
