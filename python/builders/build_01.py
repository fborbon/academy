from nbhelper import md, code, save

md(r"""
# 01 · Python Data Structures

The building blocks every Python program is made of: the four core collections, the specialised
containers in the standard library, functions, a first look at classes, comprehensions and error
handling. Every example is short and runnable; change it and run it again.

**You will learn**
1. Lists, tuples, dictionaries and sets, and when to use each one
2. The `collections` module: `deque`, `Counter`, `defaultdict`, `namedtuple`
3. Functions: default arguments, `*args`, `**kwargs`, lambdas
4. Classes in five minutes (module 05 goes much deeper)
5. Comprehensions and generators
6. Exceptions
7. Why the choice of structure matters: measuring lookup speed
""")

md(r"""
## 1. Lists
Ordered, **mutable** and allow duplicates. The default container for "a sequence of things".
""")
code(r"""
fruits = ["apple", "banana", "cherry"]
fruits.append("date")             # add at the end
fruits.insert(1, "apricot")       # add at a position
fruits.remove("banana")           # remove the first matching value
last = fruits.pop()               # remove and return the last item
print(fruits, "| popped:", last)

# slicing: [start:stop:step], stop is excluded
nums = list(range(10))
print(nums[2:5], nums[::2], nums[::-1])

# sorting: sorted() returns a new list, .sort() sorts in place
words = ["pear", "Fig", "apple"]
print(sorted(words), sorted(words, key=str.lower), sorted(words, key=len, reverse=True))
""")

md(r"""
## 2. Tuples
Ordered and **immutable**. Use them for fixed groups of values (a point, a database row) and to return
several values from a function. Being immutable, they can be dictionary keys or set members.
""")
code(r"""
point = (3, 4)
single = ("only_one",)            # the trailing comma makes it a tuple
x, y = point                      # unpacking
print(point, single, x, y)

def min_max(values):
    return min(values), max(values)   # returns a tuple

lo, hi = min_max([7, 2, 9, 4])
print(lo, hi)

first, *middle, last = [1, 2, 3, 4, 5]   # extended unpacking
print(first, middle, last)
""")

md(r"""
## 3. Dictionaries
Key-value pairs, mutable, and insertion-ordered since Python 3.7. Lookup by key is fast, independent of
the dictionary's size.
""")
code(r"""
person = {"name": "Alice", "age": 30, "city": "Berlin"}
person["age"] = 31                       # update
person["email"] = "alice@example.com"    # add
print(person.get("phone", "n/a"))        # safe access with a default
for key, value in person.items():
    print(f"{key:>6}: {value}")

# merging and dict comprehensions
defaults = {"city": "Unknown", "active": True}
merged = defaults | person               # right-hand side wins (Python 3.9+)
squares = {n: n * n for n in range(5)}
print(merged, squares)
""")

md(r"""
## 4. Sets
Unordered, mutable, **unique** elements. Ideal for membership tests and removing duplicates.
""")
code(r"""
a, b = {1, 2, 3}, {3, 4, 5}
print(a | b, a & b, a - b, a ^ b)        # union, intersection, difference, symmetric difference

nums = [1, 2, 2, 3, 3, 3]
print(sorted(set(nums)))                 # dedup (sets do not keep order, so sort if you need one)
print(3 in a, frozenset(a))              # frozenset: immutable set, usable as a dict key
""")

md(r"""
## 5. Specialised containers: `collections`
The standard library has ready-made structures for common patterns. Reaching for them makes code shorter
and often faster.

| Structure | Use it for |
|---|---|
| `deque` | queues and sliding windows: fast append/pop at **both** ends |
| `Counter` | counting things |
| `defaultdict` | grouping without "if key not in dict" checks |
| `namedtuple` | lightweight records with named fields |
| `heapq` (module) | always getting the smallest item: priority queues, top-k |
""")
code(r"""
from collections import deque, Counter, defaultdict, namedtuple
import heapq

window = deque(maxlen=3)                 # keeps only the last 3 items
for reading in [10, 12, 11, 15, 14]:
    window.append(reading)
print("last 3 readings:", list(window), "mean:", sum(window) / len(window))

text = "the quick brown fox jumps over the lazy dog the end"
print(Counter(text.split()).most_common(2))

by_letter = defaultdict(list)
for word in ["apple", "avocado", "banana", "blueberry", "cherry"]:
    by_letter[word[0]].append(word)
print(dict(by_letter))

City = namedtuple("City", ["name", "population_m"])
cities = [City("Tokyo", 37.4), City("Delhi", 32.9), City("Shanghai", 29.2), City("Madrid", 6.8)]
print(heapq.nlargest(2, cities, key=lambda c: c.population_m))
""")

md(r"""
## 6. Functions
Reusable blocks of code. Type hints document what goes in and out; Python does not enforce them at run
time, but editors and checkers such as `mypy` use them.
""")
code(r"""
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

print(greet("Alice"), greet("Bob", greeting="Hi"))

def total(*args: float) -> float:          # any number of positional arguments
    return sum(args)

def describe(**kwargs) -> str:             # any number of keyword arguments
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())

print(total(1, 2, 3), "|", describe(colour="red", size="M"))

square = lambda n: n ** 2                   # small anonymous function
print(list(map(square, [1, 2, 3])), sorted(["bb", "a", "ccc"], key=lambda s: -len(s)))
""")

md(r"""
**A classic trap: mutable default arguments.** The default value is created once, when the function is
defined, not on every call.
""")
code(r"""
def add_item_bad(item, items=[]):
    items.append(item)
    return items

def add_item_good(item, items=None):
    items = [] if items is None else items
    items.append(item)
    return items

print(add_item_bad("a"), add_item_bad("b"))     # the same list is reused: ['a', 'b'] twice
print(add_item_good("a"), add_item_good("b"))
""")

md(r"""
## 7. Classes in five minutes
A class is a blueprint, an object is one instance of it, and methods are functions that belong to the
class. Module 05 covers object-oriented programming in depth; this is just enough to read code that uses it.
""")
code(r"""
class Dog:
    species = "Canis familiaris"            # class attribute: shared by every dog

    def __init__(self, name: str, age: int):
        self.name = name                    # instance attributes: one per dog
        self.age = age

    def speak(self) -> str:
        return f"{self.name} says woof"

    def __repr__(self) -> str:              # how the object prints
        return f"Dog(name={self.name!r}, age={self.age})"

rex = Dog("Rex", 4)
print(rex, rex.speak(), Dog.species)
""")

md(r"""
## 8. Comprehensions and generators
Comprehensions build a collection in one readable line. A **generator** produces values lazily, one at a
time, so it uses almost no memory even for millions of items. That idea returns in module 03.
""")
code(r"""
import sys

squares = [n ** 2 for n in range(10) if n % 2 == 0]
pairs = [(x, y) for x in range(3) for y in "ab"]
print(squares, pairs)

big_list = [n for n in range(1_000_000)]
big_gen = (n for n in range(1_000_000))
print(f"list: {sys.getsizeof(big_list) / 1e6:.1f} MB, generator: {sys.getsizeof(big_gen)} bytes")
print("sum from the generator:", sum(big_gen))

def countdown(n):
    while n > 0:
        yield n          # 'yield' makes this function a generator
        n -= 1

print(list(countdown(5)))
""")

md(r"""
## 9. Exceptions
Handle the errors you expect, let the unexpected ones surface. `else` runs only when nothing failed,
`finally` always runs (cleanup).
""")
code(r"""
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("cannot divide by zero")
        return None
    except TypeError as e:
        print("bad input:", e)
        return None
    else:
        return result
    finally:
        print(f"divide({a!r}, {b!r}) finished")

print(divide(10, 2), divide(1, 0), divide("a", 1))

class InsufficientFunds(Exception):
    '''A custom exception makes the failure explicit for the caller.'''

try:
    raise InsufficientFunds("balance 10, requested 50")
except InsufficientFunds as e:
    print("caught:", e)
""")

md(r"""
## 10. Why the choice of structure matters
Checking whether an item is **in** a list scans it item by item (time grows with the size, O(n)). A set
or dict uses hashing and answers in about the same time whatever its size (O(1) on average). Measured:
""")
code(r"""
import timeit
import matplotlib.pyplot as plt

sizes = [1_000, 10_000, 100_000, 1_000_000]
t_list, t_set = [], []
for n in sizes:
    data_list = list(range(n))
    data_set = set(data_list)
    target = n - 1                          # worst case for the list: the last element
    t_list.append(timeit.timeit(lambda: target in data_list, number=50) / 50 * 1e6)
    t_set.append(timeit.timeit(lambda: target in data_set, number=50) / 50 * 1e6)

for n, a, b in zip(sizes, t_list, t_set):
    print(f"n={n:>9,}  list: {a:10.1f} µs   set: {b:6.2f} µs")

plt.figure(figsize=(6, 3.5))
plt.loglog(sizes, t_list, "o-", label="list (O(n))")
plt.loglog(sizes, t_set, "s-", label="set (O(1))")
plt.xlabel("number of items"); plt.ylabel("time per lookup (µs)"); plt.legend(); plt.title("Membership test")
plt.tight_layout(); plt.show()
""")

md(r"""
| Operation | list | tuple | dict | set |
|---|---|---|---|---|
| ordered | yes | yes | yes (insertion) | no |
| mutable | yes | no | yes | yes |
| duplicates | yes | yes | keys unique | no |
| `x in c` | O(n) | O(n) | O(1) on keys | O(1) |
| typical use | sequences | fixed records | lookups by key | membership, dedup |

## Exercises
1. Given a list of words, return the five most frequent with their counts (one line with `Counter`).
2. Group a list of `(student, grade)` tuples into a dict `grade -> [students]` with `defaultdict`.
3. Write a generator `fibonacci()` that yields the sequence forever, and print the first 15 values with `itertools.islice`.
4. Repeat the timing experiment with a `dict` and a `tuple`. Which behaves like a list and which like a set?
""")

save("01_data_structures.ipynb")
