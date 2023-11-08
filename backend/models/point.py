import attr

@attr.attrs
class Point:
    x = attr.ib(type=int, default=0)
    y = attr.ib(type=int, default=0)