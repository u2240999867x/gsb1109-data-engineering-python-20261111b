class InvalidGlob(Exception):
    """The glob passed is invalid."""
    pass


class PathNotFound(Exception):
    """One or more elements of the requested path did not exist in the object"""
    pass


class InvalidKeyName(Exception):
    """This key contains the separator character or another invalid character"""
    pass


class FilteredValue(Exception):
    """Unable to return a value, since the filter rejected it"""
    pass


class ShapeError(Exception):
    """Base class for failures while planning the shape of a whole row."""

    def __init__(self, message, sources=()):
        super().__init__(message)
        self.sources = tuple(sources)


class InfeasibleShapeError(ShapeError):
    """The columns of a row cannot be materialized under one shared shape.

    ``sources`` holds the identifiers of the columns whose declarations
    contradict each other.
    """
    pass


class AmbiguousShapeError(ShapeError):
    """More than one target shape remains feasible for the row.

    ``structures`` contains the distinct feasible target structures.
    """

    def __init__(self, message, structures, sources=()):
        super().__init__(message, sources)
        self.structures = tuple(structures)
