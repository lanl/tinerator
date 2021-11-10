import numpy as np
from ..logging import log, warn, debug, error


def __line_connectivity(nodes: np.ndarray, connect_ends: bool = False) -> np.ndarray:
    """
    Simple function to define a closed or open polyline for a set of
    nodes. Assumes adjacency in array == implicit connection.
    That is, requires a clockwise- or counter-clockwise set of nodes.
    """

    delta = 0 if connect_ends else -1
    size = np.shape(nodes)[0]
    connectivity = np.empty((size + delta, 2), dtype=int)
    for i in range(size - 1):
        connectivity[i] = np.array((i + 1, i + 2))
    if connect_ends:
        connectivity[-1] = np.array((size, 1))
    return connectivity


# TODO: reimplement in cython!!!!
def square_trace_boundary(
    A: np.ndarray, NDV: float, dist: float = 10.0, connect_ends: bool = False
):
    """
    Uses a square-tracing algorithm to quickly find a set of points
    composing the boundary at the interface of data and "empty" (noDataValue)
    cells in a DEM matrix.

    The optional parameter 'dist' denotes the seperation the points should
    have between each other.
    A smaller value will result in more points being created.

    References:
    1. A Square Tracing method of finding boundary in a psuedo-masked array.
       C T. Pavlidis, Algorithms for Graphics and Image Processing,
       Computer Science Press, Rockville, Maryland, 1982.

    # Arguments
    A (np.ndarray): matrix to perform boundary analysis on
    NDV (float): value characterizing undefined array values
    dist (float): spacing between boundary nodes

    # Returns
    Boundary nodes
    """

    if np.isnan(NDV):
        is_ndv = lambda x: np.isnan(x)
    else:
        is_ndv = lambda x: np.isclose(x, NDV)

    nRows = np.shape(A)[0] - 1
    nCols = np.shape(A)[1] - 1

    global last_point  # This point is used to see if boundary is being packed too tightly
    last_point = [-5e4, -5e4]
    bounds = np.zeros(
        (nRows, nCols), dtype=np.uint8
    )  # Array containing visited locations
    maxiters = 1e7  # Maximum times to run through algorithm

    # This point class contains movement and direction functions
    class _Point:
        # Init. point class
        def __init__(self):
            # Constant values
            self.north = 0
            self.east = 1
            self.south = 2
            self.west = 3

            # Dynamic values
            self.x = None
            self.y = None
            self.direction = None

        # Turn right
        def right(self):
            self.direction = (self.direction + 1) % 4

        # Turn left
        def left(self):
            if self.direction == 0:
                self.direction = 3
            else:
                self.direction -= 1

        def move(self):
            if self.direction == self.north:
                self.y = self.y - 1
            elif self.direction == self.east:
                self.x = self.x + 1
            elif self.direction == self.south:
                self.y = self.y + 1
            elif self.direction == self.west:
                self.x = self.x - 1
            else:
                error(
                    "Cannot move. Value "
                    + str(self.direction)
                    + " is not recognized as one of the cardinal directions."
                )

        def moveLeft(self):
            self.left()
            self.move()

        def moveRight(self):
            self.right()
            self.move()

        def getDir(self):
            return self.direction

        def position(self):
            return [self.x, self.y]

    # Update the boundary
    tmp_points = []

    def _updateB(x, y):
        global last_point  # TODO: um...what? why does this need to be here?
        bounds[y][x] = 1  # Mark as visited
        current_point = [x, y, A[y][x]]

        # Check if the current point and last saved point are far enough away
        if (_distance(current_point, last_point) >= dist) or (dist is None):
            tmp_points.append(current_point)
            last_point = current_point

    # Is the pixel you're on a valid pixel to move to?
    def _blackPixel(x, y, Xmax, Ymax):
        if (x >= Xmax) or (y >= Ymax) or (x < 0) or (y < 0):
            return False

        if not is_ndv(A[y][x]):
            return True
        else:
            return False

    # Move across the array until a valid pixel is found.
    # This will be the starting pixel for the trace
    def _getStartingPixel(arr, nrows, ncols):
        s = None
        for y in range(nrows):
            for x in range(ncols):
                if not is_ndv(arr[y][x]):
                    s = [x, y]
                    break
        if s is None:
            warn("Could not find starting pixel")

        return s

    # Find the distance between two points
    def _distance(v1, v2):
        x1, y1 = v1[:2]
        x2, y2 = v2[:2]
        return ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5

    debug(f"Running boundary algorithm at distance {dist}")

    # Find starting pixel
    p = _Point()
    s = _getStartingPixel(A, nRows, nCols)

    p.x, p.y = s
    p.direction = p.north
    _updateB(p.x, p.y)
    p.moveLeft()

    iters = 0
    while iters < maxiters:
        iters += 1  # Break if no convergence

        # Are we back at the origin?
        if [p.x, p.y] == s:
            break

        if _blackPixel(p.x, p.y, nCols, nRows):
            _updateB(p.x, p.y)
            p.moveLeft()
        else:
            p.moveRight()

    debug(f"Finished generating boundary with {len(tmp_points)} points")

    boundary = np.array(tmp_points, dtype=np.double) + 1.0

    return boundary, __line_connectivity(boundary, connect_ends=connect_ends)
