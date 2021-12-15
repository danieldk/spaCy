from enum import Enum


class Mode(str, Enum):
    default = "default"
    floret = "floret"

    @classmethod
    def values(cls):
        return list(cls.__members__.keys())


cdef class Vectors:
    """Store, save and load word vectors.

    Vectors data is kept in the vectors.data attribute, which should be an
    instance of numpy.ndarray (for CPU vectors) or cupy.ndarray
    (for GPU vectors).

    In the default mode, `vectors.key2row` is a dictionary mapping word hashes
    to rows in the vectors.data table. Multiple keys can be mapped to the same
    vector, and not all of the rows in the table need to be assigned - so
    len(list(vectors.keys())) may be greater or smaller than vectors.shape[0].

    In floret mode, the floret settings (minn, maxn, etc.) are used to
    calculate the vector from the rows corresponding to the key's ngrams.

    DOCS: https://spacy.io/api/vectors
    """

    @property
    def shape(self):
        """Get `(rows, dims)` tuples of number of rows and number of dimensions
        in the vector table.

        RETURNS (tuple): A `(rows, dims)` pair.

        DOCS: https://spacy.io/api/vectors#shape
        """
        raise NotImplementedError

    @property
    def size(self):
        """The vector size i,e. rows * dims.

        RETURNS (int): The vector size.

        DOCS: https://spacy.io/api/vectors#size
        """
        raise NotImplementedError

    @property
    def is_full(self):
        """Whether the vectors table is full.

        RETURNS (bool): `True` if no slots are available for new keys.

        DOCS: https://spacy.io/api/vectors#is_full
        """
        raise NotImplementedError

    @property
    def n_keys(self):
        """Get the number of keys in the table. Note that this is the number
        of all keys, not just unique vectors.

        RETURNS (int): The number of keys in the table for default vectors.
        For floret vectors, return -1.

        DOCS: https://spacy.io/api/vectors#n_keys
        """
        raise NotImplementedError

    def __reduce__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """Get a vector by key. If the key is not found, a KeyError is raised.

        key (str/int): The key to get the vector for.
        RETURNS (ndarray): The vector for the key.

        DOCS: https://spacy.io/api/vectors#getitem
        """
        raise NotImplementedError

    def __setitem__(self, key, vector):
        """Set a vector for the given key.

        key (str/int): The key to set the vector for.
        vector (ndarray): The vector to set.

        DOCS: https://spacy.io/api/vectors#setitem
        """
        raise NotImplementedError

    def __iter__(self):
        """Iterate over the keys in the table.

        YIELDS (int): A key in the table.

        DOCS: https://spacy.io/api/vectors#iter
        """
        raise NotImplementedError

    def __len__(self):
        """Return the number of vectors in the table.

        RETURNS (int): The number of vectors in the data.

        DOCS: https://spacy.io/api/vectors#len
        """
        raise NotImplementedError

    def __contains__(self, key):
        """Check whether a key has been mapped to a vector entry in the table.

        key (int): The key to check.
        RETURNS (bool): Whether the key has a vector entry.

        DOCS: https://spacy.io/api/vectors#contains
        """
        raise NotImplementedError

    def resize(self, shape, inplace=False):
        """Resize the underlying vectors array. If inplace=True, the memory
        is reallocated. This may cause other references to the data to become
        invalid, so only use inplace=True if you're sure that's what you want.

        If the number of vectors is reduced, keys mapped to rows that have been
        deleted are removed. These removed items are returned as a list of
        `(key, row)` tuples.

        shape (tuple): A `(rows, dims)` tuple.
        inplace (bool): Reallocate the memory.
        RETURNS (list): The removed items as a list of `(key, row)` tuples.

        DOCS: https://spacy.io/api/vectors#resize
        """
        raise NotImplementedError

    def keys(self):
        """RETURNS (iterable): A sequence of keys in the table."""
        raise NotImplementedError

    def values(self):
        """Iterate over vectors that have been assigned to at least one key.

        Note that some vectors may be unassigned, so the number of vectors
        returned may be less than the length of the vectors table.

        YIELDS (ndarray): A vector in the table.

        DOCS: https://spacy.io/api/vectors#values
        """
        raise NotImplementedError

    def items(self):
        """Iterate over `(key, vector)` pairs.

        YIELDS (tuple): A key/vector pair.

        DOCS: https://spacy.io/api/vectors#items
        """
        raise NotImplementedError

    def find(self, *, key=None, keys=None, row=None, rows=None):
        """Look up one or more keys by row, or vice versa.

        key (Union[int, str]): Find the row that the given key points to.
            Returns int, -1 if missing.
        keys (Iterable[Union[int, str]]): Find rows that the keys point to.
            Returns ndarray.
        row (int): Find the first key that points to the row.
            Returns int.
        rows (Iterable[int]): Find the keys that point to the rows.
            Returns ndarray.
        RETURNS: The requested key, keys, row or rows.
        """
        raise NotImplementedError

    def get_batch(self, keys):
        """Get the vectors for the provided keys efficiently as a batch.
        keys (Iterable[Union[int, str]]): The keys.
        RETURNS: The requested vectors from the vector table.
        """
        raise NotImplementedError

    def add(self, key, *, vector=None, row=None):
        """Add a key to the table. Keys can be mapped to an existing vector
        by setting `row`, or a new vector can be added.

        key (int): The key to add.
        vector (ndarray / None): A vector to add for the key.
        row (int / None): The row number of a vector to map the key to.
        RETURNS (int): The row the vector was added to.

        DOCS: https://spacy.io/api/vectors#add
        """
        raise NotImplementedError

    def most_similar(self, queries, *, batch_size=1024, n=1, sort=True):
        """For each of the given vectors, find the n most similar entries
        to it, by cosine.

        Queries are by vector. Results are returned as a `(keys, best_rows,
        scores)` tuple. If `queries` is large, the calculations are performed in
        chunks, to avoid consuming too much memory. You can set the `batch_size`
        to control the size/space trade-off during the calculations.

        queries (ndarray): An array with one or more vectors.
        batch_size (int): The batch size to use.
        n (int): The number of entries to return for each query.
        sort (bool): Whether to sort the n entries returned by score.
        RETURNS (tuple): The most similar entries as a `(keys, best_rows, scores)`
            tuple.
        """
        raise NotImplementedError

    def to_disk(self, path, *, exclude=tuple()):
        """Save the current state to a directory.

        path (str / Path): A path to a directory, which will be created if
            it doesn't exists.

        DOCS: https://spacy.io/api/vectors#to_disk
        """
        raise NotImplementedError

    def from_disk(self, path, *, exclude=tuple()):
        """Loads state from a directory. Modifies the object in place and
        returns it.

        path (str / Path): Directory path, string or Path-like object.
        RETURNS (Vectors): The modified object.

        DOCS: https://spacy.io/api/vectors#from_disk
        """
        raise NotImplementedError

    def to_bytes(self, *, exclude=tuple()):
        """Serialize the current state to a binary string.

        exclude (list): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized form of the `Vectors` object.

        DOCS: https://spacy.io/api/vectors#to_bytes
        """
        raise NotImplementedError

    def from_bytes(self, data, *, exclude=tuple()):
        """Load state from a binary string.

        data (bytes): The data to load from.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Vectors): The `Vectors` object.

        DOCS: https://spacy.io/api/vectors#from_bytes
        """
        raise NotImplementedError

    def clear(self):
        """Clear all entries in the vector table.

        DOCS: https://spacy.io/api/vectors#clear
        """
        raise NotImplementedError
