cimport numpy as np
from libc.stdint cimport uint32_t
from cython.operator cimport dereference as deref
from libcpp.set cimport set as cppset
from murmurhash.mrmr cimport hash128_x64

from snakefusion import Embeddings

import functools
import numpy
from typing import cast
import warnings
from enum import Enum
import srsly
from thinc.api import Ops, get_current_ops
from thinc.types import Floats2d

from .strings cimport StringStore
from .vectors cimport Vectors

from .strings import get_string_id
from .errors import Errors, Warnings
from .vectors import Mode
from . import util


def unpickle_vectors(bytes_data):
    return FinalfusionVectors().from_bytes(bytes_data)


cdef class FinalfusionVectors(Vectors):
    """Store, save and load finalfusion word vectors.

    This class supports loading of finalfusion word embeddings.

    DOCS: https://spacy.io/api/vectors
    """
    cdef object embeddings
    cdef object ops

    def __init__(self, embeddings_path: str, *, mmap: bool=False, name: str=None, strings=None):
        """Load finalfusion embeddings

        embeddings_path (str): The path to the finalfusion embeddings.
        mmap (bool): whether the embeddings should be memory-mapped (default: False).
        name (str): A name to identify the vectors table.

        DOCS: https://spacy.io/api/vectors#init
        """
        self.strings = strings
        if self.strings is None:
            self.strings = StringStore()

        self.name = name

        self.ops = get_current_ops()

        if embeddings_path == None:
            return

        self.embeddings = Embeddings(embeddings_path, mmap=mmap)

        for word in self.embeddings.vocab:
            self.strings.add(word)

        self.mode = Mode.finalfusion

    @property
    def shape(self):
        """Get `(rows, dims)` tuples of number of rows and number of dimensions
        in the vector table.

        RETURNS (tuple): A `(rows, dims)` pair.

        DOCS: https://spacy.io/api/vectors#shape
        """
        return self.embeddings.storage.shape

    @property
    def size(self):
        """The vector size i,e. rows * dims.

        RETURNS (int): The vector size.

        DOCS: https://spacy.io/api/vectors#size
        """
        return self.embeddings.storage.shape[0] * self.embeddings.storage.shape[1]

    @property
    def is_full(self):
        """Whether the vectors table is full.

        RETURNS (bool): `True` if no slots are available for new keys.

        DOCS: https://spacy.io/api/vectors#is_full
        """
        return True

    @property
    def n_keys(self):
        """Get the number of keys in the table. Note that this is the number
        of all keys, not just unique vectors.

        RETURNS (int): The number of keys in the table for default vectors.

        DOCS: https://spacy.io/api/vectors#n_keys
        """
        return len(self.embeddings.vocab)

    def __reduce__(self):
        return (unpickle_vectors, (self.to_bytes(),))

    def __getitem__(self, key):
        """Get a vector by key. If the key is not found, a KeyError is raised.

        key (str/int): The key to get the vector for.
        RETURNS (ndarray): The vector for the key.

        DOCS: https://spacy.io/api/vectors#getitem
        """
        try:
            self.embeddings[key]
        except KeyError:
            raise KeyError(Errors.E058.format(key=key))

    def __setitem__(self, key, vector):
        """Set a vector for the given key.

        key (str/int): The key to set the vector for.
        vector (ndarray): The vector to set.

        DOCS: https://spacy.io/api/vectors#setitem
        """
        warnings.warn(Warnings.W115.format(method="FinalfusionVectors.__setitem__"))
        return

    def __iter__(self):
        """Iterate over the keys in the table.

        YIELDS (int): A key in the table.

        DOCS: https://spacy.io/api/vectors#iter
        """
        yield from self.embeddings.vocab

    def __len__(self):
        """Return the number of vectors in the table.

        RETURNS (int): The number of vectors in the data.

        DOCS: https://spacy.io/api/vectors#len
        """
        return self.embeddings.storage.shape[0]

    def __contains__(self, key):
        """Check whether a key has been mapped to a vector entry in the table.

        key (int): The key to check.
        RETURNS (bool): Whether the key has a vector entry.

        DOCS: https://spacy.io/api/vectors#contains
        """
        return self.embeddings.vocab.get(key) is not None

    def resize(self, shape, inplace=False):
        """Resize the underlying vectors array. This operation is not supported
        on finalfusion embeddings.

        shape (tuple): A `(rows, dims)` tuple.
        inplace (bool): Reallocate the memory.
        RETURNS (list): -1.

        DOCS: https://spacy.io/api/vectors#resize
        """
        warnings.warn(Warnings.W115.format(method="NdArrayVectors.resize"))
        return -1

    def keys(self):
        """RETURNS (iterable): A sequence of keys in the table."""
        return self.embeddings.vocab

    def values(self):
        """Iterate over vectors that have been assigned to at least one key.

        Note that some vectors may be unassigned, so the number of vectors
        returned may be less than the length of the vectors table.

        YIELDS (ndarray): A vector in the table.

        DOCS: https://spacy.io/api/vectors#values
        """
        yield from self.embeddings.storage

    def items(self):
        """Iterate over `(key, vector)` pairs.

        YIELDS (tuple): A key/vector pair.

        DOCS: https://spacy.io/api/vectors#items
        """
        yield from self.embeddings

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
        raise ValueError(
            Errors.E858.format(
                mode=self.mode,
                alternative="Use Vectors[key] instead.",
            )
        )

    def get_batch(self, keys):
        """Get the vectors for the provided keys efficiently as a batch.
        keys (Iterable[Union[int, str]]): The keys.
        RETURNS: The requested vectors from the vector table.
        """
        keys = [self.strings.as_string(key) for key in keys]
        return self.ops.asarray2f(self.embeddings.embedding_batch(keys)[0])

    def add(self, key, *, vector=None, row=None):
        """Add a key to the table. Keys can be mapped to an existing vector
        by setting `row`, or a new vector can be added.

        This method is a noop with finalfusion embeddings.

        key (int): The key to add.
        vector (ndarray / None): A vector to add for the key.
        row (int / None): The row number of a vector to map the key to.
        RETURNS (int): -1

        DOCS: https://spacy.io/api/vectors#add
        """
        warnings.warn(Warnings.W115.format(method="FinalfusionVectors.add"))
        return -1

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

        # TODO: Similarity queries are supported by finalfusion, but not with a
        # batch size
        raise ValueError(Errors.E858.format(
            mode=self.mode,
            alternative="",
        ))

    def to_ops(self, ops: Ops):
        self.ops = ops

    def _get_cfg(self):
        return {
            "mode": Mode(self.mode).value,
        }

    def _set_cfg(self, cfg):
        self.mode = Mode(cfg.get("mode", Mode.default)).value

    def to_disk(self, path, *, exclude=tuple()):
        """Save the current state to a directory.

        path (str / Path): A path to a directory, which will be created if
            it doesn't exists.

        DOCS: https://spacy.io/api/vectors#to_disk
        """
        def save_vectors(path):
            self.embeddings.write(path)

        serializers = {
            "strings": lambda p: self.strings.to_disk(p.with_suffix(".json")),
            "embeddings": lambda p: save_vectors(p),
            "vectors.cfg": lambda p: srsly.write_json(p, self._get_cfg()),
        }
        return util.to_disk(path, serializers, exclude)

    def from_disk(self, path, *, exclude=tuple()):
        """Loads state from a directory. Modifies the object in place and
        returns it.

        path (str / Path): Directory path, string or Path-like object.
        RETURNS (Vectors): The modified object.

        DOCS: https://spacy.io/api/vectors#from_disk
        """
        def load_embeddings(path):
            self.embeddings = Embeddings(path)

        def load_settings(path):
            if path.exists():
                self._set_cfg(srsly.read_json(path))

        serializers = {
            "strings": lambda p: self.strings.from_disk(p.with_suffix(".json")),
            "embeddings": load_embeddings,
            "vectors.cfg": load_settings,
        }

        util.from_disk(path, serializers, exclude)

        return self

    def to_bytes(self, *, exclude=tuple()):
        """Serialize the current state to a binary string.

        exclude (list): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized form of the `Vectors` object.

        DOCS: https://spacy.io/api/vectors#to_bytes
        """
        serializers = {
            "strings": lambda: self.strings.to_bytes(),
            "embeddings": lambda: self.embeddings.to_bytes(),
            "vectors.cfg": lambda: srsly.json_dumps(self._get_cfg()),
        }
        return util.to_bytes(serializers, exclude)

    def from_bytes(self, data, *, exclude=tuple()):
        """Load state from a binary string.

        data (bytes): The data to load from.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Vectors): The `Vectors` object.

        DOCS: https://spacy.io/api/vectors#from_bytes
        """
        deserializers = {
            "strings": lambda b: self.strings.from_bytes(b),
            "embeddings": lambda b: Embeddings.from_bytes(b),
            "vectors.cfg": lambda b: self._set_cfg(srsly.json_loads(b))
        }
        util.from_bytes(data, deserializers, exclude)

        return self

    def clear(self):
        """Clear all entries in the vector table.

        Raises an error for FinalfusionEmbeddings, since clearing the embedding
        table is not supported.

        DOCS: https://spacy.io/api/vectors#clear
        """
        raise ValueError(Errors.E859)
