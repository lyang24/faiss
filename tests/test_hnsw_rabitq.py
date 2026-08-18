# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestHNSWRaBitQ(unittest.TestCase):
    def make_data(self, d=32, nt=400, nb=600, nq=20):
        rs = np.random.RandomState(123)
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")
        return xt, xb, xq

    def make_index(self, metric=faiss.METRIC_L2, nb_bits=3):
        xt, xb, xq = self.make_data()
        if metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(xt)
            faiss.normalize_L2(xb)
            faiss.normalize_L2(xq)
        index = faiss.IndexHNSWRaBitQ(xt.shape[1], 8, nb_bits, metric)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64
        index.train(xt)
        index.add(xb)
        return index, xb, xq

    @staticmethod
    def recall_at_k(actual, expected):
        matches = sum(
            len(set(actual[i]).intersection(expected[i]))
            for i in range(actual.shape[0])
        )
        return matches / actual.size

    def test_staged_search_quality(self):
        index, xb, xq = self.make_index(nb_bits=3)

        faiss.cvar.rabitq_stats.reset()
        D, I = index.search(xq, 10)
        stats = faiss.cvar.rabitq_stats
        self.assertGreater(stats.n_1bit, 0)
        self.assertGreater(stats.n_refine, 0)
        self.assertLess(stats.n_refine, stats.n_1bit)

        exact = faiss.IndexFlatL2(xb.shape[1])
        exact.add(xb)
        _, Iexact = exact.search(xq, 10)
        self.assertGreaterEqual(self.recall_at_k(I, Iexact), 0.8)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_unfinalized_clone_and_io_remain_mutable(self):
        index, xb, xq = self.make_index(nb_bits=3)
        storage = faiss.downcast_index(index.storage)
        storage.centered = True
        Dref, Iref = index.search(xq, 10)

        cloned = faiss.clone_index(index)
        self.assertIsInstance(cloned, faiss.IndexHNSWRaBitQ)
        self.assertIsNotNone(cloned.build_storage)
        Dclone, Iclone = cloned.search(xq, 10)
        np.testing.assert_array_equal(Iclone, Iref)
        np.testing.assert_array_equal(Dclone, Dref)
        cloned.add(xb[:1])
        self.assertEqual(cloned.ntotal, len(xb) + 1)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexHNSWRaBitQ)
        self.assertIsNotNone(loaded.build_storage)
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(loaded_storage.rabitq.nb_bits, 3)
        self.assertTrue(loaded_storage.centered)
        Dloaded, Iloaded = loaded.search(xq, 10)
        np.testing.assert_array_equal(Iloaded, Iref)
        np.testing.assert_array_equal(Dloaded, Dref)
        loaded.add(xb[:1])
        self.assertEqual(loaded.ntotal, len(xb) + 1)

    def test_finalize_io_and_reset(self):
        index, xb, xq = self.make_index(nb_bits=4)
        index.finalize()
        with self.assertRaises(RuntimeError):
            index.add(xb[:1])

        cloned = faiss.clone_index(index)
        self.assertIsNone(cloned.build_storage)
        with self.assertRaises(RuntimeError):
            cloned.add(xb[:1])

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsNone(loaded.build_storage)
        with self.assertRaises(RuntimeError):
            loaded.add(xb[:1])

        loaded.reset()
        loaded.add(xb)
        self.assertEqual(loaded.ntotal, len(xb))
        D, I = loaded.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_one_bit_fallback(self):
        index, _, xq = self.make_index(nb_bits=1)
        self.assertEqual(index.hnsw.search_method, 0)
        faiss.cvar.rabitq_stats.reset()
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertEqual(faiss.cvar.rabitq_stats.n_1bit, 0)
        self.assertEqual(faiss.cvar.rabitq_stats.n_refine, 0)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(loaded_storage.rabitq.nb_bits, 1)
        self.assertEqual(loaded.hnsw.search_method, 0)
        Dloaded, Iloaded = loaded.search(xq, 10)
        np.testing.assert_array_equal(Dloaded, D)
        np.testing.assert_array_equal(Iloaded, I)

    def test_staged_search_requires_bounded_queue(self):
        index, _, xq = self.make_index(nb_bits=3)
        index.hnsw.search_bounded_queue = False
        with self.assertRaises(RuntimeError):
            index.search(xq, 10)

    def test_inner_product_and_reset(self):
        index, xb, xq = self.make_index(
            metric=faiss.METRIC_INNER_PRODUCT, nb_bits=3
        )
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))

        exact = faiss.IndexFlatIP(xb.shape[1])
        exact.add(xb)
        _, Iexact = exact.search(xq, 10)
        self.assertGreaterEqual(self.recall_at_k(I, Iexact), 0.8)

        index.reset()
        self.assertEqual(index.ntotal, 0)
        index.add(xb)
        self.assertEqual(index.ntotal, len(xb))

    def test_permute_keeps_exact_build_storage_aligned(self):
        index, xb, _ = self.make_index(nb_bits=3)
        perm = np.arange(len(xb) - 1, -1, -1, dtype=np.int64)
        index.permute_entries(perm)

        build_storage = faiss.downcast_index(index.build_storage)
        rebuilt = build_storage.reconstruct_n(0, len(xb))
        np.testing.assert_allclose(rebuilt, xb[perm], rtol=0, atol=0)

        index.add(xb[:2])
        self.assertEqual(index.ntotal, len(xb) + 2)

    def test_factory(self):
        default_index = faiss.index_factory(32, "HNSW8,RaBitQ")
        self.assertIsInstance(default_index, faiss.IndexHNSWRaBitQ)
        self.assertFalse(default_index.is_trained)
        default_storage = faiss.downcast_index(default_index.storage)
        self.assertIsInstance(default_storage, faiss.IndexRaBitQ)
        self.assertFalse(default_storage.is_trained)
        self.assertEqual(default_storage.rabitq.nb_bits, 1)
        self.assertEqual(default_index.hnsw.search_method, 0)
        with self.assertRaises(AttributeError):
            default_index.build_storage = None
        with self.assertRaises(RuntimeError):
            default_index.add(np.zeros((1, 32), dtype="float32"))

        four_bit = faiss.index_factory(32, "HNSW8,RaBitQ4")
        four_bit_storage = faiss.downcast_index(four_bit.storage)
        self.assertEqual(four_bit_storage.rabitq.nb_bits, 4)
        self.assertNotEqual(four_bit.hnsw.search_method, 0)

    def test_skip_storage_can_be_reserialized(self):
        index, _, _ = self.make_index(nb_bits=3)
        data = faiss.serialize_index(index, faiss.IO_FLAG_SKIP_STORAGE)
        metadata_only = faiss.deserialize_index(data)
        self.assertIsNone(metadata_only.storage)
        self.assertIsNone(metadata_only.build_storage)

        data_again = faiss.serialize_index(
            metadata_only, faiss.IO_FLAG_SKIP_STORAGE
        )
        metadata_again = faiss.deserialize_index(data_again)
        self.assertIsNone(metadata_again.storage)
        self.assertIsNone(metadata_again.build_storage)

    def test_unsupported_metric_throws(self):
        with self.assertRaises(RuntimeError):
            faiss.IndexHNSWRaBitQ(32, 8, 1, faiss.METRIC_L1)


if __name__ == "__main__":
    unittest.main()
