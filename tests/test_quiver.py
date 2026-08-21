# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import faiss


class TestQuiverCodec(unittest.TestCase):
    def test_sign_magnitude_distance(self):
        index = faiss.IndexQuiver(4)
        xb = np.array(
            [
                [2.0, 1.0, -2.0, -1.0],
                [-2.0, 1.0, 2.0, -1.0],
            ],
            dtype="float32",
        )
        index.add(xb)

        distances, labels = index.search(xb[:1], 2)
        np.testing.assert_array_equal(labels, [[0, 1]])
        # Dimensions 0 and 2 have different signs and are strong in both
        # vectors, so each contributes the maximum penalty of four.
        np.testing.assert_array_equal(distances, [[0.0, -8.0]])

    def test_code_size_and_roundtrip_reconstruction(self):
        d = 13
        index = faiss.IndexQuiver(d)
        self.assertEqual(index.plane_size(), 2)
        self.assertEqual(index.sa_code_size(), 4)

        rng = np.random.RandomState(123)
        xb = rng.randn(7, d).astype("float32")
        index.add(xb)
        decoded = index.reconstruct_n(0, len(xb))
        self.assertTrue(np.isfinite(decoded).all())
        np.testing.assert_allclose(
            np.linalg.norm(decoded, axis=1), np.ones(len(xb)), atol=1e-6
        )

    def test_inner_product_only(self):
        with self.assertRaises(RuntimeError):
            faiss.IndexQuiver(32, faiss.METRIC_L2)


class TestHNSWQuiver(unittest.TestCase):
    def make_data(self, d=64, nb=2000, nq=100):
        rng = np.random.RandomState(123)
        centers = rng.randn(20, d).astype("float32")
        faiss.normalize_L2(centers)
        assignments = rng.randint(0, len(centers), size=nb)
        xb = centers[assignments] + 0.08 * rng.randn(nb, d).astype("float32")
        faiss.normalize_L2(xb)
        xq = xb[:nq] + 0.02 * rng.randn(nq, d).astype("float32")
        faiss.normalize_L2(xq)
        return xb, xq

    def test_factory_builds_in_quantized_space(self):
        d = 64
        index = faiss.index_factory(
            d, "HNSW16,Quiver", faiss.METRIC_INNER_PRODUCT
        )
        self.assertIsInstance(index, faiss.IndexHNSW)
        self.assertIsInstance(faiss.downcast_index(index.storage), faiss.IndexQuiver)

        xb, xq = self.make_data(d=d)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        index.add(xb)
        distances, labels = index.search(xq, 10)
        self.assertTrue(np.isfinite(distances).all())
        self.assertTrue((labels >= 0).all())

    def test_refine_returns_exact_scores(self):
        d = 64
        index = faiss.index_factory(
            d, "HNSW16,Quiver,RFlat", faiss.METRIC_INNER_PRODUCT
        )
        index.k_factor = 8
        base = faiss.downcast_index(index.base_index)
        base.hnsw.efConstruction = 80
        base.hnsw.efSearch = 80

        xb, xq = self.make_data(d=d)
        index.add(xb)
        distances, labels = index.search(xq, 10)
        expected = np.take_along_axis(xq @ xb.T, labels, axis=1)
        np.testing.assert_allclose(distances, expected, atol=1e-6)

    def test_clone_and_io(self):
        d = 32
        xb, xq = self.make_data(d=d, nb=300, nq=10)
        index = faiss.index_factory(
            d, "HNSW8,Quiver", faiss.METRIC_INNER_PRODUCT
        )
        index.add(xb)
        index.hnsw.efSearch = 32
        expected = index.search(xq, 5)

        cloned = faiss.clone_index(index)
        actual_clone = cloned.search(xq, 5)
        np.testing.assert_array_equal(expected[1], actual_clone[1])
        np.testing.assert_array_equal(expected[0], actual_clone[0])

        serialized = faiss.serialize_index(index)
        restored = faiss.deserialize_index(serialized)
        actual_restored = restored.search(xq, 5)
        np.testing.assert_array_equal(expected[1], actual_restored[1])
        np.testing.assert_array_equal(expected[0], actual_restored[0])


if __name__ == "__main__":
    unittest.main()
