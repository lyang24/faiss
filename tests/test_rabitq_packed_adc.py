# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestPackedRaBitQADC(unittest.TestCase):
    def test_scalar_oracle_and_expanded_equivalence(self):
        rs = np.random.RandomState(5341)
        ids = (16, 0, 7, 2, 16)
        for bits in range(2, 9):
            for d in (1, 7, 16, 17, 32, 33, 65, 128, 768, 960, 1536):
                with self.subTest(bits=bits, d=d):
                    storage = faiss.IndexRaBitQ(d, faiss.METRIC_L2, bits)
                    storage.train(rs.randn(80, d).astype("float32"))
                    center = faiss.vector_to_array(storage.center).copy()
                    xb = rs.randn(17, d).astype("float32")
                    xb[0] = center
                    storage.add(xb)
                    raw = faiss.vector_to_array(storage.codes).reshape(17, -1)
                    offset = (d + 7) // 8 + 12
                    levels = np.empty((len(ids), d), dtype="int64")
                    for row, doc in enumerate(ids):
                        for j in range(d):
                            sign = (int(raw[doc, j // 8]) >> (j % 8)) & 1
                            low = 0
                            for b in range(bits - 1):
                                pos = j * (bits - 1) + b
                                byte = int(raw[doc, offset + pos // 8])
                                low |= ((byte >> (pos % 8)) & 1) << b
                            midpoint = 1 << (bits - 1)
                            levels[row, j] = sign * midpoint + low - midpoint
                    queries = [
                        rs.randn(d).astype("float32"),
                        center.copy(),
                        xb[1].copy(),
                        np.full(d, 1e6, dtype="float32"),
                    ]
                    controls = []
                    storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
                    dc = storage.get_FlatCodesDistanceComputer()
                    for q in queries:
                        dc.set_query(faiss.swig_ptr(q))
                        controls.append([dc(i) for i in ids])
                    del dc
                    storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
                    self.assertEqual(storage.expanded_codes.size(), 0)
                    np.testing.assert_array_equal(
                        faiss.vector_to_array(storage.codes).reshape(raw.shape), raw
                    )
                    np.testing.assert_array_equal(
                        faiss.vector_to_array(storage.center), center
                    )
                    dc = storage.get_FlatCodesDistanceComputer()
                    for q, control in zip(queries, controls):
                        dc.set_query(faiss.swig_ptr(q))
                        actual = np.array([dc(i) for i in ids])
                        np.testing.assert_array_equal(actual, control)
                        residual = (q - center).astype("float64")
                        scale = np.max(np.abs(residual)) / 127 if np.any(residual) else 1
                        # Replicate FP32 query quantization; accumulate the
                        # independent dot in int64 and distance in FP64.
                        quantized = np.clip(
                            np.rint((q - center) / np.float32(scale)), -127, 127
                        ).astype("int64")
                        for row, doc in enumerate(ids):
                            add, rescale = np.frombuffer(
                                raw[doc, -8:].tobytes(), dtype="float32"
                            )
                            dot = int(np.dot(quantized, levels[row]))
                            norm = np.dot(residual, residual)
                            expected = max(
                                0,
                                norm + float(add) + float(rescale)
                                * (np.float32(scale) * dot + 0.5 * residual.sum()),
                            )
                            # Bound FP32 sequential norm/sum rounding, including
                            # cancellation for queries equal to a document.
                            magnitude = (
                                norm + abs(float(add)) + abs(float(rescale))
                                * (scale * np.abs(quantized * levels[row]).sum()
                                   + 0.5 * np.abs(residual).sum())
                            )
                            tolerance = 8 * d * np.finfo("float32").eps
                            self.assertLessEqual(
                                abs(actual[row] - expected),
                                tolerance * max(1, magnitude),
                            )

    def test_hnsw_lifecycle_and_refinement(self):
        rs = np.random.RandomState(777)
        xb = rs.randn(200, 65).astype("float32")
        xq = rs.randn(8, 65).astype("float32")
        for low, high in ((2, 6), (2, 8), (4, 8), (7, 8)):
            index = faiss.IndexHNSWRaBitQ(65, 8, low, faiss.METRIC_L2)
            index.train(xb)
            index.add(xb)
            index.hnsw.efSearch = 100
            index.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
            storage = faiss.downcast_index(index.storage)
            self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
            index.add(xb[:3])
            self.assertEqual(
                storage.full_code_mode, faiss.RABITQ_FULL_CODE_PACKED_INT8
            )
            self.assertEqual(storage.expanded_codes.size(), 0)
            refiner = faiss.IndexRaBitQ(65, faiss.METRIC_L2, high)
            refiner.train(xb)
            refiner.add(np.vstack((xb, xb[:3])))
            refiner.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
            combined = faiss.IndexRefine(index, refiner)
            combined.k_factor = 5
            D, I = combined.search(xq, 10)
            self.assertTrue(np.isfinite(D).all())
            self.assertTrue((I >= 0).all())
            for one, expected in zip(xq, I):
                np.testing.assert_array_equal(
                    combined.search(one[None], 10)[1][0], expected
                )
            cloned = faiss.clone_index(index)
            self.assertEqual(
                faiss.downcast_index(cloned.storage).full_code_mode,
                faiss.RABITQ_FULL_CODE_PACKED_INT8,
            )
            np.testing.assert_array_equal(
                cloned.search(xq, 10)[1], index.search(xq, 10)[1]
            )
            loaded = faiss.deserialize_index(faiss.serialize_index(index))
            self.assertEqual(
                faiss.downcast_index(loaded.storage).full_code_mode,
                faiss.RABITQ_FULL_CODE_PACKED,
            )
            loaded.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
            np.testing.assert_array_equal(
                loaded.search(xq, 10)[1], index.search(xq, 10)[1]
            )
            cloned.reset()
            cloned.add(xb)
            self.assertEqual(
                faiss.downcast_index(cloned.storage).expanded_codes.size(), 0
            )

    def test_flat_lifecycle_and_invalid_modes(self):
        rs = np.random.RandomState(456)
        xb = rs.randn(32, 17).astype("float32")
        storage = faiss.IndexRaBitQ(17, faiss.METRIC_L2, 4)
        storage.train(xb)
        storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
        storage.add(xb)
        storage.remove_ids(faiss.IDSelectorRange(1, 3))
        permutation = np.arange(storage.ntotal - 1, -1, -1, dtype="int64")
        storage.permute_entries(permutation)
        other = faiss.IndexRaBitQ(17, faiss.METRIC_L2, 4)
        other.train(xb)
        other.add(xb[:3])
        storage.merge_from(other)
        self.assertEqual(storage.expanded_codes.size(), 0)
        self.assertEqual(other.ntotal, 0)
        D, I = storage.search(xb[:3], 10)
        storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
        De, Ie = storage.search(xb[:3], 10)
        np.testing.assert_array_equal(D, De)
        np.testing.assert_array_equal(I, Ie)
        storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
        storage.reset()
        storage.add_sa_codes(other.sa_encode(xb[:3]))
        self.assertEqual(storage.ntotal, 3)
        self.assertEqual(storage.expanded_codes.size(), 0)
        for metric, bits in (
            (faiss.METRIC_INNER_PRODUCT, 4),
            (faiss.METRIC_L2, 1),
            (faiss.METRIC_L2, 9),
        ):
            invalid = faiss.IndexRaBitQ(17, metric, bits)
            with self.assertRaises(RuntimeError):
                invalid.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED_INT8)
            self.assertEqual(invalid.full_code_mode, faiss.RABITQ_FULL_CODE_PACKED)

    def test_filtered_low_high_pipeline(self):
        rs = np.random.RandomState(37742)
        for d in (17, 65, 128):
            xb = rs.randn(64, d).astype("float32")
            xq = rs.randn(5, d).astype("float32")
            graph = faiss.IndexHNSWFlat(d, 8)
            graph.add(xb)
            rotation = faiss.RandomRotationMatrix(d, d)
            rotation.init(1234)
            rotated = rotation.apply(xb)
            for low_bits, high_bits in ((2, 6), (2, 8), (4, 6), (4, 8)):
                low = faiss.IndexRaBitQ(d, faiss.METRIC_L2, low_bits)
                high = faiss.IndexRaBitQ(d, faiss.METRIC_L2, high_bits)
                for storage in (low, high):
                    storage.train(rotated)
                    storage.add(rotated)
                nav = faiss.IndexHNSW(low, 8)
                nav.own_fields = False
                nav.ntotal = len(xb)
                nav.is_trained = True
                nav.hnsw = graph.hnsw
                nav.hnsw.search_method = faiss.HNSW.SM_DEFAULT
                refine = faiss.IndexRefine(nav, high)
                pipeline = faiss.IndexPreTransform(rotation, refine)
                order = rs.permutation(len(xb))
                for count in (0, 1, 7, 10, 32, 64):
                    eligible = np.sort(order[:count]).astype("int64")
                    selector = faiss.IDSelectorBatch(eligible)
                    h = faiss.SearchParametersHNSW(efSearch=256, sel=selector)
                    r = faiss.IndexRefineSearchParameters()
                    r.k_factor = 8
                    r.base_index_params = h
                    p = faiss.SearchParametersPreTransform()
                    p.index_params = r
                    outputs = []
                    for mode in (
                        faiss.RABITQ_FULL_CODE_INT8,
                        faiss.RABITQ_FULL_CODE_PACKED_INT8,
                    ):
                        low.set_full_code_mode(mode)
                        high.set_full_code_mode(mode)
                        D, I = pipeline.search(xq, 10, params=p)
                        np.testing.assert_array_equal(
                            (I >= 0).sum(axis=1),
                            np.full(len(xq), min(10, count)),
                        )
                        self.assertTrue(np.isin(I[I >= 0], eligible).all())
                        self.assertTrue(np.isfinite(D[I >= 0]).all())
                        self.assertTrue((I[I < 0] == -1).all())
                        outputs.append((D, I))
                    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
                    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


if __name__ == "__main__":
    unittest.main()
