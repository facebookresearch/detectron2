import unittest
import torch

from detectron2.layers.soft_nms import (
    batched_soft_nms,
    batched_soft_nms_rotated,
    soft_nms,
    soft_nms_rotated,
)


def keeps_are_equal(keep1, keep2, tolerance):
    """ Due to floating point precision, sometimes the positions of two kept boxes are reversed
    """

    equality_matrix = keep1[:, None] == keep2[None, :]
    equality_indices = equality_matrix.nonzero()
    difference = equality_indices[0] - equality_indices[0]
    max_abs_dif = difference.abs().max()
    return max_abs_dif < tolerance


class TestSoftNMS(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["gaussian", "linear", "hard"]

    def test_single_box(self):
        boxes = torch.tensor([[10, 10, 15, 15]], dtype=torch.float, device=self.device)
        scores = torch.tensor([1.0], device=self.device)
        for method in self.methods:
            keep, new_scores = soft_nms(boxes, scores, method, 0.5, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0], device=self.device)
            ), "Single box not kept for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Single box score not kept for soft nms method {}".format(method)

    def test_two_separate_boxes(self):
        boxes = torch.tensor([[10, 10, 15, 15], [20, 20, 25, 25]], dtype=torch.float)
        scores = torch.tensor([1.0, 0.99])
        for method in self.methods:
            keep, new_scores = soft_nms(boxes, scores, method, 0.5, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0, 1])
            ), "Separate boxes not kept for soft nms method {}".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Separate boxes scores not kept for soft nms method {}".format(method)

    def test_two_full_overlap(self):
        boxes = torch.tensor([[10, 10, 15, 15], [10, 10, 15, 15]], dtype=torch.float)
        scores = torch.tensor(([1.0, 0.4]))

        for method in self.methods:
            keep, new_scores = soft_nms(boxes, scores, method, 0.05, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0])
            ), "Box not suppressed properly for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores[:1]
            ), "Scores not kept correctly for soft nms method {}".format(method)

    def test_no_boxes(self):
        boxes = torch.tensor([], dtype=torch.float).reshape(0, 4)
        scores = torch.tensor(([]))
        for method in self.methods:
            keep, new_scores = soft_nms(boxes, scores, method, 0.5, 0.3, 0.001)
            assert keep.size()[0] == 0, "Soft nms failed for method {}".format(method)
            assert new_scores.size()[0] == 0, "Soft nms failed for method {}".format(method)

    def test_batched_single_box(self):
        boxes = torch.tensor([[10, 10, 15, 15], [10, 10, 15, 15]], dtype=torch.float)
        scores = torch.tensor([1.0, 0.4])
        category_idxs = torch.tensor([0, 1])
        for method in self.methods:
            keep, new_scores = batched_soft_nms(
                boxes, scores, category_idxs, method, 0.5, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 1])
            ), "Single box not kept for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Single box score not kept for soft nms method {}".format(method)

    def test_batched_two_separate_boxes(self):
        boxes = torch.tensor(
            [[10, 10, 15, 15], [20, 20, 25, 25], [10, 10, 15, 15], [20, 20, 25, 25]],
            dtype=torch.float,
        )
        scores = torch.tensor([1.0, 0.99, 0.98, 0.97])
        category_idxs = torch.tensor([0, 0, 1, 1])
        for method in self.methods:
            keep, new_scores = batched_soft_nms(
                boxes, scores, category_idxs, method, 0.5, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 1, 2, 3])
            ), "Separate boxes not kept for soft nms method {}".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Separate boxes scores not kept for soft nms method {}".format(method)

    def test_batched_two_full_overlap(self):
        boxes = torch.tensor(
            [[10, 10, 15, 15], [10, 10, 15, 15], [10, 10, 15, 15], [10, 10, 15, 15]],
            dtype=torch.float,
        )
        scores = torch.tensor(([1.0, 0.4, 0.99, 0.4]))
        category_idxs = torch.tensor([0, 0, 1, 1])
        for method in self.methods:
            keep, new_scores = batched_soft_nms(
                boxes, scores, category_idxs, method, 0.05, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 2])
            ), "Box not suppressed properly for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores[[0, 2]]
            ), "Scores not kept correctly for soft nms method {}".format(method)


class TestSoftNMSRotated(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["gaussian", "linear", "hard"]

    def _create_tensors(self, N):
        boxes = torch.rand(N, 4, device=self.device) * 100
        # Note: the implementation of this function in torchvision is:
        # boxes[:, 2:] += torch.rand(N, 2) * 100
        # but it does not guarantee non-negative widths/heights constraints:
        # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.rand(N, device=self.device)
        return boxes, scores

    def test_single_box(self):
        boxes = torch.tensor([[10, 10, 5, 5, 0]], dtype=torch.float)
        scores = torch.tensor([1.0])
        for method in self.methods:
            keep, new_scores = soft_nms_rotated(boxes, scores, method, 0.5, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0])
            ), "Single box not kept for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Single box score not kept for soft nms method {}".format(method)

    def test_two_separate_boxes(self):
        boxes = torch.tensor(
            [[10, 10, 5, 5, 0], [20, 20, 5, 5, 0]], dtype=torch.float, device=self.device
        )
        scores = torch.tensor([1.0, 0.99], device=self.device)
        for method in self.methods:
            keep, new_scores = soft_nms_rotated(boxes, scores, method, 0.5, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0, 1], device=self.device)
            ), "Separate boxes not kept for soft nms method {}".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Separate boxes scores not kept for soft nms method {}".format(method)

    def test_two_full_overlap(self):
        boxes = torch.tensor(
            [[10, 10, 5, 5, 0], [10, 10, 5, 5, 0]], dtype=torch.float, device=self.device
        )
        scores = torch.tensor([1.0, 0.4], device=self.device)
        for method in self.methods:
            keep, new_scores = soft_nms_rotated(boxes, scores, method, 0.05, 0.3, 0.001)
            assert torch.equal(
                keep, torch.tensor([0], device=self.device)
            ), "Box not suppressed properly for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores[:1]
            ), "Scores not kept correctly for soft nms method {}".format(method)

    def test_batched_single_box(self):
        boxes = torch.tensor(
            [[10, 10, 5, 5, 0], [10, 10, 5, 5, 0]], dtype=torch.float, device=self.device
        )
        scores = torch.tensor([1.0, 0.4], device=self.device)
        category_idxs = torch.tensor([0, 1], device=self.device)
        for method in self.methods:
            keep, new_scores = batched_soft_nms_rotated(
                boxes, scores, category_idxs, method, 0.5, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 1], device=self.device)
            ), "Single box not kept for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Single box score not kept for soft nms method {}".format(method)

    def test_batched_two_separate_boxes(self):
        boxes = torch.tensor(
            [[10, 10, 5, 5, 0], [20, 20, 5, 5, 0], [10, 10, 5, 5, 0], [20, 20, 5, 5, 0]],
            dtype=torch.float,
            device=self.device,
        )
        scores = torch.tensor([1.0, 0.99, 0.98, 0.97], device=self.device)
        category_idxs = torch.tensor([0, 0, 1, 1], device=self.device)
        for method in self.methods:
            keep, new_scores = batched_soft_nms_rotated(
                boxes, scores, category_idxs, method, 0.5, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 1, 2, 3], device=self.device)
            ), "Separate boxes not kept for soft nms method {}".format(method)
            assert torch.equal(
                new_scores, scores
            ), "Separate boxes scores not kept for soft nms method {}".format(method)

    def test_batched_two_full_overlap(self):
        boxes = torch.tensor(
            [[10, 10, 5, 5, 0], [10, 10, 5, 5, 0], [10, 10, 5, 5, 0], [10, 10, 5, 5, 0]],
            dtype=torch.float,
            device=self.device,
        )
        scores = torch.tensor([1.0, 0.4, 0.99, 0.4], device=self.device)
        category_idxs = torch.tensor([0, 0, 1, 1], device=self.device)
        for method in self.methods:
            keep, new_scores = batched_soft_nms_rotated(
                boxes, scores, category_idxs, method, 0.05, 0.3, 0.001
            )
            assert torch.equal(
                keep, torch.tensor([0, 2], device=self.device)
            ), "Box not suppressed properly for soft nms method {}.".format(method)
            assert torch.equal(
                new_scores, scores[[0, 2]]
            ), "Scores not kept correctly for soft nms method {}".format(method)

    def test_batched_nms_rotated_0_degree(self):
        # torch.manual_seed(0)
        N = 2000
        num_classes = 50
        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,))
        rotated_boxes = torch.zeros(N, 5, device=self.device)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        err_msg = (
            "Rotated Soft NMS with 0 degree is incompatible with horizontal Soft NMS "
            "for method={}, gaussian_sigma={}, linear_threshold={}"
        )
        for method in self.methods:
            for gaussian_sigma in [0.5, 1.0, 2.0]:
                for linear_threshold in [0.2, 0.5, 0.8]:
                    backup = boxes.clone()
                    keep_ref, _ = batched_soft_nms(
                        boxes, scores, idxs, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    assert torch.allclose(boxes, backup), "boxes modified by batched_soft_nms"
                    backup = rotated_boxes.clone()
                    keep, _ = batched_soft_nms_rotated(
                        rotated_boxes, scores, idxs, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    assert torch.allclose(
                        rotated_boxes, backup
                    ), "rotated_boxes modified by batched_soft_nms_rotated"
                    assert keeps_are_equal(keep, keep_ref, 1), err_msg.format(
                        method, gaussian_sigma, linear_threshold
                    )

    def test_soft_nms_rotated_90_degrees(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5, device=self.device)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        # Note for rotated_boxes[:, 2] and rotated_boxes[:, 3]:
        # widths and heights are intentionally swapped here for 90 degrees case
        # so that the reference horizontal nms could be used
        rotated_boxes[:, 2] = boxes[:, 3] - boxes[:, 1]
        rotated_boxes[:, 3] = boxes[:, 2] - boxes[:, 0]

        rotated_boxes[:, 4] = torch.ones(N) * 90
        err_msg = (
            "Rotated Soft NMS with 90 degree is incompatible with horizontal Soft NMS "
            "for method={}, gaussian_sigma={}, linear_threshold={}"
        )
        for method in self.methods:
            for gaussian_sigma in [0.5, 1.0, 2.0]:
                for linear_threshold in [0.2, 0.5, 0.8]:
                    keep_ref, _ = soft_nms(
                        boxes, scores, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    keep, _ = soft_nms_rotated(
                        rotated_boxes, scores, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    assert keeps_are_equal(keep, keep_ref, 1), err_msg.format(
                        method, gaussian_sigma, linear_threshold
                    )

    def test_soft_nms_rotated_180_degrees(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5, device=self.device)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        rotated_boxes[:, 4] = torch.ones(N) * 180
        err_msg = (
            "Rotated Soft NMS with 180 degree is incompatible with horizontal Soft NMS "
            "for method={}, gaussian_sigma={}, linear_threshold={}"
        )
        for method in self.methods:
            for gaussian_sigma in [0.5, 1.0, 2.0]:
                for linear_threshold in [0.2, 0.5, 0.8]:
                    keep_ref, _ = soft_nms(
                        boxes, scores, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    keep, _ = soft_nms_rotated(
                        rotated_boxes, scores, method, gaussian_sigma, linear_threshold, 0.001
                    )
                    assert keeps_are_equal(keep, keep_ref, 1), err_msg.format(
                        method, gaussian_sigma, linear_threshold
                    )


if __name__ == "__main__":
    unittest.main()
