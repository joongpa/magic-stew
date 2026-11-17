"""Standard Ultralytics single-keypoint trainer with named class transfer."""

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel

from .typed_sticker_keypoints import CLASS_NAMES


class TypedStickerTrainer(PoseTrainer):
    """Use the native loss and set names before loading pretrained weights.

    Matching landmark classifier rows can transfer by name, alongside the
    backbone, neck and compatible one-keypoint head weights. No custom pose
    loss or symmetry routing is needed for independent one-point objects.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        names = self.data["names"]
        if tuple(names[i] for i in range(len(names))) != CLASS_NAMES or list(
            self.data["kpt_shape"]
        ) != [1, 3]:
            raise ValueError("Expected the 21-class, single-keypoint dataset")
        model = PoseModel(
            cfg, ch=self.data.get("channels", 3), nc=21, data_kpt_shape=[1, 3], verbose=verbose
        )
        model.args = self.args
        model.names = dict(enumerate(CLASS_NAMES))
        if weights:
            model.load(weights)
        return model
