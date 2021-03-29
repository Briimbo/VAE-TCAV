import tensorflow as tf


class PosNegAccuracy(tf.metrics.BinaryAccuracy):
    """
    Accuracy for binary labels of -1/+1
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state((y_true+1)/2., (y_pred+1)/2., sample_weight=sample_weight)
