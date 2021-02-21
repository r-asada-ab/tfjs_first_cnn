import * as tf from "@tensorflow/tfjs";

// バッチデータ
export default class BatchData {

    // データ
    public xs: tf.Tensor4D
    // ラベル
    public labels: tf.Tensor2D

    // コンストラクタ
    constructor(xs: tf.Tensor4D, labels: tf.Tensor2D) {
        this.xs = xs
        this.labels = labels
    }
}