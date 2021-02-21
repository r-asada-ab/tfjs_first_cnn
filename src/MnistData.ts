import * as tf from '@tensorflow/tfjs';
import BatchData from './BatchData';

// MNISTデータセットを扱うクラス
export default class MnistData {

    // 画像の幅
    public static IMAGE_H = 28
    // 画像の高さ
    public static IMAGE_W = 28

    // 画像サイズ
    private static IMAGE_SIZE = MnistData.IMAGE_H * MnistData.IMAGE_W
    // 画像の分類数(10個に分類する)
    private static NUM_CLASSES = 10
    // データセットの要素数
    private static NUM_DATASET_ELEMENTS = 65000
    // 訓練データの要素数
    private static NUM_TRAIN_ELEMENTS = 55000
    // テストデータの要素数
    private static NUM_TEST_ELEMENTS = 
        MnistData.NUM_DATASET_ELEMENTS - MnistData.NUM_TRAIN_ELEMENTS
    // MNISTの画像スプライトのパス
    private static MNIST_IMAGES_SPRITE_PATH =
        'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png'
    // MNISTのラベルへのパス
    private static MNIST_LABELS_PATH =
        'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'


    // データセット(画像)
    private datasetImages: Float32Array
    // データセット(ラベル)
    private datasetLabels: Uint8Array
    // 訓練データセット(画像)
    private trainImages: Float32Array
    // 訓練データセット(ラベル)
    private trainLabels: Uint8Array
    // テストデータセット(画像)
    private testImages: Float32Array
    // テストデータセット(ラベル)
    private testLabels: Uint8Array

    // コンストラクタ
    constructor() {}

    // データをロードする
    public async load() {
        const img = new Image()
        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")

        // 画像データの取得
        const imgRequest = new Promise<void>((resolve, reject) => {
            img.crossOrigin = ''
            img.onload = () => {
                img.width = img.naturalWidth
                img.height = img.naturalHeight

                const size = MnistData.NUM_DATASET_ELEMENTS * MnistData.IMAGE_SIZE * 4
                const datasetByteBuffer = new ArrayBuffer(size)

                const chunkSize = 5000
                canvas.width = img.width 
                canvas.height = chunkSize

                for (let i = 0; i < MnistData.NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const offset = i * MnistData.IMAGE_SIZE * chunkSize * 4
                    const length = MnistData.IMAGE_SIZE * chunkSize
                    const datasetBytesView = new Float32Array(datasetByteBuffer, offset, length)
                
                    ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 
                        0, 0, img.width, chunkSize)
                    
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        datasetBytesView[j] = imageData.data[j * 4] / 255
                    }
                }
                this.datasetImages = new Float32Array(datasetByteBuffer)

                resolve()
            }
            img.src = MnistData.MNIST_IMAGES_SPRITE_PATH
        })

        // ラベルの取得
        const labelRequest = fetch(MnistData.MNIST_LABELS_PATH)
        const [imgResponse, labelResponce] = await Promise.all([imgRequest,
             labelRequest])
        
        this.datasetLabels = new Uint8Array(await labelResponce.arrayBuffer())

        // 画像を訓練用とテスト用に分ける
        // 色値は使用していないので4要素分取る必要はない
        const trainImgEnd = MnistData.IMAGE_SIZE * MnistData.NUM_TRAIN_ELEMENTS
        this.trainImages = this.datasetImages.slice(0, trainImgEnd)
        this.testImages = this.datasetImages.slice(trainImgEnd)

        // ラベルを訓練用とテスト用に分ける
        const trainLabelEnd = MnistData.NUM_CLASSES * MnistData.NUM_TRAIN_ELEMENTS
        this.trainLabels = this.datasetLabels.slice(0, trainLabelEnd)
        this.testLabels = this.datasetLabels.slice(trainLabelEnd)
    }

    // 訓練データを取得する
    public getTrainData(): BatchData {
        const ni = this.trainImages.length / MnistData.IMAGE_SIZE
        const xs = tf.tensor4d(
            this.trainImages,
            [ni, MnistData.IMAGE_H, MnistData.IMAGE_W, 1]
        )
        const nl = this.trainLabels.length / MnistData.NUM_CLASSES
        const labels = tf.tensor2d(
            this.trainLabels, 
            [nl, MnistData.NUM_CLASSES]
        )
        return new BatchData(xs, labels)
    }

    // テストデータを取得する
    public getTestData(numExamples: number | null = null): BatchData {
        const ni = this.testImages.length / MnistData.IMAGE_SIZE
        let xs = tf.tensor4d(
            this.testImages,
            [ni, MnistData.IMAGE_H, MnistData.IMAGE_W, 1]
        )
        const nl = this.testLabels.length / MnistData.NUM_CLASSES
        let labels = tf.tensor2d(
            this.testLabels, 
            [nl, MnistData.NUM_CLASSES]
        )

        if (numExamples != null) {
            xs = xs.slice([0, 0, 0, 0],
                 [numExamples, MnistData.IMAGE_H, MnistData.IMAGE_W, 1]);
            labels = labels.slice([0, 0], [numExamples, MnistData.NUM_CLASSES]);
        }
        return new BatchData(xs, labels)
    }
}