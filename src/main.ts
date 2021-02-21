import * as tf from "@tensorflow/tfjs";
import MnistData from "./MnistData";
import * as ui from './ui';

// データ
let data: MnistData;

// CNNモデルを作成する
function createConvModel(): tf.Sequential {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
        inputShape: [MnistData.IMAGE_H, MnistData.IMAGE_W, 1],
        kernelSize: 3,
        filters: 16,
        activation: 'relu'
    }))

    model.add(tf.layers.maxPooling2d({
        poolSize: 2, 
        strides: 2
    }))

    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        activation: 'relu'
    }))

    model.add(tf.layers.maxPooling2d({
        poolSize: 2, 
        strides: 2
    }))

    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        activation: 'relu'}));

    model.add(tf.layers.flatten({}))

    model.add(tf.layers.dense({
        units: 64, 
        activation: 'relu'
    }))

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }))

    return model
}

// モデルを訓練する
async function train(model: tf.Sequential, func: (model: tf.Sequential) => {}) {
    ui.logStatus("モデルのトレーニングを開始します")

    const optimizer = 'rmsprop'
    
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy', // ラベルがone-hotの時
        metrics: ['accuracy']
    })

    const batchSize = 320
    const validationSplit = 0.15
    const trainEpochs = 3

    let trainBatchCount = 0

    const trainData = data.getTrainData()
    const testData = data.getTestData()

    const totalNumBatches =
    Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
    trainEpochs;

    let valAcc;
    await model.fit(trainData.xs, trainData.labels, {
        batchSize, 
        validationSplit,
        epochs: trainEpochs,
        callbacks: {
            onBatchEnd: async (batch: number, logs: tf.Logs) => {
                trainBatchCount++
                ui.logStatus(
                    `訓練中です(` + 
                    `${( trainBatchCount / totalNumBatches * 100).toFixed(1)}%)`
                )
                ui.plotLoss(trainBatchCount, logs.loss, 'train');
                ui.plotAccuracy(trainBatchCount, logs.acc, 'train')
                if (batch % 10 === 0) {
                    func(model)
                }
                await tf.nextFrame()
            },
            onEpochEnd: async (epochs: number, logs: tf.Logs) => {
                valAcc = logs.val_acc
                ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
                ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation')
                func(model)
                await tf.nextFrame()
            }
        }
    })

    const testResult = model.evaluate(testData.xs, testData.labels)
    const testAccPercent = testResult[1].dataSync()[0] * 100
    const finalValAccPercent = valAcc * 100
    ui.logStatus(
        `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
        `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

async function showPrediction(model: tf.Sequential) {
    const testExamples = 50
    const examples = data.getTestData(testExamples)

    tf.tidy(() => {
        const output = <tf.Tensor>model.predict(examples.xs)
        
        const axis = 1
        const labels = Array.from(examples.labels.argMax(axis).dataSync())
        const predictions = Array.from(output.argMax(axis).dataSync());

        ui.showTestResults(examples, predictions, labels)
    })

}

async function load() {
    data = new MnistData()
    await data.load()
}

function main() {
    let button = document.getElementById("train")
    button.addEventListener("click", async () => {
        ui.logStatus("MNISTデータセットをロードします")
        await load()

        ui.logStatus("モデルを作成します")
        const model = createConvModel()
        model.summary()

        ui.logStatus("モデルの訓練を開始します")
        await train(model, showPrediction)
    })
}

main()