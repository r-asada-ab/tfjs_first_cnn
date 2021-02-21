import * as tf from "@tensorflow/tfjs";
import * as tfvis from '@tensorflow/tfjs-vis'
import BatchData from "./BatchData";

// 状態を表示する要素
const statusElement = document.getElementById("status")
// App内からのメッセージを表示する要素
const messageElement = document.getElementById("message")
// テストデータを表示する画像群の要素
const imagesElement = document.getElementById("images")

// ログを表示する
export function logStatus(message: string) {
    statusElement.innerText = message
}

// 訓練ログを出力する
export function trainingLog(message: string) {
    messageElement.innerText = `${message}\n`
}

// テスト結果を出力する
export function showTestResults(batch: BatchData, predictions: any[],
     labels: any[]) {
    const testExmamples = batch.xs.shape[0]
    imagesElement.innerHTML = ''

    for (let i = 0; i < testExmamples; i++) {
        const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]])

        // コンテナ
        const div = document.createElement("div")
        div.className = 'pred-container'

        // 画像
        const canvas = document.createElement('canvas')
        canvas.className = "prediction-canvas"
        draw(image.flatten(), canvas)

        // 予測結果
        const pred = document.createElement("div")
        const prediction = predictions[i]
        const label = labels[i]
        const correct = prediction === label // 予測結果とラベルが一致しているか  

        pred.className = `pred ${ (correct ? 'pred-correct': 'pred-incorrect')}`
        pred.innerText = `pred: ${ prediction }`

        div.appendChild(pred)
        div.appendChild(canvas)

        imagesElement.appendChild(div)
    }
}

// 描画する
export function draw(image: tf.Tensor1D, canvas: HTMLCanvasElement) {
    const [width, height] = [28, 28]
    canvas.width = width
    canvas.height = height 

    const ctx = canvas.getContext('2d')
    const imageData = new ImageData(width, height)
    const data = image.dataSync()

    for (let i = 0; i < height * width; ++i) {
        const j = i * 4
        imageData.data[j + 0] = data[i] * 255 // R
        imageData.data[j + 1] = data[i] * 255 // G
        imageData.data[j + 2] = data[i] * 255 // B
        imageData.data[j + 3] = 255           // A
    }
    ctx.putImageData(imageData, 0, 0)
}

//  損失を図示する
const lossLabelElement = document.getElementById('loss-label')
const lossValues = [[], []]
export function plotLoss(batchCount: number, loss: number, set: string) {
    if (set == 'train') {
        const p0 = {x: batchCount, y: loss}
        lossValues[0].push(p0)
        const p1 = {x: batchCount, y: 0}
        lossValues[1].push(p1)
    } else {
        const p0 = {x: batchCount, y: 0}
        lossValues[0].push(p0)
        const p1 = {x: batchCount, y: loss}
        lossValues[1].push(p1)
    }

    const lossContainer = document.getElementById('loss-canvas')
    tfvis.render.linechart(
        lossContainer, {values: lossValues, series: ['train', 'validation']}, 
        {
            xLabel: 'Batch #', 
            yLabel: 'Loss',
            width: 400, 
            height: 300
        })
    lossLabelElement.innerText = `last loss: ${ loss.toFixed(3) }`
}

// 正確性を図示する
const accuracyLabelElement = document.getElementById('accuracy-label')
const accuracyValues = [[], []]
export function plotAccuracy(batchCount: number, accuracy: number, set) {
    const accuracyContainer = document.getElementById('accuracy-canvas')
    if (set == 'train') {
        const p0 = {x: batchCount, y: accuracy}
        accuracyValues[0].push(p0)
        const p1 = {x: batchCount, y: 0}
        accuracyValues[1].push(p1)
    } else {
        const p0 = {x: batchCount, y: 0}
        accuracyValues[0].push(p0)
        const p1 = {x: batchCount, y: accuracy}
        accuracyValues[1].push(p1)
    }
    tfvis.render.linechart(
        accuracyContainer, 
        { // データ
            values: accuracyValues, series: ['train', 'validation']
        }, 
        { // オプション
            xLabel: 'Batch #',
            yLabel: 'Accuracy',
            width: 400, 
            height: 300
        }
    )
    const value = (accuracy * 100).toFixed(1)
    accuracyLabelElement.innerText = `last accuracy: ${ value }`
}
