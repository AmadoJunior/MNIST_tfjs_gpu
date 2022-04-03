const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');

function getMNISTData2D(){
    //Load Data
    var trainData  = fs.readFileSync(__dirname + '\\MNISTData\\mnist_train.csv', {encoding: "utf8"}).toString().split('\n').slice(1);
    var testData = fs.readFileSync(__dirname + '\\MNISTData\\mnist_test.csv', {encoding: "utf8"}).toString().split('\n').slice(1);
    let xTrains = [];
    let yTrains = [];
    let xTests = [];
    let yTests = [];

    // console.log(testData.length);
    // console.log(trainData.length);

    for(let i = 0; i < trainData.length-1; i++){
        let values = trainData[i].split(",");
        //console.log(values.length);
        yTrains.push(parseInt(values[0]));
        xTrains[i] = [];
        for(let j = 1; j < values.length; j++){
            xTrains[i].push((parseInt(values[j]))/255);
        }
    }

    for(let i = 0; i < testData.length-1; i++){
        let values = testData[i].split(",");
        //console.log(values.length);
        yTests.push(parseInt(values[0]));
        xTests[i] = [];
        for(let j = 1; j < values.length; j++){
            xTests[i].push((parseInt(values[j]))/255);
        }
    }

    //console.log(xTrains);
    //console.log(yTrains);
    return tf.tidy(() => {
        const xs = tf.tensor2d(xTrains);
        const ys = tf.oneHot(tf.tensor1d(yTrains).toInt(), 10);
        const xsTests = tf.tensor2d(xTests);
        const ysTests = tf.oneHot(tf.tensor1d(yTests).toInt(), 10);
        return [xs, ys, xsTests, ysTests];
    });
}

function getMNISTData3D(){
    //Load Data
    var trainData  = fs.readFileSync(__dirname + '\\MNISTData\\mnist_train.csv', {encoding: "utf8"}).toString().split('\n').slice(1);
    var testData = fs.readFileSync(__dirname + '\\MNISTData\\mnist_test.csv', {encoding: "utf8"}).toString().split('\n').slice(1);
    let xTrains = [];
    let yTrains = [];
    let xTests = [];
    let yTests = [];

    // console.log(testData.length);
    // console.log(trainData.length);

    for(let i = 0; i < trainData.length-1; i++){
        let values = trainData[i].split(",");
        //console.log(values.length);
        yTrains.push(parseInt(values[0]));
        xTrains[i] = [];
        for(let k = 0; k < 28; k++){
            xTrains[i].push([]);
            for(let h = 0; h < 28; h++){
                xTrains[i][k].push((parseInt(values[k+h]))/255);
            }
        }
    }

    for(let i = 0; i < testData.length-1; i++){
        let values = testData[i].split(",");
        //console.log(values.length);
        yTests.push(parseInt(values[0]));
        xTests[i] = [];
        for(let k = 0; k < 28; k++){
            xTests[i].push([]);
            for(let h = 0; h < 28; h++){
                xTests[i][k].push((parseInt(values[k+h]))/255);
            }
        }
    }

    //console.log(xTrains);
    //console.log(yTrains);
    return tf.tidy(() => {
        const xs = tf.tensor3d(xTrains);
        const ys = tf.oneHot(tf.tensor1d(yTrains).toInt(), 10);
        const xsTests = tf.tensor3d(xTests);
        const ysTests = tf.oneHot(tf.tensor1d(yTests).toInt(), 10);
        return [xs, ys, xsTests, ysTests];
    });
}

module.exports = {
    getMNISTData2D,
    getMNISTData3D
}