//Dependencies
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const { Console } = require('console');

(async () => {

    function getMNISTData(){
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


    async function trainModel(xs, ys, xsTests, ysTests){
        //Define Vars
        const model = tf.sequential();
        const learningRate = .01;
        const numberOfEpochs = 100;
        const optimizer = tf.train.adam(learningRate);
        //console.log(xs.shape);
        //console.log(ys.shape);

        //Add Layers
        model.add(tf.layers.dense({
            units:16, activation: "sigmoid", inputShape: [xs.shape[1]]
        }));
        model.add(tf.layers.dense({
            units:10, activation: "softmax"
        }));
        model.compile({
            optimizer: optimizer,
            loss: "categoricalCrossentropy",
            metrics: ["accuracy"]
        });

        //Train
        const history = await model.fit(xs, ys, {
            batchSize: 128,
            epochs: numberOfEpochs,
            //We pass some validation for
            //monitoring validation loss and metrics
            //at the end of each epoch
            validationData: [xsTests, ysTests],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log("Epoch: " + epoch + "Logs:" + logs.loss);
                    console.log("# Tensors: " + tf.memory().numTensors);
                    await tf.nextFrame();
                }
            }
        });


        //Evaluate
        console.log("Evaluating Model...");
        const results = model.evaluate(xsTests, ysTests, batch_size=128)
        console.log("Test Lost, Test Acc: " + results);

        //Save Model
        let save = await model.save('file://myModel1');
        console.log(save);
        console.log("Model Saved");
    }

    async function doPrediction(xsTests, ysTests){
        //Get Stored Model
        const model = await tf.loadLayersModel('file://myModel1/model.json');
        //Get Data Predictions/Test Answers
        //argMax(): Returns the indices of the maximum values of a tensor along axis
        //The result has the same shape as input with the dimension along axis removed.
        const yPred = await model.predict(xsTests).argMax(1).dataSync();
        const yTrue = ysTests.argMax(1).dataSync();
        console.log(yPred);
        console.log(yTrue);

        //Count Correct/Wrong
        let correct = 0;
        let wrong = 0;
        for(let i = 0; i < yTrue.length; i++){
            if(yTrue.length == yPred.length){
                if(yTrue[i] == yPred[i]){
                    correct++;
                } else {
                    wrong++;
                }
            } else {
                console.log("Different Lengths Output and Tests");
                break;
            }
        }
        console.log("Accuracy: " + correct/(correct+wrong));
        console.log("Correct: " + correct + "\nWrong: " + wrong);
    }

    //Get Data
    const [xs, ys, xsTests, ysTests] = getMNISTData();

    // console.log(ys.dataSync());
    // console.log(ysTests.dataSync());

    //await trainModel(xs, ys, xsTests, ysTests);

    await doPrediction(xsTests, ysTests);

})();