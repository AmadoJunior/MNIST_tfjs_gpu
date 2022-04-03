const {getMNISTData2D} = require("./DataHandler");
const tf = require('@tensorflow/tfjs-node-gpu');

//Methods
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
        units:300, activation: "relu", inputShape: [xs.shape[1]]
    }));
    model.add(tf.layers.dense({
        units:100, activation: "relu"
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

//Main
(async () => {
    //Get Data
    const [xs, ys, xsTests, ysTests] = getMNISTData2D();

    // console.log(ys.dataSync());
    // console.log(ysTests.dataSync());

    await trainModel(xs, ys, xsTests, ysTests);

    await doPrediction(xsTests, ysTests);

})();