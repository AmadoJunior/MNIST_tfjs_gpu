const {getMNISTData3D} = require("./DataHandler");
const tf = require('@tensorflow/tfjs-node-gpu');

//Methods
async function trainModel(xs, ys, xsTests, ysTests) {
    const model = tf.sequential();
    
    const learningRate = .01;
    const numberOfEpochs = 100;
    const optimizer = tf.train.adam(learningRate);

    console.log(xs);
    //xs.print();
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv1d({
      inputShape: [28, 28],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling1d({poolSize: 2, strides: 2}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv1d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling1d({poolSize: 2, strides: 2}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());
  
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
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
  }

//Main
(async () => {
    //Get Data
    const [xs, ys, xsTests, ysTests] = getMNISTData3D();

    // console.log(ys.dataSync());
    // console.log(ysTests.dataSync());

    await trainModel(xs, ys, xsTests, ysTests);
})();