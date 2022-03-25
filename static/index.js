// declare variables
let mobilenet;
let model;
// creates webcam object and point it to DOM object id='wc' (webcam.js)
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var group1Samples=0, group2Samples=0, group3Samples=0, group4Samples=0, group5Samples=0;
let isPredicting = false;


// get model into a object
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  // get one of the output layers from pretrained mobilenet model
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  // we will create new model from  mobilenet and 'conv_pw_13_relu' as output
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


// new DNN that is used to classify with transfer learning
// unlike in Python this is not connected to original model (mobilenet)
// this is separate model that takes ist inputs the output from the previous one (connected)


async function train() {
  // One-hot endoce the labels from the dataset before training
  // well empty the ys to get one hot encoding labels (3 labels - rock, sisers, paper)
  dataset.ys = null;
  dataset.encodeLabels(5);
    // input is the output from truncated mobilenet ()
  model = tf.sequential({
    layers: [
      // first layer is the flattened output from truncated mobilenet model ('conv_pw_13_relu'), so define shape accordingly
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      // the original mobilenet has 1000 classes
      // output layer will be one with 3 units (rock, sisors, paper)
      tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 30,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
        
      }
      
   });

}






// each button has an "id" [0,1,2] (retrain.html)
// we called handleButton(this) - this is a reference to the DOM
function handleButton(elem){
	switch(elem.id){
		case "0":
			group1Samples++;
			document.getElementById("group1samples").innerText = "Group 1 samples:" + group1Samples;
			break;

		case "1":
			group2Samples++;
			document.getElementById("group2samples").innerText = "Group 2 samples:" + group2Samples;
			break;

		case "2":
			group3Samples++;
			document.getElementById("group3samples").innerText = "Group 3 samples:" + group3Samples;
			break;

        case "3":
			group4Samples++;
			document.getElementById("group4samples").innerText = "Group 4 samples:" + group4Samples;
			break;

        case "4":
			group5Samples++;
			document.getElementById("group5samples").innerText = "Group 5 samples:" + group5Samples;
			break;
	}
    // extract label from id by converting it to int
	label = parseInt(elem.id);
    // capture contenct from a webcam
	const img = webcam.capture();
    // we are not adding data from webcam to dataset, 
    // we are adding prediction of that image from mobilenet.
    // We are doing transfer learning by removing bottom layers from mobilenet
    // truncating it so we want it's output to be features on the higher level.
    // If we predict on truncated model 'mobilenet.predict(img)' that is the output we will 
    // get so we can train another NN on thoe features instead of the raw webcam data 
    //and well have a transfer learning.
    // then we pass the 'label'[0,1,2,3,4] to the dataset
    // labels are provided by UI buttons
	dataset.addExample(mobilenet.predict(img), label);

}


// 
async function predict() {
  // if continous predictions are enabled in html
// 1. Predict class
  while (isPredicting) {
    // read a frame from a webcam , use mobilenet to get activationd and, 
    // then get a prediction from mobilenet activation and our retrain model
    // return predictions
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
      
// 2. Update user interface  
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "Group 1";
			break;
		case 1:
			predictionText = "Group 2";
			break;
		case 2:
			predictionText = "Group 3";
			break;
        case 3:
            predictionText = "Group 4";
            break;
        case 4:
			predictionText = "Group 5";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
// 3. clean up    
    predictedClass.dispose();
    //  not to lock up UI thread so the page stays responsive
    await tf.nextFrame();
  }
}



function doTraining() {
	train();
}


// continous predictions - via html button and function call
function startPredicting(){
	isPredicting = true;
	predict();
}

// stop continous predictions - via html button and function call
function stopPredicting(){
	isPredicting = false;
	predict();
}

// Main function
async function init(){
	await webcam.setup();
    // load asynchronously mobilenet
	mobilenet = await loadMobilenet();
    // webcam.capture()  - grabs image from webcam in browser and converts it to tensor
    // mobilenet.predict() returns inferential
    // tf.tidy() - after the function => has finished it cleans all unused tensors exept the return one
	tf.tidy(() => mobilenet.predict(webcam.capture()));	
  
}



init();
