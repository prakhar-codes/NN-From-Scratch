public class ANN {

    //Dummified Inputs and Outputs
    double trainInput[][];
    double trainOutput[][];
    Layer inputLayer, outputLayer;
    Layer hiddenLayers[];
    
    public ANN() {
        inputLayer = new Layer();
        outputLayer = new Layer();
        inputLayer.nextLayer = outputLayer;
        outputLayer.prevLayer = inputLayer;
    }

    public void setInput(double input[][]) {
        System.out.println("Adding Input...");
        int input_size = input[0].length;
        trainInput = input;
        addInputLayer(input_size); 
    }

    public void setOutput(double output[][]) {
        System.out.println("Adding Output...");
        int output_size = output[0].length;
        trainOutput = output;
        addOutputLayer(output_size); 
    }

    private void addInputLayer(int input_size) {
        inputLayer.createNodes(input_size);     
    }

    private void addOutputLayer(int output_size) {
        outputLayer.createNodes(output_size);     
    }

    public void addDenseLayer(int numOfLayers, int nodesInLayer) {
        System.out.println("Adding Dense Layer..");
        hiddenLayers = new Layer[numOfLayers];

        for(int i=0; i<numOfLayers; i++) {
            Layer currentLayer = new Layer();
            hiddenLayers[i] = currentLayer;
            currentLayer.createNodes(nodesInLayer);
            if(i==0) {
                currentLayer.prevLayer = inputLayer;
                inputLayer.nextLayer = currentLayer;
            }
            else {
                currentLayer.prevLayer = hiddenLayers[i-1];
                hiddenLayers[i-1].nextLayer = currentLayer;
            }
        }

        hiddenLayers[numOfLayers-1].nextLayer = outputLayer;
        outputLayer.prevLayer = hiddenLayers[numOfLayers-1];
        System.out.println("Setting weights...");
        setWeights(inputLayer.nextLayer);
    }

    private void setWeights(Layer layer) {
        if (layer!=null) {
            Node nodes[] = layer.nodes;
            for(int i=0; i<nodes.length; i++) {
                nodes[i].weights = new double[layer.prevLayer.nodes.length];
                nodes[i].initializeWeights();
            }
            setWeights(layer.nextLayer);
        }
    }

    private double calculateLoss(int labelNumber) {
        double loss = 0.0;
        for(int i=0; i<trainOutput[labelNumber].length; i++) {
            loss += Math.pow(trainOutput[labelNumber][i] - outputLayer.nodes[i].activatedValue, 2);
        }
        return loss/trainOutput[labelNumber].length;
    }

    private void setGradients(Layer layer, int labelNumber) {
        if (layer!=inputLayer) {
            for(int i=0; i<layer.nodes.length; i++) {
                Node node = layer.nodes[i];
                if(layer!=outputLayer) {
                    int numOfGradients = layer.nextLayer.nodes.length;
                    node.gradients = new double[numOfGradients];
                    for(int j=0; j<numOfGradients; j++) {
                        node.gradients[j] = node.getActivationDerivative() * layer.nextLayer.nodes[j].weights[i] * layer.nextLayer.nodes[j].getSumOfGradients();
                    }
                    
                } else {
                    double gradient = 2.0*(node.activatedValue - trainOutput[labelNumber][i]) * node.getActivationDerivative() / layer.nodes.length;
                    node.gradients = new double[1];
                    node.gradients[0] = gradient;

                }
            }
            setGradients(layer.prevLayer, labelNumber);
        } 
    }

    private void setNewWeights(Layer layer, int labelNumber, double learningRate) {
        if(layer!=inputLayer) {
            Node nodes[] = layer.nodes;
            for(int i=0; i<nodes.length; i++) {
                for(int j=0; j<nodes[i].weights.length; j++) {
                    double oldWeight = nodes[i].weights[j];
                    double dy_dw = layer.prevLayer.nodes[j].activatedValue;
                    double newWeight = oldWeight - learningRate*dy_dw*nodes[i].getSumOfGradients();
                    nodes[i].weights[j] = newWeight;
                }
            }
            setNewWeights(layer.prevLayer, labelNumber, learningRate);
        }
    }

    public void train(int epochs, double learningRate) {
        for(int k=0; k<epochs; k++) {
            double epochloss = 0.0;
            System.out.println("Epoch #"+(k+1));
            double correctPred = 0.0;
            for(int i=0; i<trainInput.length; i++) {
                
                // Forward Propagation
                // System.out.println("\n\nInput #"+(i+1));
                inputLayer.fill(trainInput[i]);
                // System.out.println("Input Layer : ");
                // print(inputLayer);
                for(int j=0; j<hiddenLayers.length; j++) {
                    // System.out.println("\nHidden layer"+(j+1)+" : ");
                    hiddenLayers[j].fill();
                    // print(hiddenLayers[j]);
                }
                outputLayer.fill();
                // System.out.println("\nOutput Layer : ");
                // print(outputLayer);

                // Loss calculation
                double loss = calculateLoss(i);
                epochloss+=loss;
                // System.out.println("\nLoss : "+loss);

                if((outputLayer.nodes[0].activatedValue > 0.5 && trainOutput[i][0]==1)||(outputLayer.nodes[0].activatedValue < 0.5 && trainOutput[i][0]==0)) correctPred++;

                // Backward Propagation
                setGradients(outputLayer, i);
                setNewWeights(outputLayer, i, learningRate);
            }
            epochloss = epochloss/trainInput.length;
            double accuracy = correctPred/trainOutput.length;
            System.out.println("Loss : "+epochloss+" Accuracy : "+accuracy);
        }
    }

    public void predict(double input[]) {
        inputLayer.fill(input);
        for(int j=0; j<hiddenLayers.length; j++) {
            hiddenLayers[j].fill();
        }
        outputLayer.fill();
        if(outputLayer.nodes[0].activatedValue>0.5) System.out.println("Predicted value : 1");
        else System.out.println("Predicted value : 0");
    }

    public void print(Layer layer) {
        for(int i=0; i<layer.nodes.length; i++) {
            System.out.print("Node "+(i+1)+", Value : "+layer.nodes[i].value+", Activated Value : "+layer.nodes[i].activatedValue+", Weights : ");
            if(layer == inputLayer) continue;
            for(int j=0; j<layer.nodes[i].weights.length; j++) {
                System.out.print(layer.nodes[i].weights[j]+" ");
            }
        }
    }
}
