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
        int num_of_rows = input.length;
        int input_size = input[0].length;
        trainInput = new double[num_of_rows][input_size];
        addInputLayer(input_size); 
    }

    public void setOutput(double output[][]) {
        System.out.println("Adding Output...");
        int num_of_rows = output.length;
        int output_size = output[0].length;
        trainOutput = new double[num_of_rows][output_size];
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

    public void print(Layer layer, int j) {
        if (layer!=null) {
            System.out.print("Layer "+j+" : ");
            Node nodes[] = layer.nodes;
            System.out.print(nodes.length+" nodes : ");
            for(int i=0; i<nodes.length; i++) {
                System.out.print(" Node "+(i+1)+" : ");
                if(nodes[i].weights != null) {
                    for(int k=0; k<nodes[i].weights.length; k++) System.out.print(" "+nodes[i].weights[k]+ " ");
                }
            }
            System.out.println();
            print(layer.nextLayer, j+1);
        }
    }
}
