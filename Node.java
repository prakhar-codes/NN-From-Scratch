public class Node {

    double value;
    double activatedValue;
    double weights[];

    public Node() {
        value = 0.0d;
        activatedValue = 0.0d;
    }

    public double activationFunction(double valueToBeActivated) {
        //sigmoid activation function
        double activatedVal = 1.0/(1.0 + Math.exp(-valueToBeActivated));
        return activatedVal; 
    }

    public void initializeWeights() {
        // Random initialization from (0,1)
        for(int i=0; i<weights.length; i++) {
            weights[i] = Math.random();
        }
    }

    public void calculate(Layer prevLayer) {
        double sum = 0.0;
        for(int i=0; i<weights.length; i++) {
            sum+=prevLayer.nodes[i].value*weights[i];
        }
        this.value = sum;
        this.activatedValue = activationFunction(this.value);
    }

}
