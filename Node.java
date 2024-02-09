public class Node {

    double value;
    double activatedValue;
    double weights[];
    double gradients[];

    public Node() {
        value = 0.0d;
        activatedValue = 0.0d;
    }

    private double activationFunction(double valueToBeActivated) {
        //sigmoid activation function
        double activatedVal = 1.0/(1.0 + Math.exp(-valueToBeActivated));
        return activatedVal; 
    }

    private double activationDerivative(double value) {
        return activationFunction(value)*(1-activationFunction(value));
    }

    public double getActivationDerivative() {
        return activationDerivative(activatedValue);
    } 

    public double getSumOfGradients() {
        double sum=0.0;
        for(int i=0; i<gradients.length; i++) {
            sum+=gradients[i];
        }
        return sum;
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
            sum+=prevLayer.nodes[i].activatedValue*weights[i];
        }
        this.value = sum;
        this.activatedValue = activationFunction(this.value);
    }

}
