public class Node {

    double value;
    double activatedValue;
    double weights[];
    double bias;
    double gradients[];

    public Node() {
        value = 0.0d;
        activatedValue = 0.0d;
    }

    private double activationFunction(double valueToBeActivated) {
        double activatedVal = 1.0/(1.0 + Math.exp(-valueToBeActivated)); // sigmoid
        //double activatedVal = Math.tanh(valueToBeActivated); // tanh
        //double activatedVal = Math.max(0, valueToBeActivated); // ReLU 
        return activatedVal; 
    }

    private double activationDerivative(double value) {
        return activationFunction(value)*(1-activationFunction(value)); // sigmoid
        //return (1 - Math.pow(Math.tanh(value), 2)); // tanh
        //return value>0?1:0; // ReLU
    }

    public double getActivationDerivative() {
        return activationDerivative(this.value);
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
            weights[i] = 2*Math.random()-1;
            // System.out.println(weights[i]);
        }
        bias = 2*Math.random()-1;
    }

    public void calculate(Layer prevLayer) {
        double sum = 0.0;
        for(int i=0; i<weights.length; i++) {
            sum+=prevLayer.nodes[i].activatedValue*weights[i];
        }
        this.value = sum + bias;
        this.activatedValue = activationFunction(this.value);
    }

}
