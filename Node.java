public class Node {

    double value;
    double activatedValue;
    double weights[];
    double bias;
    double gradients[];
    String activation;

    public Node(String activation) {
        value = 0.0d;
        activatedValue = 0.0d;
        this.activation = activation;
    }

    private double activationFunction(double valueToBeActivated) {
        double activatedVal = 0.0;
        if(activation.equals("sigmoid")) activatedVal = 1.0/(1.0 + Math.exp(-valueToBeActivated)); // sigmoid
        if(activation.equals("tanh")) activatedVal = Math.tanh(valueToBeActivated); // tanh
        if(activation.equals("relu")) activatedVal = Math.max(0, valueToBeActivated); // ReLU
        if(activation.equals("none")) activatedVal = valueToBeActivated;
        return activatedVal; 
    }

    private double activationDerivative(double value) {
        if(activation.equals("sigmoid")) return activationFunction(value)*(1-activationFunction(value)); // sigmoid
        else if(activation.equals("tanh"))return (1 - Math.pow(Math.tanh(value), 2)); // tanh
        else if(activation.equals("relu"))return value>0?1:0; // ReLU
        else return 1;
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
        if(!activation.equals("softmax")) this.activatedValue = activationFunction(this.value);
    }

}
