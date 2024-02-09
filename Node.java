public class Node {

    double value;
    double activatedValue;
    int weights[];

    public Node() {
        value = 0.0d;
        activatedValue = 0.0d;
    }

    public double activationFunction(double valueToBeActivated) {
        //sigmoid activation function
        double activatedVal = 1.0/(1.0 + Math.exp(-valueToBeActivated));
        return activatedVal; 
    }

}
