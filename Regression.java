public class Regression {
    public static void main(String[] args) {
        // Regression Task - On a paraboloid

        double x_train[][] = new double[1000][2];
        double y_train[][] = new double[1000][1];
        for(int i=0; i<x_train.length; i++) {
            double x[] = {(2*Math.random() - 1), (2*Math.random() - 1)};
            double y[] = {x[0]*x[0] + x[1]*x[1]};
            // y = x1^2 + x2^2
            x_train[i] = x;
            y_train[i] = y;
        }

        System.out.println("Training ANN...");
        ANN ann = new ANN();
        ann.setInput(x_train);
        ann.setOutput(y_train, "none");
        ann.addDenseLayer(2, 4, "sigmoid");
        ann.train(1000, 0.01, "meansquared");

        double arr1[] = {0.5, 1}; //1.25
        ann.predict(arr1);
        double arr2[] = {0.5, 0.8}; //0.89
        ann.predict(arr2);
    }
}
