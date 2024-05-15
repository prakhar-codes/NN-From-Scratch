public class Classification {
    public static void main(String[] args) {
        // Classification Task - Inside Circle

        double x_train[][] = new double[1000][2];
        double y_train[][] = new double[1000][1];
        for(int i=0; i<x_train.length; i++) {
            double x[] = {Math.sqrt(Math.PI/2)*(2*Math.random() - 1), Math.sqrt(Math.PI/2)*(2*Math.random() - 1)};
            double y[] = {(x[0]*x[0] + x[1]*x[1])<1 ? 1 : 0, (x[0]*x[0] + x[1]*x[1])<1 ? 0 : 1};
            // Inside = [1 0], Outside = [0 1]
            x_train[i] = x;
            y_train[i] = y;
        }

        System.out.println("Training ANN...");
        ANN ann = new ANN();
        ann.setInput(x_train);
        ann.setOutput(y_train, "softmax");
        ann.addDenseLayer(2, 4, "sigmoid");
        ann.train(100, 0.1, "crossentropy");

        double arr1[] = {0.5, 1}; //[0 1]
        ann.predict(arr1);
        double arr2[] = {0.5, 0.8}; //[1 0]
        ann.predict(arr2);
    }
}
