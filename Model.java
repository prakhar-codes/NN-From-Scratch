public class Model {
    public static void main(String[] args) {
        double x_train[][] = {{0,0,0}, {0,0,1},{0,1,0},{0,1,1}, {1,0,1},{1,1,0},{1,1,1},{1,0,0}};
        double y_train[][] = {{0},{1},{1},{0},{0},{0},{1},{1}};
        // double x_train[][] = {{0,0,1}, {0,1,1},{1,0,1},{0,1,1}};
        // double y_train[][] = {{0},{1},{1},{0}};
        // double x_train[][] = {{0,0}, {0,1},{1,0},{1,1}};
        // double y_train[][] = {{0},{1},{1},{0}};
        System.out.println("Training ANN...");
        ANN ann = new ANN();
        ann.setInput(x_train);
        ann.setOutput(y_train);
        ann.addDenseLayer(4, 32);
        ann.train(1000, 1.0);
        double arr[] = {1,0,0}; //1
        ann.predict(arr);
    }
}
