public class Model {
    public static void main(String[] args) {
        // double x_train[][] = {{0,0,0}, {0,0,1},{0,1,1}, {1,0,1},{1,1,0},{1,1,1}};
        // double y_train[][] = {{0},{1},{1},{1},{1},{1}};
        // double x_train[][] = {{0,0,1}, {0,1,1},{1,0,1},{0,1,1}};
        // double y_train[][] = {{0},{1},{1},{0}};
        // double x_train[][] = {{0,0}, {0,1},{1,0},{1,1}};
        // double y_train[][] = {{0},{1},{1},{0}};
        // double x_train[][] = {{0,0,0}, {0,0,1},{0,1,1}, {1,0,1},{1,1,0},{1,1,1}};
        // double y_train[][] = {{0},{1},{1},{1},{1},{1}};

        // double x_train[][] = new double[1000][2];
        // double y_train[][] = new double[1000][1];

        // for(int i=0; i<x_train.length; i++) {
        //     double x[] = {2*Math.random(), -2*Math.random()};
        //     double y[] = {x[0]-x[1]>2 ? 0 : 1};
        //     x_train[i] = x;
        //     y_train[i] = y;
        // }

        double x_train[][] = new double[1000][2];
        double y_train[][] = new double[1000][1];
        double per = 0.0;
        for(int i=0; i<x_train.length; i++) {
            double x[] = {Math.sqrt(Math.PI/2)*(2*Math.random() - 1), Math.sqrt(Math.PI/2)*(2*Math.random() - 1)};
            double y[] = {(x[0]*x[0] + x[1]*x[1])<1 ? 1 : 0};
            if(y[0]==1) per++;
            x_train[i] = x;
            y_train[i] = y;
        }
        per = per/1000;

        System.out.println("Training ANN...");
        ANN ann = new ANN();
        ann.setInput(x_train);
        ann.setOutput(y_train);
        ann.addDenseLayer(2, 4);
        ann.train(600, 0.1);
        double arr1[] = {0.5, 1}; //0
        ann.predict(arr1);
        double arr2[] = {0.5, 0.8}; //1
        ann.predict(arr2);
        System.out.println(per*100);
    }
}
