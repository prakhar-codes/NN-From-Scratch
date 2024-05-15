public class Layer {
    
    Node nodes[];
    Layer nextLayer;
    Layer prevLayer;

    public Layer() {
    }

    public void createNodes(int size, String activation) {
        nodes = new Node[size];
        for(int i=0; i<size; i++) {
            nodes[i] = new Node(activation);
        }
    }

    public void fill(double data[]) {
        for(int i=0; i<this.nodes.length; i++) {
            this.nodes[i].value = data[i];
            this.nodes[i].activatedValue = data[i];
        }
    }

    public void fill(boolean isOutput) {
        for(int i=0; i<nodes.length; i++) {
            nodes[i].calculate(this.prevLayer);
        }
        if(isOutput) {
            double sum = 0.0;
            for(int i=0; i<nodes.length; i++) {
                if(nodes[i].activation.equals("softmax")) {
                    sum += Math.exp(nodes[i].value);
                }
            }
            for(int i=0; i<nodes.length; i++) {
                if(nodes[i].activation.equals("softmax")) {
                    nodes[i].activatedValue = Math.exp(nodes[i].value)/sum;
                }
            }
        }
    }
}
